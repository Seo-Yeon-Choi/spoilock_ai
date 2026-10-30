"""
영상 → SRT → 사건 추출 연결 테스트 (Whisper·Gemini·Chroma 없이 가짜 객체로 실행)
  python -m unittest tests.test_video -v
"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingest_core as core      # noqa: E402
import ingest_jobs as jobs      # noqa: E402
import video_pipeline as vp     # noqa: E402
from test_ingest import FakeCollection, FakeGemini, FakeModel  # noqa: E402


class FakeWhisper:
    """대사 60줄, 그중 몇 줄은 'EVENT:' (FakeGemini 가 사건으로 인식)"""

    def __init__(self, lines=60, events_at=(5, 40), fail=False):
        self.lines, self.events_at, self.fail = lines, events_at, fail
        self.calls = 0

    def transcribe(self, path, language=None, vad_filter=True, beam_size=5):
        self.calls += 1
        if self.fail:
            raise RuntimeError("decoder error")
        segs = []
        for i in range(1, self.lines + 1):
            text = f"EVENT: secret {i}" if i in self.events_at else f"Line number {i}."
            segs.append(SimpleNamespace(start=i * 3.0, end=i * 3.0 + 2.0, text=f" {text} "))
        return iter(segs), SimpleNamespace(duration=self.lines * 3.0 + 5, language="en")


class Inline:
    """executor 대신 submit 된 작업을 모아뒀다가 run() 으로 실행"""

    def __init__(self):
        self.queue = []

    def submit(self, fn, *args):
        self.queue.append((fn, args))

    def run(self):
        while self.queue:
            fn, args = self.queue.pop(0)
            fn(*args)


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._orig = (vp.JOBS_DIR, vp._executor, vp._model, vp._transcriber_factory,
                      jobs.JOBS_DIR, jobs._executor, jobs._provider, core.KB_DIR, core.time.sleep)
        vp.JOBS_DIR = root / "video_jobs"
        jobs.JOBS_DIR = root / "ingest_jobs"
        core.KB_DIR = root / "kb"
        core.time.sleep = lambda s: None
        self.video_exec, self.ingest_exec = Inline(), Inline()
        vp._executor, jobs._executor = self.video_exec, self.ingest_exec
        self.whisper = FakeWhisper()
        vp._model = None
        vp._transcriber_factory = lambda: self.whisper
        self.col, self.gemini = FakeCollection(), FakeGemini()
        jobs.bind(lambda: {"col": self.col, "model": FakeModel(), "client": self.gemini,
                           "gemini_model": "fake"})

    def tearDown(self):
        (vp.JOBS_DIR, vp._executor, vp._model, vp._transcriber_factory,
         jobs.JOBS_DIR, jobs._executor, jobs._provider, core.KB_DIR, core.time.sleep) = self._orig
        self.tmp.cleanup()

    def upload(self, **kw):
        job_id, path = vp.new_job_dir("ep03.mp4")
        path.write_bytes(b"fake video")
        params = dict(drama_id=2, episode=3, original_name="ep03.mp4", size_bytes=10,
                      language="en", episode_id=33)
        params.update(kw)
        return vp.create_job(job_id, path, **params)

    def run_all(self):
        self.video_exec.run()
        self.ingest_exec.run()

    def test_full_chain_video_to_events(self):
        job = self.upload()
        self.video_exec.run()
        mid = vp.get_job(job["job_id"])
        self.assertEqual(mid["status"], vp.STATUS_INGESTING)      # 음성 인식 끝, 사건 추출 대기
        self.assertEqual(mid["progress"], 0.5)
        self.assertTrue(Path(mid["srt_path"]).is_file())

        self.ingest_exec.run()
        done = vp.get_job(job["job_id"])
        self.assertEqual(done["status"], vp.STATUS_DONE, done.get("error"))
        self.assertEqual(done["progress"], 1.0)
        self.assertEqual(done["event_count"], 2)
        metas = [m for _, m in self.col.rows.values()]
        self.assertTrue(all(m["drama_id"] == 2 and m["source_episode"] == 3
                            and m["episode_id"] == 33 for m in metas))
        # 회차 폴더에도 최신본 저장
        self.assertTrue((core.episode_dir(2, 3) / "chunks.json").is_file())

    def test_without_episode_id_stops_at_srt(self):
        job = self.upload(episode_id=None)
        self.run_all()
        done = vp.get_job(job["job_id"])
        self.assertEqual(done["status"], vp.STATUS_SRT_READY)
        self.assertEqual(done["progress"], 1.0)
        self.assertEqual(self.col.rows, {})

    def test_existing_events_conflict_then_replace(self):
        first = self.upload()
        self.run_all()
        self.assertEqual(vp.get_job(first["job_id"])["status"], vp.STATUS_DONE)

        second = self.upload()                       # replace 없이 같은 회차
        self.run_all()
        failed = vp.get_job(second["job_id"])
        self.assertEqual(failed["status"], vp.STATUS_FAILED)
        self.assertIn("이미 사건이 등록된 회차", failed["error"])

        third = self.upload(replace=True)
        self.run_all()
        self.assertEqual(vp.get_job(third["job_id"])["status"], vp.STATUS_DONE)
        self.assertEqual(len(self.col.rows), 2)      # 교체되어 중복 없음

    def test_ingest_failure_retry_skips_transcription(self):
        real = self.gemini.models.generate_content
        calls = {"n": 0}

        def flaky(**kw):
            calls["n"] += 1
            if calls["n"] > 1:
                raise ConnectionError("down")
            return real(**kw)

        self.gemini.models.generate_content = flaky
        self.whisper.lines = 300                     # 배치가 여러 개가 되도록
        job = self.upload()
        self.run_all()
        failed = vp.get_job(job["job_id"])
        self.assertEqual(failed["status"], vp.STATUS_FAILED)
        self.assertTrue(failed["error"].startswith("사건 추출 실패"))

        self.gemini.models.generate_content = real
        vp.retry_job(job["job_id"])
        self.run_all()
        done = vp.get_job(job["job_id"])
        self.assertEqual(done["status"], vp.STATUS_DONE, done.get("error"))
        self.assertEqual(self.whisper.calls, 1)      # 음성 인식은 다시 안 함

    def test_transcription_failure_and_retry(self):
        self.whisper.fail = True
        job = self.upload()
        self.run_all()
        self.assertEqual(vp.get_job(job["job_id"])["status"], vp.STATUS_FAILED)
        self.whisper.fail = False
        vp.retry_job(job["job_id"])
        self.run_all()
        self.assertEqual(vp.get_job(job["job_id"])["status"], vp.STATUS_DONE)

    def test_no_speech_fails_clearly(self):
        self.whisper.lines = 0
        job = self.upload()
        self.run_all()
        failed = vp.get_job(job["job_id"])
        self.assertEqual(failed["status"], vp.STATUS_FAILED)
        self.assertIn("인식된 대사가 없습니다", failed["error"])

    def test_restart_between_srt_and_ingest(self):
        job = self.upload()
        self.video_exec.run()
        # 사건 추출 시작 전에 꺼진 상황 재현
        vp._write_status(job["job_id"], status=vp.STATUS_SRT_READY, ingest_job_id=None)
        self.ingest_exec.queue.clear()
        vp.recover_interrupted_jobs()
        self.assertEqual(vp.get_job(job["job_id"])["status"], vp.STATUS_FAILED)
        # 재시도하면 SRT 로 바로 시작 (음성 인식 생략)
        # 직전 ingest 작업은 QUEUED 로 남아 있으므로 먼저 서버 재시작 복구 처리
        jobs.recover_interrupted_jobs()
        vp.retry_job(job["job_id"])
        self.run_all()
        self.assertEqual(vp.get_job(job["job_id"])["status"], vp.STATUS_DONE)
        self.assertEqual(self.whisper.calls, 1)


if __name__ == "__main__":
    unittest.main()
