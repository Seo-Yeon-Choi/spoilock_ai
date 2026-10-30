"""
자막 + 영상 화면 분석 모드 테스트 (Gemini·Chroma 없이 가짜 객체로 실행)
  python -m unittest tests.test_ingest_video_mode -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import events_admin as admin   # noqa: E402
import ingest_core as core     # noqa: E402
import ingest_jobs as jobs     # noqa: E402
import ingest_video            # noqa: E402
import video_pipeline as vp    # noqa: E402
from test_ingest import FakeCollection, FakeGemini, FakeModel, make_srt  # noqa: E402


class FakeVision:
    """
    VideoBatchExtractor 대역. 실제와 같은 검증 함수(validate_events)를 거치게 해서
    - 'EVENT:' 대사 → 대사 근거 사건
    - 배치마다 화면 근거 사건 1개 (구간 시작 + 3초)
    """
    instances = []

    def __init__(self, client, model, video_path, workdir, fail_on_batch=None):
        self.video_path, self.prepared, self.closed, self.calls = video_path, 0, 0, 0
        self.fail_on_batch = fail_on_batch
        self.usage = {"prompt_tokens": 0, "output_tokens": 0, "calls": 0}
        FakeVision.instances.append(self)

    def prepare(self):
        assert Path(self.video_path).is_file(), "작업 폴더에 영상이 있어야 함"
        self.prepared += 1

    def extract(self, episode, batch, characters):
        self.calls += 1
        if self.fail_on_batch is not None and self.calls >= self.fail_on_batch:
            raise ConnectionError("gemini down")
        start, end = ingest_video.segment_for(batch)
        events = []
        for c in batch:
            for line in c["document"].splitlines():
                if line.startswith("EVENT:"):
                    events.append({"description_ko": line[6:].strip() + " 대사 사건", "event_type": "REVELATION",
                                   "fact_status": "DIRECT_EVENT", "evidence_type": "DIALOGUE",
                                   "dialogue_evidence": [{"document_id": c["id"], "quote": line}],
                                   "visual_evidence": []})
        who = characters[0] if characters else "누군가"
        events.append({"description_ko": f"{who}가 {ingest_video.mmss(start)} 장면에서 로봇을 멈춘다",
                       "event_type": "MAJOR_EVENT", "fact_status": "DIRECT_EVENT", "evidence_type": "VISUAL",
                       "dialogue_evidence": [], "visual_evidence": [
                           {"timestamp": "00:03", "description_ko": "로봇이 멈춘다"}]})
        return ingest_video.validate_events({"events": events}, batch, start, end), (start, end)

    def close(self, keep=False):
        self.closed += 1


class VideoModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._orig = (jobs.JOBS_DIR, jobs._executor, jobs._provider, jobs._video_extractor_factory,
                      jobs.VIDEO_BATCH_SLEEP, core.KB_DIR, core.time.sleep,
                      vp.JOBS_DIR, vp._executor, vp._model, vp._transcriber_factory)
        jobs.JOBS_DIR, core.KB_DIR = root / "ingest_jobs", root / "kb"
        core.time.sleep = lambda s: None
        jobs.VIDEO_BATCH_SLEEP = 0
        self.queue = []
        jobs._executor = SimpleNamespace(submit=lambda fn, *a: self.queue.append((fn, a)))
        self.col, self.gemini = FakeCollection(), FakeGemini()
        jobs.bind(lambda: {"col": self.col, "model": FakeModel(), "client": self.gemini,
                           "gemini_model": "fake"})
        self.fail_on_batch = None
        FakeVision.instances = []
        jobs._video_extractor_factory = lambda c, m, v, w: FakeVision(c, m, v, w, self.fail_on_batch)
        self.root = root

    def tearDown(self):
        (jobs.JOBS_DIR, jobs._executor, jobs._provider, jobs._video_extractor_factory,
         jobs.VIDEO_BATCH_SLEEP, core.KB_DIR, core.time.sleep,
         vp.JOBS_DIR, vp._executor, vp._model, vp._transcriber_factory) = self._orig
        self.tmp.cleanup()

    def run_queue(self):
        while self.queue:
            fn, args = self.queue.pop(0)
            fn(*args)

    def new_video(self):
        path = self.root / "incoming.mp4"
        path.write_bytes(b"fake video")
        return path

    def create(self, video=True, **kw):
        params = dict(drama_id=4, episode_id=14, episode=1, original_name="tos.srt",
                      characters=["Thom", "Celia"])
        params.update(kw)
        return jobs.create_job(make_srt(n=200, events_at=(5, 150)).encode(),
                               video_path=self.new_video() if video else None, **params)

    def metas(self):
        return [m for _, m in self.col.rows.values()]

    def test_subtitle_plus_video_indexes_visual_events(self):
        job = self.create()
        self.assertEqual(job["source_mode"], "SUBTITLE_VIDEO")
        self.assertFalse((self.root / "incoming.mp4").exists())            # 작업 폴더로 이동
        self.run_queue()
        done = jobs.get_job(job["job_id"])
        self.assertEqual(done["status"], jobs.STATUS_DONE, done.get("error"))
        vision = FakeVision.instances[0]
        self.assertEqual((vision.prepared, vision.closed), (1, 1))           # 한 번 올리고 정리
        self.assertEqual(vision.calls, done["total_batches"])
        self.assertFalse((jobs.JOBS_DIR / job["job_id"] / done["video_file"]).exists())  # 끝나면 삭제

        metas = self.metas()
        self.assertTrue(all(m["source_mode"] == "SUBTITLE_VIDEO" for m in metas))
        types_ = sorted(m["evidence_type"] for m in metas)
        self.assertEqual(types_.count("DIALOGUE"), 2)
        visual = [m for m in metas if m["evidence_type"] == "VISUAL"]
        self.assertTrue(visual)
        self.assertEqual(json.loads(visual[0]["evidence_json"]), [])         # 대사 인용 없음
        ve = json.loads(visual[0]["visual_evidence_json"])
        self.assertRegex(ve[0]["timestamp"], r"^\d{2}:\d{2}$")
        self.assertEqual(ve[0]["description"], "로봇이 멈춘다")
        # 인물 이름이 분석기에 전달됨
        self.assertTrue(any("Thom가" in d for d, _ in self.col.rows.values()))

        # 관리자 API 응답에도 노출
        ev = admin.list_events(self.col, drama_id=4, kb_version="kb1", episode=1)
        vis = [e for e in ev if e["evidence_type"] == "VISUAL"][0]
        self.assertEqual(vis["source_mode"], "SUBTITLE_VIDEO")
        self.assertEqual(vis["visual_evidence"][0]["description"], "로봇이 멈춘다")

    def test_subtitle_only_unchanged(self):
        job = self.create(video=False)
        self.assertEqual(job["source_mode"], "SUBTITLE")
        self.run_queue()
        self.assertEqual(jobs.get_job(job["job_id"])["status"], jobs.STATUS_DONE)
        self.assertEqual(FakeVision.instances, [])
        metas = self.metas()
        self.assertTrue(all(m["source_mode"] == "SUBTITLE" and m["evidence_type"] == "DIALOGUE" for m in metas))
        self.assertTrue(all(json.loads(m["visual_evidence_json"]) == [] for m in metas))

    def test_failure_keeps_video_and_retry_resumes(self):
        self.fail_on_batch = 2
        job = self.create()
        self.run_queue()
        failed = jobs.get_job(job["job_id"])
        self.assertEqual(failed["status"], jobs.STATUS_FAILED)
        self.assertEqual(failed["done_batches"], 1)
        self.assertTrue((jobs.JOBS_DIR / job["job_id"] / failed["video_file"]).exists())   # 재시도용 보관
        self.assertEqual(FakeVision.instances[0].closed, 1)                 # 실패해도 업로드 정리

        self.fail_on_batch = None
        jobs.retry_job(job["job_id"])
        self.run_queue()
        done = jobs.get_job(job["job_id"])
        self.assertEqual(done["status"], jobs.STATUS_DONE, done.get("error"))
        self.assertEqual(FakeVision.instances[1].calls, done["total_batches"] - 1)   # 끝난 배치는 건너뜀

    def test_conflict_does_not_consume_video(self):
        self.create(video=False)
        self.run_queue()
        video = self.new_video()
        with self.assertRaises(jobs.JobConflict):
            jobs.create_job(make_srt(n=50).encode(), drama_id=4, episode_id=14, episode=1, video_path=video)
        self.assertTrue(video.exists())          # 호출한 쪽(API)이 정리할 수 있게 남김

    def test_video_stt_marks_source_mode(self):
        """영상만 올려 음성 인식으로 만든 사건은 VIDEO_STT 로 표시"""
        from test_video import FakeWhisper, Inline
        vp.JOBS_DIR = self.root / "video_jobs"
        vp._executor = Inline()
        vp._model = None
        vp._transcriber_factory = lambda: FakeWhisper()
        vid, path = vp.new_job_dir("ep.mp4")
        path.write_bytes(b"x")
        vp.create_job(vid, path, drama_id=5, episode=1, original_name="ep.mp4", size_bytes=1,
                      language="en", episode_id=50)
        vp._executor.run()
        self.run_queue()
        self.assertEqual(vp.get_job(vid)["status"], vp.STATUS_DONE)
        self.assertTrue(all(m["source_mode"] == "VIDEO_STT" for m in self.metas()))


    def test_kb_version_separates_environments(self):
        """로컬(local)과 배포본(kb1)이 같은 작품 번호를 써도 사건·파일이 섞이지 않음"""
        jobs.create_job(make_srt(n=60, events_at=(5,)).encode(), drama_id=6, episode_id=60, episode=1)
        self.run_queue()
        local = jobs.create_job(make_srt(n=60, events_at=(5,)).encode(), drama_id=6, episode_id=99,
                                episode=1, kb_version="local")       # replace 없이도 충돌 없음
        self.run_queue()
        self.assertEqual(jobs.get_job(local["job_id"])["status"], jobs.STATUS_DONE)
        kbs = sorted(m["kb_version"] for m in self.metas())
        self.assertEqual(kbs, ["kb1", "local"])
        self.assertTrue((core.episode_dir(6, 1, "kb1") / "chunks.json").is_file())
        self.assertTrue((core.episode_dir(6, 1, "local") / "chunks.json").is_file())

    def test_video_stt_uses_requested_kb_version(self):
        from test_video import FakeWhisper, Inline
        vp.JOBS_DIR = self.root / "video_jobs"
        vp._executor = Inline()
        vp._model = None
        vp._transcriber_factory = lambda: FakeWhisper()
        vid, path = vp.new_job_dir("ep.mp4")
        path.write_bytes(b"x")
        vp.create_job(vid, path, drama_id=7, episode=1, original_name="ep.mp4", size_bytes=1,
                      language="en", episode_id=70, kb_version="local")
        vp._executor.run()
        self.run_queue()
        self.assertEqual(vp.get_job(vid)["status"], vp.STATUS_DONE)
        self.assertTrue(all(m["kb_version"] == "local" for m in self.metas()))


if __name__ == "__main__":
    unittest.main()
