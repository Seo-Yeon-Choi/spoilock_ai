"""
영상 + 자막 실험 스크립트 테스트 (Gemini 없이 가짜 클라이언트로 실행)
  python -m unittest tests.test_video_srt_experiment -v
"""
import json
import sys
import tempfile
import types as pytypes
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# google-genai 가 없어도 돌도록 최소한의 types 모듈을 끼워 넣음
if "google.genai" not in sys.modules:
    class _Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    _types = pytypes.ModuleType("google.genai.types")
    _types.Part = type("Part", (_Obj,), {})
    _types.FileData = type("FileData", (_Obj,), {})
    _types.VideoMetadata = type("VideoMetadata", (_Obj,), {})
    _types.MediaResolution = SimpleNamespace(MEDIA_RESOLUTION_LOW="LOW", MEDIA_RESOLUTION_MEDIUM="MEDIUM",
                                             MEDIA_RESOLUTION_HIGH="HIGH")
    _genai = pytypes.ModuleType("google.genai")
    _genai.types = _types
    _google = sys.modules.setdefault("google", pytypes.ModuleType("google"))
    _google.genai = _genai
    sys.modules["google.genai"] = _genai
    sys.modules["google.genai.types"] = _types

import video_srt_experiment as exp   # noqa: E402
from test_ingest import make_srt      # noqa: E402


class FakeFiles:
    def __init__(self):
        self.uploaded, self.deleted = [], []

    def upload(self, file):
        self.uploaded.append(file)
        return SimpleNamespace(name=f"files/{len(self.uploaded)}", uri="gs://fake", mime_type="video/mp4",
                               state=SimpleNamespace(name="ACTIVE"))

    def get(self, name):
        raise AssertionError("already ACTIVE")

    def delete(self, name):
        self.deleted.append(name)


class FakeGemini:
    """
    자막 청크에 'EVENT:' 줄이 있으면 대사 근거 사건,
    배치마다 화면 근거 사건 1개(구간 시작 기준 00:05),
    구간 밖 시각(99:00)을 가진 사건 1개를 돌려줌
    """

    def __init__(self, fail_first=0):
        self.files = FakeFiles()
        self.calls, self.fail_first = [], fail_first
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.calls.append(contents)
        if self.fail_first > 0:
            self.fail_first -= 1
            raise ConnectionError("503")
        part, text = contents
        body = json.loads(text)
        events = []
        for c in body["subtitle_chunks"]:
            for line in c["text"].splitlines():
                if line.startswith("EVENT:"):
                    events.append({"description_ko": line[6:].strip() + " 대사 사건", "event_type": "REVELATION",
                                   "fact_status": "DIRECT_EVENT", "evidence_type": "DIALOGUE",
                                   "dialogue_evidence": [{"document_id": c["document_id"], "quote": line}],
                                   "visual_evidence": []})
        start = part.video_metadata.start_offset
        events.append({"description_ko": f"구간 {start} 화면 사건", "event_type": "MAJOR_EVENT",
                       "fact_status": "DIRECT_EVENT", "evidence_type": "VISUAL",
                       "dialogue_evidence": [], "visual_evidence": [
                           {"timestamp": "00:05", "description_ko": "로봇이 멈춘다"}]})
        events.append({"description_ko": "지어낸 사건", "event_type": "DEATH", "fact_status": "DIRECT_EVENT",
                       "evidence_type": "VISUAL", "dialogue_evidence": [{"document_id": "nope", "quote": "x"}],
                       "visual_evidence": [{"timestamp": "99:00", "description_ko": "?"}]})
        return SimpleNamespace(text=json.dumps({"events": events}, ensure_ascii=False),
                               usage_metadata=SimpleNamespace(prompt_token_count=1000, candidates_token_count=50))


class HelperTests(unittest.TestCase):
    def test_parse_mmss(self):
        self.assertEqual(exp.parse_mmss("01:05"), 65_000)
        self.assertEqual(exp.parse_mmss("1:02:03"), 3_723_000)
        self.assertIsNone(exp.parse_mmss("1:75"))
        self.assertIsNone(exp.parse_mmss("abc"))

    def test_validate_converts_relative_and_recomputes_type(self):
        batch = [{"id": "c1", "document": "Hello there\nYou left me", "metadata": {}}]
        payload = {"events": [
            {"description_ko": "a", "event_type": "DEATH", "fact_status": "DIRECT_EVENT", "evidence_type": "BOTH",
             "dialogue_evidence": [{"document_id": "c1", "quote": "You  left me"}, {"document_id": "c1", "quote": "no"}],
             "visual_evidence": [{"timestamp": "00:10", "description_ko": "본다"}]},
            {"description_ko": "b", "event_type": "DEATH", "fact_status": "DIRECT_EVENT", "evidence_type": "BOTH",
             "dialogue_evidence": [{"document_id": "c1", "quote": "Hello there"}],
             "visual_evidence": [{"timestamp": "05:00", "description_ko": "구간 밖"}]},
        ]}
        ok, bad = exp.validate_events(payload, batch, 60_000, 120_000)
        self.assertEqual(bad, [])
        self.assertEqual(ok[0]["evidence_type"], "BOTH")
        self.assertEqual(ok[0]["visual_evidence"][0]["timestamp"], "01:10")   # 60초 + 10초
        self.assertEqual(ok[0]["dropped_quotes"], 1)
        self.assertEqual(ok[1]["evidence_type"], "DIALOGUE")                  # 화면 근거가 구간 밖이라 제외
        self.assertEqual(ok[1]["claimed_evidence_type"], "BOTH")


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.srt = root / "ep.srt"
        self.srt.write_text(make_srt(n=200, events_at=(5, 150)), encoding="utf-8")
        self.video = root / "ep.mp4"
        self.video.write_bytes(b"fake")
        self.out = root / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, **kw):
        base = dict(video=str(self.video), srt=str(self.srt), episode=1, language="en",
                    characters="Thom, Celia", batch_size=6, overlap=2, clip_mode="api",
                    resolution="low", fps=None, model=None, max_batches=0, sleep=0,
                    out=str(self.out), compare=None, keep_files=False)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_full_run_report_and_cleanup(self):
        g = FakeGemini(fail_first=1)
        result = exp.run(self.args(), client=g, model="fake", sleep=lambda s: None)
        self.assertTrue(result["finished"])
        self.assertEqual(len(g.files.uploaded), 1)                 # 영상은 한 번만 업로드
        self.assertEqual(g.files.deleted, ["files/1"])             # 끝나면 삭제
        types_ = [e["evidence_type"] for e in result["events"]]
        self.assertEqual(types_.count("DIALOGUE"), 2)
        self.assertGreaterEqual(types_.count("VISUAL"), 2)
        self.assertTrue(all(r["reason"] == "no valid evidence" for r in result["rejected"]))
        # 구간 지정: 첫 배치는 0초부터, 끝은 올림
        first = g.calls[1][0].video_metadata
        self.assertEqual(first.start_offset, "0s")
        self.assertTrue(first.end_offset.endswith("s"))
        self.assertEqual(g.calls[1][0].file_data.mime_type, "video/mp4")
        report = (self.out / "report.md").read_text(encoding="utf-8")
        self.assertIn("화면만", report)
        self.assertIn("로봇이 멈춘다", report)

    def test_resume_and_compare(self):
        g = FakeGemini()
        partial = exp.run(self.args(max_batches=2), client=g, model="fake", sleep=lambda s: None)
        self.assertFalse(partial["finished"])
        self.assertEqual(partial["batches_done"], 2)

        draft = Path(self.tmp.name) / "draft.json"
        draft.write_text(json.dumps({"events": [{"description": "자막 사건 A"}]}, ensure_ascii=False),
                         encoding="utf-8")
        g2 = FakeGemini()
        done = exp.run(self.args(compare=str(draft)), client=g2, model="fake", sleep=lambda s: None)
        self.assertTrue(done["finished"])
        self.assertEqual(len(g2.calls), done["batches_total"] - 2)  # 끝난 배치는 다시 안 부름
        self.assertIn("자막 사건 A", (self.out / "report.md").read_text(encoding="utf-8"))

    def test_missing_video_fails_before_upload(self):
        g = FakeGemini()
        with self.assertRaises(SystemExit) as ctx:
            exp.run(self.args(video=str(self.video) + ".nope"), client=g, model="fake", sleep=lambda s: None)
        self.assertIn("파일을 찾을 수 없습니다", str(ctx.exception))
        self.assertEqual(g.files.uploaded, [])

    def test_rerun_after_finish_only_rebuilds_report(self):
        exp.run(self.args(), client=FakeGemini(), model="fake", sleep=lambda s: None)
        draft = Path(self.tmp.name) / "draft.json"
        draft.write_text(json.dumps({"events": [{"description": "비교용"}]}, ensure_ascii=False), encoding="utf-8")
        g = FakeGemini()
        exp.run(self.args(compare=str(draft)), client=g, model="fake", sleep=lambda s: None)
        self.assertEqual(g.files.uploaded, [])       # 다시 올리지 않음
        self.assertEqual(g.calls, [])
        self.assertIn("비교용", (self.out / "report.md").read_text(encoding="utf-8"))

    def test_missing_compare_fails_at_start(self):
        g = FakeGemini()
        with self.assertRaises(SystemExit):
            exp.run(self.args(compare="nope.json"), client=g, model="fake", sleep=lambda s: None)
        self.assertEqual(g.files.uploaded, [])

    def test_settings_change_rejected(self):
        exp.run(self.args(max_batches=1), client=FakeGemini(), model="fake", sleep=lambda s: None)
        with self.assertRaises(SystemExit):
            exp.run(self.args(batch_size=8), client=FakeGemini(), model="fake", sleep=lambda s: None)


if __name__ == "__main__":
    unittest.main()
