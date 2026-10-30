"""
자막 분석 작업 테스트 (Gemini·Chroma·임베딩 모델 없이 가짜 객체로 실행)
  python -m unittest tests.test_ingest -v      (pytest 로도 실행 가능)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ingest_core as core   # noqa: E402
import ingest_jobs as jobs   # noqa: E402


# ---------------------------------------------------------------- 가짜 객체
class FakeCollection:
    def __init__(self):
        self.rows = {}   # id -> (document, metadata)

    @staticmethod
    def _match(meta, where):
        if where is None:
            return True
        if "$and" in where:
            return all(FakeCollection._match(meta, w) for w in where["$and"])
        (key, cond), = where.items()
        if isinstance(cond, dict) and "$in" in cond:
            return meta.get(key) in cond["$in"]
        return meta.get(key) == cond

    def get(self, ids=None, where=None, include=None):
        keys = [i for i in (ids if ids is not None else self.rows) if i in self.rows]
        keys = [i for i in keys if self._match(self.rows[i][1], where)]
        return {"ids": keys, "documents": [self.rows[i][0] for i in keys],
                "metadatas": [self.rows[i][1] for i in keys]}

    def upsert(self, ids, documents, metadatas, embeddings):
        assert len(ids) == len(documents) == len(metadatas) == len(embeddings)
        for i, d, m in zip(ids, documents, metadatas):
            assert all(v is not None for v in m.values()), "Chroma metadata cannot be None"
            self.rows[i] = (d, dict(m))

    def delete(self, ids):
        for i in ids:
            self.rows.pop(i, None)


class FakeModel:
    def encode(self, texts, **kw):
        return SimpleNamespace(tolist=lambda: [[0.1, 0.2] for _ in texts])


class FakeGemini:
    """청크 본문에서 'EVENT:' 로 시작하는 줄을 사건으로 돌려줌. bad_quote=True 면 일부 인용을 틀리게"""
    def __init__(self, fail_times=0, bad_quote=False):
        self.calls = 0
        self.fail_times = fail_times
        self.bad_quote = bad_quote
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ConnectionError("temporary 503")
        body = json.loads(contents)
        events = []
        for chunk in body["chunks"]:
            for line in chunk["text"].splitlines():
                if line.startswith("EVENT:"):
                    events.append({"description_ko": line[6:].strip() + " 사건",
                                   "event_type": "REVELATION", "fact_status": "DIRECT_EVENT",
                                   "evidence": [{"document_id": chunk["document_id"], "quote": line}]})
        if self.bad_quote and events:
            events.append({"description_ko": "지어낸 사건", "event_type": "DEATH",
                           "fact_status": "DIRECT_EVENT",
                           "evidence": [{"document_id": body["chunks"][0]["document_id"], "quote": "not in source"}]})
        return SimpleNamespace(text=json.dumps({"events": events}, ensure_ascii=False))


def make_srt(n=40, events_at=(5, 22), gap_after=None):
    blocks, t = [], 0
    for i in range(1, n + 1):
        if gap_after and i == gap_after + 1:
            t += 60_000
        text = f"EVENT: secret {i}" if i in events_at else f"Line number {i}."
        s, e = t, t + 2000
        fmt = lambda ms: f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
        blocks.append(f"{i}\n{fmt(s)} --> {fmt(e)}\n{text}")
        t = e + 500
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------- SRT / 청크
class SrtTests(unittest.TestCase):
    def test_decode_cp949_and_utf8_bom(self):
        text = "1\n00:00:01,000 --> 00:00:02,000\n안녕하세요\n"
        self.assertIn("안녕하세요", core.decode_srt(text.encode("cp949")))
        self.assertIn("안녕하세요", core.decode_srt(b"\xef\xbb\xbf" + text.encode("utf-8")))
        self.assertIn("안녕하세요", core.decode_srt(text.encode("utf-16")))

    def test_parse_skips_empty_and_strips_tags(self):
        srt = ("1\n00:00:01,000 --> 00:00:02,000\n<i>Hello</i> &amp; bye\n\n"
               "2\n00:00:03.000 --> 00:00:04.000 X1:10\n<font color=red></font>\n\n"
               "3\n00:00:05,000 --> 00:00:06,000\nok")
        rows, skipped = core.parse_srt_text(srt)
        self.assertEqual([r["text"] for r in rows], ["Hello & bye", "ok"])
        self.assertEqual(skipped, 1)

    def test_parse_rejects_bad_order(self):
        srt = "1\n00:00:05,000 --> 00:00:06,000\na\n\n2\n00:00:01,000 --> 00:00:02,000\nb"
        with self.assertRaises(core.IngestError):
            core.parse_srt_text(srt)

    def test_chunks_use_spring_ids_and_gap(self):
        rows, _ = core.parse_srt_text(make_srt(n=20, gap_after=10))
        chunks = core.build_chunks(rows, drama_id=3, episode_id=57, episode=2, kb_version="kb1",
                                   language="en", source_name="x.srt")
        self.assertTrue(chunks[0]["id"].startswith("d3-e57-kb1-c0001"))
        # 60초 공백에서 청크가 끊겨야 함
        self.assertTrue(any(c["metadata"]["subtitle_end"] == 10 for c in chunks))

    def test_batch_ranges(self):
        self.assertEqual(core.batch_ranges(10, 6, 2), [(0, 6), (4, 10)])
        self.assertEqual(core.batch_ranges(3, 6, 2), [(0, 3)])


# ---------------------------------------------------------------- 추출
class ExtractTests(unittest.TestCase):
    def chunks(self):
        rows, _ = core.parse_srt_text(make_srt(n=200, events_at=(5, 22, 150)))
        return core.build_chunks(rows, drama_id=1, episode_id=1, episode=1, kb_version="kb1",
                                 language="en", source_name="x.srt")

    def test_bad_quote_drops_only_that_event(self):
        state = core.extract_events(FakeGemini(bad_quote=True), "m", self.chunks(), episode=1,
                                    language="en", state=core.new_extract_state(), sleep=lambda s: None)
        descriptions = [e["description"] for e in state["events"]]
        self.assertIn("secret 5 사건", descriptions)
        self.assertNotIn("지어낸 사건", descriptions)
        self.assertTrue(state["rejected"])
        # 겹치는 배치에서 같은 사건이 다시 나와도 한 번만
        self.assertEqual(len(descriptions), len(set(descriptions)))

    def test_transient_error_retried(self):
        g = FakeGemini(fail_times=2)
        state = core.extract_events(g, "m", self.chunks(), episode=1, language="en",
                                    state=core.new_extract_state(), sleep=lambda s: None)
        self.assertEqual(len(state["events"]), 3)

    def test_persistent_error_raises(self):
        with self.assertRaises(RuntimeError):
            core.extract_events(FakeGemini(fail_times=99), "m", self.chunks(), episode=1, language="en",
                                state=core.new_extract_state(), sleep=lambda s: None)


# ---------------------------------------------------------------- 색인
class IndexTests(unittest.TestCase):
    def ev(self, text):
        return {"description": text, "event_type": "DEATH", "fact_status": "DIRECT_EVENT",
                "source_chunk_ids": ["c1"], "evidence": [{"document_id": "c1", "quote": "q"}]}

    def test_replace_keeps_reviewed_and_manual(self):
        col = FakeCollection()
        base = {"drama_id": 1, "source_episode": 2, "kb_version": "kb1"}
        col.rows = {
            "old-demo": ("옛 데모 사건", dict(base, verification_status="PENDING")),   # origin 없음 = 예전 데이터
            "verified": ("A가 죽었다", dict(base, verification_status="VERIFIED", origin="AUTO")),
            "manual": ("수동 사건", dict(base, verification_status="PENDING", origin="MANUAL")),
            "other-ep": ("다른 회차", dict(base, source_episode=3, verification_status="PENDING")),
        }
        with self.assertRaises(core.IngestError):
            core.index_extracted_events(col, FakeModel(), [self.ev("새 사건")], drama_id=1, episode_id=9,
                                        episode=2, kb_version="kb1", run_tag="abc123", replace=False)
        result = core.index_extracted_events(
            col, FakeModel(), [self.ev("새 사건"), self.ev("A가  죽었다")], drama_id=1, episode_id=9,
            episode=2, kb_version="kb1", run_tag="abc123", replace=True)
        self.assertEqual(result["indexed"], 1)
        self.assertEqual(result["kept"], 2)
        self.assertEqual(result["skipped_as_duplicate"], 1)
        self.assertNotIn("old-demo", col.rows)
        self.assertIn("other-ep", col.rows)
        new_id = "d1-e9-kb1-abc123-f0001"
        meta = col.rows[new_id][1]
        self.assertEqual(meta["first_reveal_episode"], 0)
        self.assertEqual(meta["origin"], "AUTO")
        self.assertEqual(meta["episode_id"], 9)


# ---------------------------------------------------------------- 작업 전체 흐름
class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self._orig = (jobs.JOBS_DIR, core.KB_DIR, jobs._executor, jobs._provider)
        jobs.JOBS_DIR = root / "ingest_jobs"
        core.KB_DIR = root / "kb"
        self.submitted = []
        jobs._executor = SimpleNamespace(submit=lambda fn, *a: self.submitted.append((fn, a)))
        self.col = FakeCollection()
        self.gemini = FakeGemini()
        jobs.bind(lambda: {"col": self.col, "model": FakeModel(), "client": self.gemini,
                           "gemini_model": "fake-model"})
        core_sleep = core.time.sleep
        core.time.sleep = lambda s: None
        self.addCleanup(setattr, core.time, "sleep", core_sleep)

    def tearDown(self):
        jobs.JOBS_DIR, core.KB_DIR, jobs._executor, jobs._provider = self._orig
        self.tmp.cleanup()

    def run_pending(self):
        while self.submitted:
            fn, args = self.submitted.pop(0)
            fn(*args)

    def create(self, **kw):
        params = dict(drama_id=1, episode_id=11, episode=1, original_name="ep01.srt")
        params.update(kw)
        return jobs.create_job(make_srt(n=200, events_at=(5, 22, 150)).encode("cp949"), **params)

    def test_full_flow(self):
        job = self.create()
        self.assertEqual(job["status"], jobs.STATUS_QUEUED)
        self.run_pending()
        job = jobs.get_job(job["job_id"])
        self.assertEqual(job["status"], jobs.STATUS_DONE, job.get("error"))
        self.assertEqual(job["event_count"], 3)
        self.assertEqual(job["progress"], 1.0)
        ep_dir = core.episode_dir(1, 1)
        for name in ("source.srt", "chunks.json", "events_draft.json"):
            self.assertTrue((ep_dir / name).is_file(), name)
        self.assertEqual(len(self.col.rows), 3)

    def test_conflicts(self):
        job = self.create()
        with self.assertRaises(jobs.JobConflict):       # 진행 중 작업 있음
            self.create()
        self.run_pending()
        with self.assertRaises(jobs.JobConflict):       # 이미 사건 있음
            self.create()
        again = self.create(replace=True)
        self.run_pending()
        self.assertEqual(jobs.get_job(again["job_id"])["status"], jobs.STATUS_DONE)
        self.assertEqual(len(self.col.rows), 3)          # 교체되어 중복 없음
        self.assertNotEqual(job["job_id"], again["job_id"])

    def test_bad_file_rejected_immediately(self):
        with self.assertRaises(core.IngestError):
            jobs.create_job(b"not a subtitle", drama_id=1, episode_id=1, episode=1)
        self.assertEqual(self.submitted, [])

    def test_resume_from_checkpoint_after_failure(self):
        job = self.create()
        calls = {"n": 0}
        real = self.gemini.models.generate_content

        def flaky(**kw):     # 두 번째 배치부터 계속 실패 → 작업 FAILED
            calls["n"] += 1
            if calls["n"] > 1:
                raise ConnectionError("down")
            return real(**kw)
        self.gemini.models.generate_content = flaky
        self.run_pending()
        failed = jobs.get_job(job["job_id"])
        self.assertEqual(failed["status"], jobs.STATUS_FAILED)
        self.assertEqual(failed["done_batches"], 1)

        jobs.recover_interrupted_jobs()   # FAILED 는 그대로
        self.gemini.models.generate_content = real
        self.gemini.calls = 0
        jobs.retry_job(job["job_id"])
        self.run_pending()
        done = jobs.get_job(job["job_id"])
        self.assertEqual(done["status"], jobs.STATUS_DONE, done.get("error"))
        self.assertEqual(self.gemini.calls, done["total_batches"] - 1)   # 첫 배치는 다시 안 부름
        self.assertEqual(done["event_count"], 3)

    def test_recover_marks_active_failed(self):
        job = self.create()               # QUEUED 상태로 서버가 꺼졌다고 가정
        self.submitted.clear()
        jobs.recover_interrupted_jobs()
        self.assertEqual(jobs.get_job(job["job_id"])["status"], jobs.STATUS_FAILED)


if __name__ == "__main__":
    unittest.main()
