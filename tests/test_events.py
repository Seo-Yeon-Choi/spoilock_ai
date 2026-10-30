"""
사건 관리·판정 보조 로직 테스트 (Chroma·임베딩 모델 없이 가짜 객체로 실행)
  python -m unittest tests.test_events -v
"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import event_store as base    # noqa: E402
import events_admin as admin  # noqa: E402
import ingest_core as core    # noqa: E402
import ingest_jobs as jobs    # noqa: E402
from test_ingest import FakeCollection as _BaseFake, FakeModel  # noqa: E402


class FakeCollection(_BaseFake):
    """update / query / $ne 까지 지원"""

    @staticmethod
    def _match(meta, where):
        if where is not None and "$and" not in where:
            (key, cond), = where.items()
            if isinstance(cond, dict) and "$ne" in cond:
                return key in meta and meta[key] != cond["$ne"]
        if where is not None and "$and" in where:
            return all(FakeCollection._match(meta, w) for w in where["$and"])
        return _BaseFake._match(meta, where)

    def update(self, ids, metadatas=None, documents=None, embeddings=None):
        for n, i in enumerate(ids):
            doc, meta = self.rows[i]
            if documents is not None and embeddings is None:
                raise AssertionError("embedding_function=None 이면 문서 변경 시 embeddings 필요")
            self.rows[i] = (documents[n] if documents else doc,
                            dict(metadatas[n]) if metadatas else meta)

    def query(self, query_embeddings, where, n_results, include):
        keys = [i for i in self.rows if self._match(self.rows[i][1], where)][:n_results]
        return {"ids": [keys], "documents": [[self.rows[i][0] for i in keys]],
                "metadatas": [[self.rows[i][1] for i in keys]], "distances": [[0.1] * len(keys)]}


def legacy_meta(ep, **kw):
    """기존 kb1 데이터 형태 (origin·first_reveal_episode 없음)"""
    m = {"drama_id": 1, "episode_id": 100 + ep, "source_episode": ep, "kb_version": "kb1",
         "event_type": "DEATH", "fact_status": "DIRECT_EVENT", "verification_status": "PENDING",
         "evidence_json": '[{"document_id": "c1", "quote": "q"}]', "source_chunk_ids_json": '["c1"]'}
    m.update(kw)
    return m


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._orig = jobs.JOBS_DIR
        jobs.JOBS_DIR = Path(self.tmp.name) / "ingest_jobs"
        self.col = FakeCollection()
        self.col.rows = {
            "d1-e101-kb1-f0001": ("A가 B를 죽였다", legacy_meta(1)),
            "d1-e101-kb1-f0002": ("C는 사실 스파이다", legacy_meta(1)),
            "d1-e103-kb1-f0001": ("D가 돌아왔다", legacy_meta(3)),
        }
        self.model = FakeModel()

    def tearDown(self):
        jobs.JOBS_DIR = self._orig
        self.tmp.cleanup()


class VerificationTests(Base):
    def test_effective_episode(self):
        self.assertEqual(base.effective_episode(legacy_meta(3)), 3)
        self.assertEqual(base.effective_episode(legacy_meta(3, verification_status="VERIFIED",
                                                            first_reveal_episode=2)), 2)
        # 승인됐지만 회차 미지정이면 미검수 취급
        self.assertFalse(base.is_verified(legacy_meta(3, verification_status="VERIFIED",
                                                      first_reveal_episode=0)))
        self.assertTrue(base.UNVERIFIED_BASIS.startswith("UNVERIFIED"))
        self.assertFalse(base.VERIFIED_BASIS.startswith("UNVERIFIED"))

    def test_candidates_skip_rejected_and_tolerate_missing(self):
        admin.update_event(self.col, self.model, "d1-e101-kb1-f0002", {"verification_status": "REJECTED"})
        args = SimpleNamespace(message="x", drama_id=1, kb_version="kb1", episodes=[1, 2, 3, 4],
                               retrieval="vector", top_k=20)
        ids = {c["event_id"] for c in base.candidates_for(args, collection=self.col, model=self.model)}
        self.assertEqual(ids, {"d1-e101-kb1-f0001", "d1-e103-kb1-f0001"})   # 2, 4화 없어도 에러 없음
        args.retrieval = "all"
        self.assertEqual(len(base.candidates_for(args, collection=self.col, model=self.model)), 2)
        args.episodes = [9]
        self.assertEqual(base.candidates_for(args, collection=self.col, model=self.model), [])

    def test_decide_visibility_uses_first_reveal(self):
        ev = {"source_episode": 3, "verification_status": "VERIFIED", "first_reveal_episode": 1}
        self.assertEqual(base.decide_visibility(ev, 1), ("VISIBLE", 1))
        self.assertEqual(base.decide_visibility(dict(ev, verification_status="PENDING"), 1), ("BLURRED", 3))


class AdminTests(Base):
    def test_list_with_legacy_defaults(self):
        events = admin.list_events(self.col, drama_id=1, kb_version="kb1")
        self.assertEqual([e["source_episode"] for e in events], [1, 1, 3])
        self.assertEqual(events[0]["origin"], "LEGACY")
        self.assertIsNone(events[0]["first_reveal_episode"])
        self.assertEqual(events[0]["evidence"][0]["quote"], "q")
        self.assertEqual(len(admin.list_events(self.col, drama_id=1, kb_version="kb1", episode=3)), 1)

    def test_verify_requires_valid_first_reveal(self):
        eid = "d1-e103-kb1-f0001"
        with self.assertRaises(core.IngestError):   # 회차 없이 승인 불가
            admin.update_event(self.col, self.model, eid, {"verification_status": "VERIFIED"})
        with self.assertRaises(core.IngestError):   # 사건이 나온 회차보다 늦을 수 없음
            admin.update_event(self.col, self.model, eid,
                               {"verification_status": "VERIFIED", "first_reveal_episode": 4})
        out = admin.update_event(self.col, self.model, eid,
                                 {"verification_status": "VERIFIED", "first_reveal_episode": 2})
        self.assertEqual((out["verification_status"], out["first_reveal_episode"]), ("VERIFIED", 2))
        meta = self.col.rows[eid][1]
        self.assertEqual(meta["origin"], "LEGACY")          # 기존 데이터에 기본값 채움
        self.assertEqual(meta["evidence_json"], legacy_meta(3)["evidence_json"])   # 다른 필드 유지
        with self.assertRaises(core.IngestError):   # 승인 상태에서 회차를 지우면 거부
            admin.update_event(self.col, self.model, eid, {"first_reveal_episode": None})

    def test_description_change_reembeds(self):
        out = admin.update_event(self.col, self.model, "d1-e101-kb1-f0001", {"description": "  A가   B를 살해했다 "})
        self.assertEqual(out["description"], "A가 B를 살해했다")

    def test_manual_create_and_delete(self):
        ev = admin.create_manual(self.col, self.model, drama_id=1, episode_id=7, episode=2,
                                 kb_version="kb1", description="E가 결혼했다")
        self.assertEqual(ev["origin"], "MANUAL")
        self.assertTrue(core.is_protected(self.col.rows[ev["event_id"]][1]))   # 재분석해도 보존
        admin.delete_event(self.col, ev["event_id"])
        self.assertNotIn(ev["event_id"], self.col.rows)
        with self.assertRaises(admin.EventNotFound):
            admin.delete_event(self.col, ev["event_id"])

    def test_bad_fields_rejected(self):
        with self.assertRaises(core.IngestError):
            admin.update_event(self.col, self.model, "d1-e101-kb1-f0001", {"drama_id": 2})
        with self.assertRaises(core.IngestError):
            admin.update_event(self.col, self.model, "d1-e101-kb1-f0001", {"event_type": "NOPE"})

    def test_kb_episodes(self):
        admin.update_event(self.col, self.model, "d1-e103-kb1-f0001", {"verification_status": "REJECTED"})
        rows = admin.kb_episodes(self.col, drama_id=1, kb_version="kb1")
        self.assertEqual([(r["episode"], r["active"]) for r in rows], [(1, True), (3, False)])

    def test_blocked_while_ingest_running(self):
        jobs._executor, orig = SimpleNamespace(submit=lambda *a: None), jobs._executor
        jobs.bind(lambda: {"col": self.col})
        try:
            srt = "1\n00:00:01,000 --> 00:00:02,000\nhi\n".encode()
            jobs.create_job(srt, drama_id=1, episode_id=101, episode=1, replace=True)
            with self.assertRaises(admin.EventConflict):
                admin.update_event(self.col, self.model, "d1-e101-kb1-f0001", {"verification_status": "REJECTED"})
            # 다른 회차는 수정 가능
            admin.update_event(self.col, self.model, "d1-e103-kb1-f0001", {"verification_status": "REJECTED"})
        finally:
            jobs._executor = orig
            jobs._provider = None


if __name__ == "__main__":
    unittest.main()
