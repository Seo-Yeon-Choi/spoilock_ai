"""
사건 DB 칸(kb_version) 관리 도구 테스트 (Chroma 없이 가짜 컬렉션)
  python -m unittest tests.test_kb_tool -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import kb_tool  # noqa: E402
from test_events import FakeCollection as _Base  # noqa: E402


class FakeCollection(_Base):
    """embeddings 까지 저장"""

    def __init__(self):
        super().__init__()
        self.emb = {}

    def get(self, ids=None, where=None, include=None):
        data = super().get(ids=ids, where=where, include=include)
        data["embeddings"] = [self.emb.get(i) for i in data["ids"]]
        return data

    def upsert(self, ids, documents, metadatas, embeddings):
        super().upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        for i, e in zip(ids, embeddings):
            self.emb[i] = list(e)

    def delete(self, ids=None, where=None):
        if where is not None:
            ids = self.get(where=where)["ids"]
        super().delete(ids)

    @staticmethod
    def _match(meta, where):
        if where is not None and "$and" not in where:
            (key, cond), = where.items()
            if isinstance(cond, dict) and "$in" in cond:
                return meta.get(key) in cond["$in"]
        return _Base._match(meta, where)


def meta(kb, drama, ep, status="PENDING"):
    return {"drama_id": drama, "source_episode": ep, "kb_version": kb, "verification_status": status}


class KbToolTests(unittest.TestCase):
    def setUp(self):
        self.col = FakeCollection()
        rows = [
            ("d1-e101-kb1-f0001", meta("kb1", 1, 1, "VERIFIED")),
            ("d1-e101-kb1-f0002", meta("kb1", 1, 1)),
            ("d5-e15-kb1-ab12cd-f0001", meta("kb1", 5, 1)),
            ("d5-e15-kb1-m1234abcd", meta("kb1", 5, 1)),     # 수동 등록
            ("legacy-without-kb", meta("kb1", 4, 2)),         # 칸 이름이 없는 ID
        ]
        for i, m in rows:
            self.col.upsert(ids=[i], documents=[f"doc {i}"], metadatas=[m], embeddings=[[0.1, 0.2]])
        self.log = []

    def out(self, line):
        self.log.append(line)

    def test_list(self):
        kb_tool.cmd_list(self.col, out=self.out)
        text = "\n".join(self.log)
        self.assertIn("전체 사건 5건", text)
        self.assertIn("작품 1 / 1화: 2건 (PENDING 1, VERIFIED 1)", text)

    def test_copy_dry_run_changes_nothing(self):
        n = kb_tool.cmd_copy(self.col, "kb1", "local", out=self.out)
        self.assertEqual(n, 0)
        self.assertEqual(len(self.col.rows), 5)
        self.assertIn("미리보기", self.log[-1])

    def test_copy_all_keeps_source_and_metadata(self):
        n = kb_tool.cmd_copy(self.col, "kb1", "local", apply=True, out=self.out)
        self.assertEqual(n, 5)
        self.assertEqual(len(self.col.rows), 10)
        doc, m = self.col.rows["d1-e101-local-f0001"]
        self.assertEqual(m["kb_version"], "local")
        self.assertEqual(m["verification_status"], "VERIFIED")     # 검수 상태 유지
        self.assertEqual(doc, "doc d1-e101-kb1-f0001")
        self.assertEqual(self.col.emb["d1-e101-local-f0001"], [0.1, 0.2])   # 벡터 재계산 없이 복사
        self.assertIn("local:legacy-without-kb", self.col.rows)
        self.assertEqual(self.col.rows["d1-e101-kb1-f0001"][1]["kb_version"], "kb1")   # 원본 그대로
        # 다시 실행해도 중복 생성 없이 덮어씀
        kb_tool.cmd_copy(self.col, "kb1", "local", apply=True, out=self.out)
        self.assertEqual(len(self.col.rows), 10)

    def test_move_selected_drama(self):
        kb_tool.cmd_copy(self.col, "kb1", "local", dramas=[5], apply=True, remove_source=True, out=self.out)
        kbs = {i: m["kb_version"] for i, (_, m) in self.col.rows.items()}
        self.assertEqual(kbs["d5-e15-local-ab12cd-f0001"], "local")
        self.assertNotIn("d5-e15-kb1-ab12cd-f0001", kbs)
        self.assertEqual(kbs["d1-e101-kb1-f0001"], "kb1")             # 다른 작품은 그대로

    def test_delete_with_episode_filter(self):
        kb_tool.cmd_delete(self.col, "kb1", dramas=[1], episode=1, out=self.out)
        self.assertEqual(len(self.col.rows), 5)                         # 미리보기
        kb_tool.cmd_delete(self.col, "kb1", dramas=[1], episode=1, apply=True, out=self.out)
        self.assertEqual(len(self.col.rows), 3)

    def test_cli_validation(self):
        with self.assertRaises(SystemExit):
            kb_tool.main(["copy", "--from", "kb1", "--to", "kb1", "--apply"], col=self.col)
        with self.assertRaises(SystemExit):
            kb_tool.main(["delete", "--kb", "kb1", "--episode", "1"], col=self.col)
        with self.assertRaises(SystemExit):
            kb_tool.main(["delete", "--kb", "bad name!"], col=self.col)
        kb_tool.main(["move", "--from", "kb1", "--to", "local", "--drama", "4", "5", "--apply"], col=self.col)
        kbs = Counter(m["kb_version"] for _, m in self.col.rows.values())
        self.assertEqual(kbs, {"kb1": 2, "local": 3})


from collections import Counter  # noqa: E402

if __name__ == "__main__":
    unittest.main()
