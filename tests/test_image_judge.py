"""
게시글 이미지 판정 테스트 (Gemini·Chroma 없이)
  python -m unittest tests.test_image_judge -v
"""
import json
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import fastapi  # noqa: F401
except ImportError:      # 테스트 환경에 FastAPI 가 없을 때 라우터 데코레이터만 흉내
    fa = types.ModuleType("fastapi")

    class _Router:
        def __init__(self, *a, **k):
            pass

        def post(self, *a, **k):
            return lambda f: f

    fa.APIRouter = _Router
    fa.File = fa.Form = fa.Header = lambda *a, **k: None
    fa.UploadFile = object
    fa.HTTPException = type("HTTPException", (Exception,), {})
    sys.modules["fastapi"] = fa

import event_store as base  # noqa: E402
import image_judge as ij  # noqa: E402


def ev(eid, ep, mode="SUBTITLE_VIDEO", status="PENDING", first=0, visual=None, drama=1):
    return eid, f"사건 {eid}", {
        "drama_id": drama, "kb_version": "kb1", "source_episode": ep, "source_mode": mode,
        "verification_status": status, "first_reveal_episode": first,
        "visual_evidence_json": json.dumps(visual or [], ensure_ascii=False),
    }


class FakeCol:
    def __init__(self, rows):
        self.rows = rows

    def _match(self, meta, where):
        for cond in where.get("$and", [where]):
            (k, v), = cond.items()
            if isinstance(v, dict):
                if "$in" in v and meta.get(k) not in v["$in"]:
                    return False
                if "$ne" in v and meta.get(k) == v["$ne"]:
                    return False
            elif meta.get(k) != v:
                return False
        return True

    def _sel(self, where):
        return [r for r in self.rows if self._match(r[2], where)]

    def get(self, where=None, limit=None, include=None):
        rows = self._sel(where)[: limit or None]
        return {"ids": [r[0] for r in rows]}

    def query(self, query_embeddings, where, n_results, include):
        rows = self._sel(where)[:n_results]
        return {"ids": [[r[0] for r in rows]], "documents": [[r[1] for r in rows]],
                "metadatas": [[r[2] for r in rows]]}


class FakeModel:
    def encode(self, texts, **k):
        class V(list):
            def tolist(self):
                return [[0.1, 0.2]]
        return V()


class ImageJudgeTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.answers = []
        ij._image_part = lambda data, mime: ("IMAGE", mime)

        def fake_gemini(client, model, parts, system, schema, timeout):
            self.calls.append((system[:20], parts))
            nxt = self.answers.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        ij._gemini = fake_gemini

    def run_judge(self, rows, episodes=(1, 2, 3)):
        return ij.judge_image(b"img", "image/png", 1, "kb1", list(episodes), col=FakeCol(rows),
                              model=FakeModel(), client=None, gemini_model="fake", base=base, debug=True)

    def test_no_video_events_skips_without_gemini(self):
        r = self.run_judge([ev("a", 1, mode="SUBTITLE"), ev("b", 2, mode=None)])
        self.assertEqual(r["analysis_verdict"], "SKIPPED_NO_VIDEO_EVENTS")
        self.assertEqual(self.calls, [])

    def test_video_events_of_other_drama_do_not_count(self):
        r = self.run_judge([ev("a", 1, mode="SUBTITLE"), ev("x", 1, drama=2)])
        self.assertEqual(r["analysis_verdict"], "SKIPPED_NO_VIDEO_EVENTS")

    def test_not_a_scene_is_skipped_after_describe(self):
        self.answers = [{"is_show_scene": False, "description": "음식 사진", "visible_text": ""}]
        r = self.run_judge([ev("a", 1)])
        self.assertEqual(r["analysis_verdict"], "SKIPPED_NOT_SCENE")
        self.assertEqual(len(self.calls), 1)

    def test_match_maps_alias_and_uses_latest_episode(self):
        rows = [ev("d1-a", 1, visual=[{"timestamp": "01:00", "description": "CCTV 화면"}]),
                ev("d1-b", 3, status="VERIFIED", first=2), ev("d1-c", 2, mode="SUBTITLE")]
        self.answers = [
            {"is_show_scene": True, "description": "CCTV 화면에 그림자", "visible_text": ""},
            {"verdict": "MATCH", "event_ids": ["e1", "E 2"], "reason": "그림자 장면"},
        ]
        r = self.run_judge(rows)
        self.assertEqual(r["analysis_verdict"], "MATCH")
        self.assertEqual(r["matched_event_ids"], ["d1-a", "d1-b"])
        self.assertEqual(r["matched_episodes"], [1, 2])          # d1-b 는 검수된 최초 공개 2화
        self.assertEqual(r["required_episode_number"], 2)
        self.assertEqual(r["episode_basis"], base.UNVERIFIED_BASIS)
        # 화면 근거가 Gemini 후보에 포함되고, 사건 ID 대신 짧은 번호로 전달
        payload = json.loads(self.calls[1][1][1])
        self.assertEqual(payload["candidates"][0]["event_id"], "E1")
        self.assertEqual(payload["candidates"][0]["visual_evidence"], ["CCTV 화면"])

    def test_match_without_known_ids_becomes_uncertain(self):
        self.answers = [{"is_show_scene": True, "description": "장면", "visible_text": ""},
                        {"verdict": "MATCH", "event_ids": ["E9"], "reason": "x"}]
        r = self.run_judge([ev("a", 1)])
        self.assertEqual(r["analysis_verdict"], "UNCERTAIN")
        self.assertIsNone(r["required_episode_number"])

    def test_no_match(self):
        self.answers = [{"is_show_scene": True, "description": "얼굴", "visible_text": ""},
                        {"verdict": "NO_MATCH", "event_ids": [], "reason": "인물 얼굴"}]
        r = self.run_judge([ev("a", 1)])
        self.assertEqual(r["analysis_verdict"], "NO_MATCH")
        self.assertEqual(r["matched_event_ids"], [])

    def test_gemini_error_is_error(self):
        self.answers = [RuntimeError("429 RESOURCE_EXHAUSTED")]
        r = self.run_judge([ev("a", 1)])
        self.assertEqual(r["analysis_verdict"], "ERROR")
        self.assertIn("RESOURCE_EXHAUSTED", r["error"])

    def test_sniff_and_episodes(self):
        self.assertEqual(ij._sniff_mime(b"\x89PNG\r\n\x1a\nxxxx"), "image/png")
        self.assertEqual(ij._sniff_mime(b"\xff\xd8\xff\xe0"), "image/jpeg")
        self.assertIsNone(ij._sniff_mime(b"GIF89a"))
        self.assertEqual(ij._parse_episodes("3,1, 2"), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
