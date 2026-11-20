"""
채팅 문장 규칙(message_rules.py) 단위 테스트 — FastAPI·Gemini·ChromaDB 없이 실행
  python -m unittest tests.test_message_rules -v
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import message_rules as mr  # noqa: E402


def line(text, same_author=False, seconds_before=10):
    return SimpleNamespace(text=text, same_author=same_author, seconds_before=seconds_before)


def req(message, *context):
    return SimpleNamespace(message=message, context=list(context))


class StanceOverrideTest(unittest.TestCase):
    def test_fact_ending_with_guess_becomes_guess(self):
        self.assertEqual(mr._stance_override("fact", "강도현이 범인 아닐까?"), ("guess", "fact"))

    def test_guess_with_presupposition_stays_fact(self):
        # "왜" 가 있으면 사실을 전제로 깐 질문 → 추측으로 바꾸지 않음
        self.assertEqual(mr._stance_override("fact", "왜 범인 아닐까"), ("fact", None))

    def test_guess_ending_with_fact_becomes_fact(self):
        self.assertEqual(mr._stance_override("guess", "서윤하가 죽었어"), ("fact", "guess"))

    def test_future_expectation_is_not_fact(self):
        self.assertEqual(mr._stance_override("guess", "다음 화에서 터질 것 같은 느낌이야"), ("guess", None))

    def test_reaction_with_embedded_fact_becomes_fact(self):
        self.assertEqual(mr._stance_override("reaction", "결혼한다니 기대됨"), ("fact", "reaction"))

    def test_multi_sentence_is_not_overridden(self):
        self.assertEqual(mr._stance_override("fact", "서윤하가 죽었어. 근데 진짜일까"), ("fact", None))


class GuessEndTest(unittest.TestCase):
    def test_rieul_kka_is_guess(self):
        self.assertTrue(mr._is_guess_end("범인이누굴까"))
        self.assertTrue(mr._is_guess_end("다음화에나올까ㅋㅋ"))

    def test_nikka_is_not_guess(self):
        self.assertFalse(mr._is_guess_end("범인이니까"))


class ContextInfoTest(unittest.TestCase):
    def test_no_context(self):
        info = mr._context_info(req("안녕"))
        self.assertEqual(info, {"joined": None, "question": None, "payload": []})

    def test_joins_fragments_from_same_author(self):
        info = mr._context_info(req("남겼었어", line("서윤하가", True, 20), line("영상을", True, 5)))
        self.assertEqual(info["joined"], "서윤하가 영상을 남겼었어")

    def test_answer_to_other_persons_question(self):
        info = mr._context_info(req("강도현임", line("범인 누구야?", False, 30)))
        self.assertEqual(info["question"], "범인 누구야?")
        self.assertEqual(info["payload"], [{"who": "other", "text": "범인 누구야?"}])

    def test_short_ask_includes_previous_line(self):
        info = mr._context_info(req("ㅇㅇ", line("녹음파일 찍어둠", False, 40), line("ㄹㅇ?", False, 20)))
        self.assertEqual(info["question"], "녹음파일 찍어둠 → ㄹㅇ?")

    def test_question_is_not_an_answer(self):
        info = mr._context_info(req("강도현이 전남친임 ?", line("누구야?", False, 10)))
        self.assertIsNone(info["question"])

    def test_old_question_is_ignored(self):
        info = mr._context_info(req("강도현임", line("범인 누구야?", False, mr.QUESTION_GAP_S + 1)))
        self.assertIsNone(info["question"])

    def test_laugh_only_does_not_matter(self):
        ctx = mr._context_info(req("ㅋㅋㅋ", line("범인 누구야?", False, 10)))
        self.assertFalse(mr._context_may_matter("ㅋㅋㅋ", ctx))


class SplitLongMessageTest(unittest.TestCase):
    def test_short_message_is_not_split(self):
        self.assertEqual(mr._split_long_message("짧은 글"), [])

    def test_long_message_split_by_episode_and_conjunction(self):
        text = ("1화에서 서윤하가 영상을 남겼고 다들 그걸 몰랐음 그리고 "
                "3화에서 강도현이 그 영상을 찾아내서 경찰에 넘겼다 정말 충격적인 전개였음")
        parts = mr._split_long_message(text)
        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(any(p.startswith("3화") for p in parts))
        self.assertLessEqual(len(parts), 8)


if __name__ == "__main__":
    unittest.main()
