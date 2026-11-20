"""채팅 메시지 문장 규칙 (스포일러 판정 보조).

spoiler_server.py 에서 분리한 순수 함수 모음. FastAPI·Gemini·ChromaDB 에 의존하지 않아
단위 테스트로 바로 검증할 수 있다.
  - 대화 맥락 해석: 나눠 보낸 조각 합치기, 앞 질문에 대한 답 찾기
  - stance 보정: 문장 끝맺음으로 fact / guess 를 바로잡음
  - 장문 분할: 사건 검색용 구간 나누기
"""
import os
import re

# 장문 메시지: 이 길이 이상이면 문장 단위로 나눠 사건을 추가 검색
LONG_MESSAGE_CHARS = int(os.getenv("LONG_MESSAGE_CHARS", "60"))
# 대화 맥락: 같은 작성자가 이 간격(초) 안에 연달아 보낸 짧은 조각은 한 문장으로 이어 붙임
FRAGMENT_GAP_S = int(os.getenv("FRAGMENT_GAP_S", "30"))
FRAGMENT_MAX_CHARS = int(os.getenv("FRAGMENT_MAX_CHARS", "25"))   # 조각 1개 최대 길이(공백 제외)
QUESTION_GAP_S = int(os.getenv("QUESTION_GAP_S", "120"))          # 이 시간 안의 질문에 대한 답으로 봄

_LAUGH_ONLY_RE = re.compile(r"^[ㅋㅎㅠㅜㄷ\s!?.~^;]+$")   # 웃음·울음만 → 맥락이 있어도 정보 없음
_ASK_RE = re.compile(r"[?？]|(누가|누구|뭐|뭔|뭘|무슨|언제|어디|어떻게|몇)")
# 다른 사람의 짧은 맞장구("ㅇㅇ", "맞음ㅋ", "ㄴㄴ") → 질문을 찾을 때 건너뜀 (같은 질문에 여러 명이 답하는 경우)
_SHORT_REPLY_MAX = 4
# "ㄹㅇ?", "진짜?", "정말?" 처럼 짧게 되묻는 질문 → 그 앞 말이 실제 질문 내용
_SHORT_ASK_MAX = 5
# 메시지가 직접 말한 회차 ("3화에서도", "1화에서 말했듯이"). "몇 화" 는 제외
_EPISODE_MENTION_RE = re.compile(r"(?<![0-9몇])(\d{1,2})\s*화")


def _nospace_len(t):
    return len(re.sub(r"\s", "", t))


def _context_info(req):
    """
    반환 dict
      joined   : 같은 작성자가 방금 나눠 보낸 앞 조각 + 지금 메시지 ("서윤하가 영상을 남겼었어"), 없으면 None
      question : 다른 사람이 최근에 한 질문 (지금 메시지가 그 답일 수 있음), 없으면 None
      payload  : Gemini 에 보낼 맥락 [{"who", "text"}]
    """
    lines = [c for c in req.context if c.text and c.text.strip()]
    info = {"joined": None, "question": None, "payload": []}
    if not lines:
        return info

    # 나눠 보낸 문장: 끝에서부터 같은 작성자의 짧은 메시지가 FRAGMENT_GAP_S 간격 이내로 이어지는 동안
    frags, later = [], 0
    for c in reversed(lines):
        t = c.text.strip()
        if (not c.same_author or c.seconds_before - later > FRAGMENT_GAP_S
                or _nospace_len(t) > FRAGMENT_MAX_CHARS or _LAUGH_ONLY_RE.match(t)):
            break
        frags.append(t)
        later = c.seconds_before
        if len(frags) >= 5:
            break
    if frags and _nospace_len(req.message) <= FRAGMENT_MAX_CHARS:
        info["joined"] = " ".join(list(reversed(frags)) + [req.message.strip()])

    # 질문 → 답: 다른 사람의 최근 질문 1개 (작성자 본인 메시지, 다른 사람의 짧은 맞장구는 건너뜀)
    qi = None
    # 지금 메시지가 질문이면 답이 아님 ("강도현이 전남친임 ?" 은 앞 질문의 답으로 보지 않음)
    is_question = "?" in req.message or "？" in req.message
    if not is_question:
        for i in range(len(lines) - 1, -1, -1):
            c = lines[i]
            t = c.text.strip()
            if c.seconds_before > QUESTION_GAP_S:
                break
            if c.same_author:
                continue
            if _ASK_RE.search(t):
                qi = i
                break
            if _nospace_len(t) <= _SHORT_REPLY_MAX or _LAUGH_ONLY_RE.match(t):
                continue      # "ㅇㅇ", "맞음ㅋ" → 같은 질문에 먼저 답한 사람
            break             # 질문이 아닌 일반 문장 → 그 앞 질문은 이미 지나간 대화
    if qi is not None:
        info["question"] = lines[qi].text.strip()
        # "ㄹㅇ?", "진짜?" 는 바로 앞 말을 되묻는 것 → 앞 말까지 묶어서 질문으로 봄
        #  ("녹음파일 찍어둠" → "ㄹㅇ?" → "ㅇㅇ" 에서 "ㅇㅇ" 은 녹음했다는 사실의 확인)
        if qi > 0 and _nospace_len(info["question"]) <= _SHORT_ASK_MAX:
            qi -= 1
            info["question"] = f'{lines[qi].text.strip()} → {info["question"]}'

    # Gemini 에 보낼 맥락: 오래된 다른 주제가 섞이지 않게 필요한 줄만
    #  질문이 있으면 그 질문부터 / 나눠 보낸 문장이면 그 조각들 / 그 외 직전 3줄
    if qi is not None:
        picked = lines[qi:]
    elif info["joined"]:
        picked = lines[len(lines) - len(frags):]
    else:
        picked = lines[-3:]
    info["payload"] = [{"who": "writer" if c.same_author else "other", "text": c.text.strip()}
                       for c in picked]
    return info


def _context_may_matter(message, ctx):
    """1차 필터가 건너뛰려던 짧은 메시지라도 맥락상 답·조각이면 Gemini 로 확인 (웃음만은 제외)"""
    if not (ctx["joined"] or ctx["question"]):
        return False
    return not _LAUGH_ONLY_RE.match(message.strip())


# 띄어쓰기를 지운 문장 끝의 확실한 추측·질문 표현 / 사실 보고 표현
_GUESS_END_RE = re.compile(
    r"(아닐까|일까|려나|지않을까|거아니야|거아냐|건가|인가|을까요|을까)[?？!~.ㅋㅎㄷ]*$")
_TRAIL_RE = re.compile(r"[?？!~.ㅋㅎㄷㅠ]+$")
_QUESTION_WORD_RE = re.compile(r"(누구|뭐|무엇|언제|어디|어떻게|몇)")
_WHY_RE = re.compile(r"왜")
_OPEN_QUESTION_RE = re.compile(r"(누가|누구|뭐|무엇|무슨|언제|어디|어떻게|몇|왜)")


def _is_guess_end(t):
    """띄어쓰기 없앤 문장이 추측으로 끝나는지. '할까/올까/될까'처럼 받침 ㄹ + 까 도 포함 ('니까'는 제외)"""
    if _GUESS_END_RE.search(t):
        return True
    core = _TRAIL_RE.sub("", t)
    if len(core) >= 2 and core[-1] == "까":
        prev = core[-2]
        if "가" <= prev <= "힣" and (ord(prev) - 0xAC00) % 28 == 8:   # 받침 ㄹ
            return True
    return False
_FACT_END_RE = re.compile(
    r"(했어|었어|았어|였어|더라|더라고|했대|했음|었음|았음|였음|함|됨|임|했다|었다|았다|였다|라니|구나|네|이래|래"
    r"|나봐|가봐|나봄|가봄|모양이야|모양임|모양이네|던것같아|던거같아|던것같음|던거같음)"
    r"[!~.ㅋㅎㄷㅠ]*$")
# 감상(reaction)으로 답했어도 문장 안에 사실을 품은 표현 → 줄거리 사실로 봄
#  "~연다니 기대됨", "~발표하는 거 개웃김", "~바뀐 거 보니까", "~있었던 모양이야", "~하는 장면 소름"
_EMBEDDED_FACT_RE = re.compile(
    r"(다니|라니|거\s?보니|걸\s?보니|모양이|[는은던]\s?거\s?(개|진짜|너무|완전|은근|넘|좀)|[는은던]\s?장면"
    r"|[었았였했]다는|다는\s?(사실|거|것))")
# 앞으로의 일을 예상하는 말 → 사실로 바꾸지 않음 ("다음 화에서 ~터질 것 같은 느낌이야")
_FUTURE_RE = re.compile(r"(느낌|다음\s?(화|회|에피)|나중에|앞으로)")


def _future_like(t):
    """'할 것 같아', '터질 것 같아' 처럼 받침 ㄹ + 것/거 같 = 예상. '했던 것 같아'(지난 일)는 아님"""
    for m in re.finditer(r"(것|거)같", t):
        i = m.start()
        if i > 0 and "가" <= t[i - 1] <= "힣" and (ord(t[i - 1]) - 0xAC00) % 28 == 8:
            return True
    return False
_PRESUPPOSE_RE = re.compile(r"(왜|알았|봤어\?|봤냐|봤음\?)")


def _single_sentence(text):
    parts = [x for x in _SENTENCE_SPLIT_RE.split(text.strip()) if x and len(x.strip()) >= 4]
    return len(parts) <= 1


def _stance_override(stance, message):
    """
    한 문장짜리 메시지의 끝맺음이 확실할 때만 Gemini 의 stance 를 바로잡음 (띄어쓰기에 따라 흔들리던 문제)
      fact  → guess : "~아닐까", "~거 아니야?" 로 끝나고 전제(왜/알았어/봤어?)가 없음
      guess/question → fact : 물음표·추측 표현 없이 "~했어", "~임" 처럼 사실로 끝남
    반환: (stance, 바꿨으면 원래 stance 아니면 None)
    """
    if stance == "reaction" and _EMBEDDED_FACT_RE.search(message) and not _FUTURE_RE.search(message):
        return "fact", stance          # 감상 속 사실은 여러 문장이어도 적용 (가리는 쪽이라 안전)
    if not _single_sentence(message):
        return stance, None
    t = re.sub(r"\s+", "", message)
    if stance == "fact" and _is_guess_end(t) and not _PRESUPPOSE_RE.search(t):
        return "guess", stance
    if (stance in ("guess", "question") and "?" not in t and "？" not in t
            and not _is_guess_end(t) and not _QUESTION_WORD_RE.search(t)
            and not _FUTURE_RE.search(message) and not _future_like(t) and _FACT_END_RE.search(t)):
        return "fact", stance
    return stance, None


# ---------------------------------------------------------------- 장문 검색
# 문장 경계: 문장부호·줄바꿈 / 접속 표현 / "N화" 앞
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?。…~])\s+|\n+"
    r"|\s+(?:그리고|근데|그런데|그래서|그러다가|그러고\s?나서|그 다음에|그 뒤에|그리곤|and then|then)\s+"
    r"|(?=\b\d+\s?화)|(?=\bep(?:isode)?\s?\d+)",
    re.IGNORECASE,
)


def _split_long_message(text):
    """긴 메시지를 사건 검색용 구간으로 나눔. 짧으면 [] (나누지 않음)"""
    t = text.strip()
    if len(t) < LONG_MESSAGE_CHARS:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(t) if p and p.strip()]
    parts = [p for p in parts if len(p) >= 6]
    if len(parts) <= 1:
        # 문장 경계가 없으면 단어 6개 창을 3개씩 밀며 나눔
        words = t.split()
        if len(words) < 7:
            return []
        parts = [" ".join(words[i:i + 6]) for i in range(0, len(words) - 3, 3)]
    return parts[:8]
