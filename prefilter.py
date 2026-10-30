"""
SPOILOCK 1차 필터 (스포 검증 전에 '검사할 필요가 없는 채팅'을 걸러냄)

판단 순서
  1) 규칙 통과(rule_pass)      : 메시지 '전체'가 웃음·감탄·인사·진도 잡담일 때만 → SKIP (한국어·영어)
  2) 강제 검사(must_check)     : 스포 신호어/지시어/작품 고유명사가 있으면 → 무조건 Gemini (한국어·영어)
  3) 일상 대화 통과(daily_pass) : 짧은 메시지가 밥·날씨·일정 같은 생활 주제일 때 → SKIP
                                 (2단계 뒤에 있으므로 신호어·고유명사가 섞이면 통과 못 함)
  4) 유사도 통과(similarity)   : 등록된 사건과의 최고 유사도가 임계값 미만 → SKIP
  5) 나머지                    : Gemini 검사

원칙: '통과'는 허용 목록(allowlist)으로만 한다. "스포 단어가 없으면 통과"는 금지.
"""
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEYWORDS_FILE = ROOT / "config" / "prefilter_keywords.txt"   # 작품 고유명사(인물·장소·물건) 한 줄에 하나

# 일상 대화 통과 단계 on/off (문제가 생기면 .env 에 PREFILTER_DAILY=off)
DAILY_ENABLED = os.getenv("PREFILTER_DAILY", "on").strip().lower() != "off"
# 일상 대화로 통과시킬 최대 길이(공백 제외 글자 수). 길수록 드라마 이야기가 섞일 위험이 커짐
DAILY_MAX_CHARS = int(os.getenv("PREFILTER_DAILY_MAX_CHARS", "25"))

# ---------------------------------------------------------------- 1) 규칙 통과
_REACTION_CHARS = set("ㅋㅎㅠㅜㄷㅇㄱㅊㅁㅂ!?.~^;:,-_ ")
_TAIL = r"[\sㅋㅎㅠㅜ!?.~^;]*"          # 뒤에 붙는 웃음/문장부호 허용

_WHOLE_MESSAGE_PATTERNS = [
    # 인사
    r"(안녕|안녕하세요|하이|ㅎㅇ|반가워요?|반갑습니다|다들 안녕|좋은 (아침|저녁|밤)|굿밤|잘자요?|수고하세요|들어왔어요|왔어요|ㅂㅇ|바이|잘가요?)",
    # 단독 감탄/감상 (대상·사건이 없는 것만)
    r"(와|우와|헐|대박|오+|미쳤다|미쳤네|레전드|소름|꿀잼|노잼|존잼|재밌다|재밌네|재밌어|재미있다|웃기다|웃겨|슬프다|무섭다|긴장된다|떨린다|기대된다|기대돼|최고|최고다|명작|띵작)",
    # 시청 진도·일정 잡담 (내용 언급 없음)
    r"(몇\s?화까지 (봤어|보셨어요|봄)\??|다음 화 언제( 나와| 해)?\??|다음 주까지 어떻게 기다려|같이 볼 사람|지금 보는 중|이제 보러 감|정주행 (중|시작))",
    # 배우·연출·음악 칭찬 (인물 행동/사건 언급 없음)
    r"((배우\s?)?(배우|연기|연출|ost|OST|음악|영상미|cg|CG|오프닝|엔딩곡)\s?(진짜|너무|완전)?\s?(좋다|좋네|잘한다|잘하네|미쳤다|최고))",
]
_WHOLE_RE = [re.compile(rf"^{p}{_TAIL}$") for p in _WHOLE_MESSAGE_PATTERNS]

# 영어: 메시지 '전체'가 인사·감탄·맞장구일 때만 (대소문자 무시)
_EN_TAIL = r"[\s!?.~^;:,)(]*"
_EN_WHOLE_PATTERNS = [
    # 인사
    r"(hi+|hello+|hey+|hiya|yo|sup|what'?s up|good (morning|afternoon|evening|night)|gn|bye+|bye bye|see (you|ya)( all| guys)?|hi (all|guys|everyone)|hello (all|guys|everyone))",
    # 감탄·웃음·맞장구
    r"(lol+|lmao+|rofl|ha(ha)+|he(he)+|omg+|wow+|whoa+|nice+|cool|awesome|amazing|ok(ay)?|k|yes+|yep|yeah|no+|nope|thx|thanks|thank you|ty|gg|same|true|agreed|this is (so )?(good|fun|great))",
    # 진도·일정 잡담
    r"(which episode are you on|what episode are you on|when is the next episode|can'?t wait( for the next episode)?|binge watching|watching now)",
]
_EN_WHOLE_RE = [re.compile(rf"^{p}{_EN_TAIL}$", re.IGNORECASE) for p in _EN_WHOLE_PATTERNS]


def rule_pass(text: str) -> str | None:
    """통과면 사유 문자열, 아니면 None"""
    t = text.strip()
    if not t:
        return "empty"
    if all(ch in _REACTION_CHARS or unicodedata.category(ch) in ("So", "Sk", "Cf")
           for ch in t):
        return "reaction_only"
    for rx in _WHOLE_RE:
        if rx.match(t):
            return "general_pattern"
    for rx in _EN_WHOLE_RE:
        if rx.match(t):
            return "general_pattern_en"
    return None


# ---------------------------------------------------------------- 2) 강제 검사
_SPOILER_CUES = [
    # 생사·정체·관계·결말 (작품과 무관하게 스포에 자주 쓰이는 표현)
    r"죽", r"살아", r"살았", r"사망", r"범인", r"정체", r"배신", r"결말", r"엔딩", r"마지막(에|화)",
    r"반전", r"사실은", r"알고\s?보니", r"진짜는", r"형제", r"자매", r"부모", r"아들", r"딸",
    r"아빠", r"엄마", r"돌아오", r"돌아왔", r"사라지", r"사라졌", r"실종", r"밝혀", r"드러나",
    r"결국", r"끝내", r"나중에", r"\d+\s?화(에서|에|부터)",
    # 앞 대화를 받는 지시어 ("맞아 걔임"): 문맥 없이는 판단 불가 → 검사로 보냄
    r"^(맞아|ㅇㅇ|응|어|그니까)\b", r"걔", r"그\s?(사람|놈|여자|남자|애)", r"그거", r"쟤",
]
_CUE_RE = re.compile("|".join(_SPOILER_CUES))

# 영어 스포 신호어 (단어 단위, 대소문자 무시)
_EN_SPOILER_CUES = [
    r"die[sd]?", r"dying", r"dead", r"death", r"kill(s|ed|er|ing)?", r"murder(ed|er)?",
    r"alive", r"surviv(e|es|ed|or)", r"(the )?ending", r"ends? up", r"in the end", r"finally",
    r"twist", r"turns? out", r"turned out", r"reveal(s|ed)?", r"secret(ly)?", r"betray(s|ed|al)?",
    r"traitor", r"spoiler", r"real identity", r"actually (was|is)", r"(his|her|their) (father|mother|dad|mom|brother|sister|son|daughter)",
    r"father", r"mother", r"brother", r"sister", r"son", r"daughter",
    r"come back", r"came back", r"disappear(s|ed)?", r"missing",
    r"ep(isode)?\s?\d+", r"season finale", r"last episode",
    r"that (guy|girl|man|woman|kid)", r"he'?s the", r"she'?s the",
]
_EN_CUE_RE = re.compile(r"\b(" + "|".join(_EN_SPOILER_CUES) + r")\b", re.IGNORECASE)


def _load_keywords() -> list[str]:
    if not KEYWORDS_FILE.is_file():
        return []
    words = []
    for line in KEYWORDS_FILE.read_text(encoding="utf-8-sig").splitlines():
        w = line.split("#", 1)[0].strip()
        if w:
            words.append(w)
    return words


_KEYWORDS = _load_keywords()


def reload_keywords():
    global _KEYWORDS
    _KEYWORDS = _load_keywords()
    return len(_KEYWORDS)


def must_check(text: str) -> str | None:
    """반드시 Gemini 검사가 필요하면 사유 문자열, 아니면 None"""
    m = _CUE_RE.search(text)
    if m:
        return f"cue:{m.group(0)}"
    m = _EN_CUE_RE.search(text)
    if m:
        return f"cue_en:{m.group(0)}"
    lowered = text.lower()
    for w in _KEYWORDS:
        if w.lower() in lowered:
            return f"keyword:{w}"
    return None


# ---------------------------------------------------------------- 3) 일상 대화 통과
# 드라마와 무관한 생활 주제. must_check 를 통과한(신호어·고유명사 없는) 짧은 메시지에만 적용
_DAILY_TOPICS = [
    # 식사·음식
    r"아침밥", r"점심", r"저녁", r"야식", r"밥\s?(먹|뭐|은|을|시간)", r"뭐\s?먹", r"먹을까", r"먹었", r"배고", r"배불",
    r"치킨", r"피자", r"떡볶이", r"라면", r"커피", r"맥주", r"배달",
    # 날씨·계절
    r"날씨", r"비\s?(와|온다|오네|올|옴)", r"눈\s?(와|온다|오네|올|옴)", r"덥", r"더워", r"추워", r"춥", r"미세먼지",
    # 일상·일정
    r"출근", r"퇴근", r"회사", r"학교", r"수업", r"과제", r"시험", r"알바", r"주말", r"내일", r"오늘\s?(뭐|머)",
    r"월요일", r"화요일", r"수요일", r"목요일", r"금요일", r"토요일", r"일요일",
    r"졸려", r"졸리", r"피곤", r"자러", r"잘게", r"씻고",
]
_DAILY_RE = re.compile("|".join(_DAILY_TOPICS))

_EN_DAILY_TOPICS = [
    r"lunch", r"dinner", r"breakfast", r"hungry", r"eat(ing)?", r"food", r"coffee", r"pizza",
    r"weather", r"rain(ing|y)?", r"snow(ing|y)?", r"hot", r"cold",
    r"(at|to|from|after) work", r"work(ing)? (today|late)", r"school", r"class", r"homework", r"exam", r"weekend", r"tomorrow", r"tonight",
    r"monday", r"tuesday", r"wednesday", r"thursday", r"friday", r"saturday", r"sunday",
    r"tired", r"sleepy", r"going to bed",
]
_EN_DAILY_RE = re.compile(r"\b(" + "|".join(_EN_DAILY_TOPICS) + r")\b", re.IGNORECASE)

# 인물을 가리키는 대명사가 있으면 생활 잡담이라도 통과시키지 않음 ("they eat people" 같은 경우)
_DAILY_BLOCK_RE = re.compile(
    r"\b(he|she|they|him|her|them|his|their|everyone died)\b|그(가|는|를|녀|들)", re.IGNORECASE)


def daily_pass(text: str) -> str | None:
    """짧은 생활 잡담이면 사유 문자열. must_check 가 None 일 때만 호출해야 함"""
    if not DAILY_ENABLED:
        return None
    if len(re.sub(r"\s", "", text)) > DAILY_MAX_CHARS:
        return None
    if _DAILY_BLOCK_RE.search(text):
        return None
    m = _DAILY_RE.search(text) or _EN_DAILY_RE.search(text)
    return f"daily:{m.group(0)}" if m else None


# ---------------------------------------------------------------- 종합 판단
def decide(text: str, best_sim: float | None, sim_threshold: float | None) -> dict:
    """
    반환: {"decision": "SKIP" | "CHECK", "stage": ..., "reason": ...}
    stage: rule | must_check | daily | similarity | default
    best_sim: 등록된 사건과의 최고 코사인 유사도 (사건이 없으면 None)
    sim_threshold: None이면 유사도 통과를 쓰지 않음 (보정 전 기본값)
    """
    r = rule_pass(text)
    if r:
        return {"decision": "SKIP", "stage": "rule", "reason": r}
    c = must_check(text)
    if c:
        return {"decision": "CHECK", "stage": "must_check", "reason": c}
    d = daily_pass(text)
    if d:
        return {"decision": "SKIP", "stage": "daily", "reason": d}
    if sim_threshold is not None and best_sim is not None and best_sim < sim_threshold:
        return {"decision": "SKIP", "stage": "similarity",
                "reason": f"best_sim {best_sim:.4f} < {sim_threshold:.4f}"}
    return {"decision": "CHECK", "stage": "similarity" if sim_threshold is not None else "default",
            "reason": None}