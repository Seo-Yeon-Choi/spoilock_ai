"""
SPOILOCK 채팅 스포일러 판정 테스트

정답을 붙인 채팅 목록(CSV)을 Python /judge 에 한꺼번에 보내고,
Spring 과 같은 정책으로 공개 여부를 정한 뒤 시청 진도별(기본 2·4·6화) 시청자 기준으로 채점한다.

명령 (Python 서버를 켠 상태에서, 이 파일이 있는 폴더에서 실행)
  python chat_test.py export                  사건 목록 → chat_test/events.csv
  python chat_test.py draft                   Gemini 로 테스트 채팅 초안 → chat_test/cases_draft.csv
  python chat_test.py run                     chat_test/cases.csv 판정·채점 → chat_test/results/<시각>/
  python chat_test.py run --resume <폴더>      멈춘 곳부터 이어서 (Gemini 한도·중단)
  python chat_test.py rescore --resume <폴더> --publish-no-match
                                              Gemini 호출 없이 다른 정책으로 다시 채점

공통 옵션
  --drama 1  --kb local-csy  --episodes 1-6  --server http://127.0.0.1:8000

run 옵션
  --cases chat_test/cases.csv   --viewers 2,4,6   --prefilter on|off|shadow
  --repeat 2                    같은 문장을 여러 번 판정해 결과가 흔들리는지 확인
  --publish-no-match            배포 서버처럼 NO_MATCH 를 공개 (기본: 보류)
  --no-unverified               미검수 사건 판정을 보류 (기본: 데모 허용)
  --limit 10                    앞에서 N개만 (빠른 확인용)
  --sample 200                  유형별로 고르게 200개만 (사람이 쓴 문장은 항상 포함, --seed 로 고정)
  --top-k 40                    Gemini 에 보여줄 후보 사건 수 (기본 20 = Spring 설정)
  --tag prompt2                 결과 폴더 이름 메모
  --rpm 14                      Gemini 분당 호출 제한 (기본 14, 0 = 제한 없음)
  --use-cache                   이전에 같은 조건으로 판정한 결과 재사용 (Gemini 호출 없음)

cases.csv 열
  id, category, message, expected, episode, event_id, source, note
  expected: SPOILER      등록된 사건을 드러냄 (episode = 공개 회차)
            SAFE         누구에게나 보여도 됨
            UNREGISTERED 실제 줄거리지만 사건 목록에 없음 (episode = 실제 공개 회차)
            GUESS        사건과 맞는 추측·질문 (episode = 사건 공개 회차)
                         작성자가 그 회차를 봤으면(author_episode ≥ episode) SPOILER, 아니면 SAFE 로 채점
  author_episode: 작성자의 시청 회차 (비우면 --author-episode 값, 기본 없음 = 진도 모름)
"""
import argparse
import csv
import html
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "chat_test"
CASE_FIELDS = ["id", "category", "message", "expected", "episode", "event_id", "source", "note", "author_episode"]
EXPECTED = ("SPOILER", "SAFE", "UNREGISTERED", "GUESS")

CATEGORY_NAMES = {
    "A": "사건 직접 언급",
    "B": "바꿔 말하기·별명",
    "C": "감정에 섞인 사실",
    "D": "부분 언급·암시",
    "E1": "추측(틀림)",
    "E2": "추측(맞음)",
    "F": "내용 없는 반응",
    "G": "일상·무관",
    "H": "함정(같은 단어)",
    "I": "등록 안 된 사실",
    "J": "장문·오타·혼합",
}


# ------------------------------------------------------------------ 공통
def load_env():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", encoding="utf-8-sig", override=False)
    except ImportError:
        pass


def parse_episodes(text):
    result = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.extend(range(int(a), int(b) + 1))
        elif part:
            result.append(int(part))
    return sorted(set(result))


def http_json(method, url, body=None, token=None, timeout=120):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("X-Admin-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"{method} {url} → HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"{url} 에 연결할 수 없습니다. Python 서버가 켜져 있나요? ({e.reason})") from None


def read_csv(path):
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp949"):        # 엑셀에서 저장하면 cp949 일 수 있음
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise SystemExit(f"{path}: 인코딩을 알 수 없습니다 (UTF-8 로 저장해 주세요)")
    return list(csv.DictReader(text.splitlines()))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:     # 엑셀에서 한글이 깨지지 않게
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def effective_episode(event):
    if event.get("verification_status") == "VERIFIED" and event.get("first_reveal_episode"):
        return int(event["first_reveal_episode"])
    return int(event.get("source_episode") or 0)


# ------------------------------------------------------------------ export
def fetch_events(args):
    token = os.getenv("ADMIN_TOKEN", "").strip()
    events = []
    for ep in args.episodes:
        url = f"{args.server}/admin/events?drama_id={args.drama}&episode={ep}&kb_version={args.kb}"
        events.extend(http_json("GET", url, token=token))
    return events


def cmd_export(args):
    events = fetch_events(args)
    rows = [{
        "event_id": e["event_id"],
        "episode": effective_episode(e),
        "source_episode": e.get("source_episode"),
        "verification_status": e.get("verification_status"),
        "event_type": e.get("event_type"),
        "description": e.get("description"),
    } for e in events]
    rows.sort(key=lambda r: (r["episode"], r["event_id"]))
    path = OUT_DIR / "events.csv"
    write_csv(path, rows, list(rows[0].keys()) if rows else ["event_id"])
    by_ep = Counter(r["episode"] for r in rows if r["verification_status"] != "REJECTED")
    print(f"사건 {len(rows)}건 → {path}")
    for ep in sorted(by_ep):
        print(f"  {ep}화: {by_ep[ep]}건")
    rejected = sum(1 for r in rows if r["verification_status"] == "REJECTED")
    if rejected:
        print(f"  (반려 {rejected}건은 판정에 쓰이지 않음)")


# ------------------------------------------------------------------ draft
DRAFT_SYSTEM = """You write realistic Korean group-chat messages for testing a TV-drama spoiler filter.
The drama is "{title}". Viewers chat in casual Korean (반말, 줄임말, ㅋㅋ, 감탄사), like a fan chat room.
Never copy the event sentence. Change vocabulary and sentence structure like real people do.
Return JSON only."""

EVENT_PROMPT = """Below are plot events revealed in episode {episode}.
For EACH event write these chat messages (Korean, 1 sentence each, natural chat tone):
- "B": 2 paraphrases that clearly reveal the same fact with different words or nicknames
- "C": 1 message that reveals the fact wrapped in an emotion/reaction (e.g. "~해서 소름", "~하는 장면 슬펐어")
- "D": 1 message that only partially or indirectly reveals the fact but a reader could still infer it
- "E2": 1 guess/question that happens to be correct ("~일 것 같아", "~아닐까?")

Events:
{events}"""

SAFE_PROMPT = """Write {n} chat messages for the chat room of "{title}" that reveal NO plot facts.
Mix these kinds (use the category codes):
- "F": pure reactions with no story content ("이번 화 미쳤다", "엔딩 소름", "연기 좋더라")
- "G": daily talk or questions unrelated to the plot ("몇 시에 올라와?", "점심 뭐 먹지", "자막 어디서 봐?")
- "E1": wrong guesses or predictions about the story ("~일 것 같아", "~하지 않을까")
- "H": traps that reuse words from the events below but say nothing about what happens
  (e.g. praising an actor, the camera work, or the set where an event happens)
Event list for context (do NOT reveal these):
{events}"""

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "category": {"type": "string", "enum": ["B", "C", "D", "E1", "E2", "F", "G", "H"]},
                    "message": {"type": "string"},
                },
                "required": ["category", "message"],
            },
        }
    },
    "required": ["items"],
}


def _gemini():
    try:
        import gemini_client
        return gemini_client.get_client()
    except ImportError:
        from google import genai
        return genai.Client(), os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _ask(client, model, system, prompt):
    from google.genai import types
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=DRAFT_SCHEMA,
            temperature=0.9,
        ),
    )
    return json.loads(resp.text).get("items", [])


def cmd_draft(args):
    path = OUT_DIR / "events.csv"
    if not path.is_file():
        raise SystemExit("먼저 python chat_test.py export 를 실행하세요.")
    events = [e for e in read_csv(path) if e.get("verification_status") != "REJECTED"]
    if not events:
        raise SystemExit("사건이 없습니다.")
    by_id = {e["event_id"]: e for e in events}
    client, model = _gemini()
    system = DRAFT_SYSTEM.format(title=args.title)
    rows = []

    by_ep = defaultdict(list)
    for e in events:
        by_ep[int(e["episode"])].append(e)
    for ep in sorted(by_ep):
        listing = "\n".join(f'- [{e["event_id"]}] {e["description"]}' for e in by_ep[ep])
        print(f"{ep}화 사건 {len(by_ep[ep])}건 초안 생성 중...", flush=True)
        for item in _ask(client, model, system, EVENT_PROMPT.format(episode=ep, events=listing)):
            event = by_id.get(item.get("event_id", ""))
            if event is None or item.get("category") not in ("B", "C", "D", "E2"):
                continue
            # 맞는 추측(E2)은 현재 정책상 공개가 정답
            expected = "SAFE" if item["category"] == "E2" else "SPOILER"
            rows.append({"category": item["category"], "message": item["message"].strip(),
                         "expected": expected, "episode": event["episode"],
                         "event_id": event["event_id"], "source": "AI"})
        time.sleep(1)

    listing = "\n".join(f'- {e["description"]}' for e in events)
    print(f"스포 아닌 채팅 {args.safe}개 생성 중...", flush=True)
    for item in _ask(client, model, system, SAFE_PROMPT.format(n=args.safe, title=args.title, events=listing)):
        if item.get("category") in ("E1", "F", "G", "H"):
            rows.append({"category": item["category"], "message": item["message"].strip(),
                         "expected": "SAFE", "episode": "", "event_id": "", "source": "AI"})

    for i, r in enumerate(rows, 1):
        r["id"] = f"AI{i:03d}"
    out = OUT_DIR / "cases_draft.csv"
    write_csv(out, rows, CASE_FIELDS)
    print(f"\n초안 {len(rows)}개 → {out}")
    print(Counter(r["category"] for r in rows))
    print("검토 후 팀원이 쓴 문장과 합쳐 chat_test/cases.csv 로 저장하세요.")


# ------------------------------------------------------------------ run
class Policy:
    """Spring SpoilerVerdictPolicy 와 같은 규칙"""

    def __init__(self, args):
        self.allow_unverified = not args.no_unverified
        self.publish_no_match = args.publish_no_match
        self.publish_off_topic = True
        self.publish_skipped_general = True
        self.publish_skipped_unrelated = False
        self.author_guess = not getattr(args, "no_author_guess", False)
        # Spring app.spoiler.hold-as-finale=true: 보류 → 드라마 마지막 회차(색인 최대 회차)부터 공개
        self.hold_as_finale = not getattr(args, "no_finale", False)

    def decide(self, r, indexed, author=None):
        """반환: ("ALL", None) 모두 공개 / ("FROM", n) n화 이상 공개 / ("HOLD", 이유) / ("FAIL", 이유)
        author: 작성자 시청 회차. Spring 과 같이 '작성자가 이미 본 회차의 추측'이면 그 회차부터 공개"""
        decision = self._decide_verdict(r, indexed)
        if self.author_guess:
            decision = self._apply_author_guess(decision, r, indexed, author)
        if self.hold_as_finale and decision[0] == "HOLD" and indexed:
            return "FROM", max(indexed)
        return decision

    def _apply_author_guess(self, decision, r, indexed, author):
        eps = [e for e in (r.get("guess_episodes") or []) if e and e > 0 and e in indexed]
        if decision[0] == "FAIL" or author is None or not eps:
            return decision
        author = max(0, author)
        known = max((e for e in eps if e <= author), default=None)
        if known is None:
            # 안 본 회차 추측 = 모르고 하는 추측 → 전체 공개 (MATCH 는 그대로)
            return decision if r.get("analysis_verdict") == "MATCH" else ("ALL", None)
        unverified = not r.get("guess_basis") or str(r["guess_basis"]).startswith("UNVERIFIED")
        if unverified and not self.allow_unverified:
            return ("HOLD", "작성자가 본 회차의 추측(미검증 사건)") if decision[0] in ("ALL", "FROM") else decision
        if decision[0] == "HOLD":
            return ("FROM", known) if r.get("analysis_verdict") == "NO_MATCH" else decision
        if decision[0] == "FROM" and decision[1] >= known:
            return decision
        return "FROM", known

    def _decide_verdict(self, r, indexed):
        v = r.get("analysis_verdict")
        if v == "MATCH":
            ep = r.get("required_episode_number")
            if not r.get("matched_event_id"):
                return "HOLD", "MATCH 인데 사건 ID 없음"
            if not ep or ep <= 0:
                return "HOLD", "MATCH 인데 회차 없음"
            if ep not in indexed:
                return "HOLD", f"회차 범위 밖 {ep}"
            if r.get("needs_review"):
                return "HOLD", "검토 필요"
            unverified = not r.get("episode_basis") or str(r["episode_basis"]).startswith("UNVERIFIED")
            if unverified and not self.allow_unverified:
                return "HOLD", "미검증 사건(데모 허용 꺼짐)"
            return "FROM", ep
        if v == "SKIPPED_GENERAL":
            return ("ALL", None) if self.publish_skipped_general else ("HOLD", "1차 필터: 일반 대화")
        if v == "SKIPPED_UNRELATED":
            return ("ALL", None) if self.publish_skipped_unrelated else ("HOLD", "1차 필터: 사건과 무관")
        if v == "NO_MATCH":
            if self.publish_no_match:
                return "ALL", None
            if self.publish_off_topic and r.get("about_story") is False:
                return "ALL", None
            return "HOLD", "NO_MATCH 검토 대기"
        if v == "UNCERTAIN":
            return "HOLD", "UNCERTAIN"
        return "FAIL", f"{v}: {r.get('error') or ''}"

    def describe(self):
        return (f"미검수 사건 판정 {'허용' if self.allow_unverified else '보류'}, "
                f"NO_MATCH {'공개' if self.publish_no_match else '보류(드라마 무관 대화만 공개)'}, "
                f"1차 필터 일반 대화 공개 / 사건 무관 보류, "
                f"작성자 진도 추측 판단 {'켬' if self.author_guess else '끔'}, "
                f"보류 → {'마지막 회차부터 공개' if self.hold_as_finale else '영구 보류'}")


def visible(decision, viewer):
    kind, value = decision
    if kind == "ALL":
        return True
    if kind == "FROM":
        return viewer >= value
    return False          # HOLD / FAIL → 아무에게도 안 보임 (작성자 본인 제외)


def author_of(case):
    a = str(case.get("author_episode") or "").strip()
    return int(a) if a.isdigit() else None


def effective_expected(case):
    """GUESS 는 작성자 진도에 따라 SPOILER 또는 SAFE"""
    exp = case["expected"]
    ep = int(case["episode"]) if str(case.get("episode") or "").strip() else 0
    if exp == "GUESS":
        author = author_of(case)
        return ("SPOILER", ep) if author is not None and author >= ep else ("SAFE", 0)
    return exp, ep


def grade(case, decision, viewer):
    """OK / LEAK(보이면 안 되는데 보임) / OVERBLOCK(보여도 되는데 가려짐) / HOLD_OK"""
    shown = visible(decision, viewer)
    exp, ep = effective_expected(case)
    if exp == "SAFE":
        return "OK" if shown else "OVERBLOCK"
    should_hide = viewer < ep
    if should_hide:
        return "LEAK" if shown else "OK"
    if exp == "UNREGISTERED" and not shown:
        return "HOLD_OK"          # 사건 목록에 없으니 보류는 최선의 결과
    return "OK" if shown else "OVERBLOCK"


def validate_cases(cases):
    problems = []
    for c in cases:
        c["expected"] = (c.get("expected") or "").strip().upper()
        c["category"] = (c.get("category") or "").strip().upper()
        c["message"] = (c.get("message") or "").strip()
        if not c["message"]:
            problems.append(f"{c.get('id')}: 메시지가 비어 있음")
        if c["expected"] not in EXPECTED:
            problems.append(f"{c.get('id')}: expected 는 {'/'.join(EXPECTED)} 중 하나")
        if c["expected"] in ("SPOILER", "UNREGISTERED", "GUESS") and not str(c.get("episode") or "").strip().isdigit():
            problems.append(f"{c.get('id')}: {c['expected']} 는 episode(공개 회차)가 필요")
        a = str(c.get("author_episode") or "").strip()
        if a and not a.isdigit():
            problems.append(f"{c.get('id')}: author_episode 는 숫자")
    if problems:
        raise SystemExit("cases.csv 확인 필요:\n  " + "\n  ".join(problems[:20]))


RESULT_FIELDS = ["run", "verdict", "decision", "decision_kind", "decision_episode", "required_episode",
                 "episode_basis", "needs_review", "matched_event_id", "matched_event_ids", "matched_description",
                 "about_story", "stance", "shared_fact", "guess_episodes", "guess_description", "reason",
                 "prefilter_stage",
                 "prefilter_sim", "match_cache", "ms"]


def _is_quota(r):
    """Gemini 사용 한도 초과 (분당·일일)"""
    text = f'{r.get("error") or ""} {r.get("reason") or ""}'.upper()
    return r.get("analysis_verdict") == "ERROR" and any(
        k in text for k in ("429", "RESOURCE_EXHAUSTED", "QUOTA", "RATE LIMIT", "TOO MANY"))


def _used_gemini(r):
    """이번 판정에서 Gemini 를 실제로 불렀는지 (1차 필터에서 끝났거나 캐시면 안 부름)"""
    if str(r.get("analysis_verdict") or "").startswith("SKIPPED"):
        return False
    return not str(r.get("match_cache") or "").startswith("HIT")


def build_row(c, rep, r, ms, policy, viewers, indexed):
    decision = policy.decide(r, indexed, author_of(c))
    pf = r.get("prefilter") or {}
    row = {
        **{k: c.get(k, "") for k in CASE_FIELDS}, "run": rep,
        "verdict": r.get("analysis_verdict"),
        "decision": (f"{decision[1]}화부터 공개" if decision[0] == "FROM"
                     else "모두 공개" if decision[0] == "ALL"
                     else f"보류: {decision[1]}" if decision[0] == "HOLD" else f"오류: {decision[1]}"),
        "decision_kind": decision[0],
        "decision_episode": decision[1] if decision[0] == "FROM" else "",
        "required_episode": r.get("required_episode_number") or "",
        "episode_basis": r.get("episode_basis") or "",
        "needs_review": r.get("needs_review"),
        "matched_event_id": r.get("matched_event_id") or "",
        "matched_event_ids": ";".join(r.get("matched_event_ids") or []),
        "matched_description": r.get("matched_event_description") or "",
        "about_story": r.get("about_story"),
        # v2: 말하는 방식. 코드가 바로잡았으면 "guess(모델:fact)" 처럼 표시
        "stance": (r.get("stance") or "") + (
            f'(모델:{r.get("stance_model")})' if r.get("stance_model") and r.get("stance_model") != r.get("stance") else ""),
        "shared_fact": r.get("shared_fact") or "",
        "guess_episodes": ";".join(str(e) for e in (r.get("guess_episodes") or [])),
        "guess_description": " / ".join(r.get("guess_event_descriptions") or []),
        "reason": r.get("reason") or r.get("error") or "",
        "prefilter_stage": (f'{pf.get("decision", "")} ({pf.get("stage", "")})' if pf else ""),
        "prefilter_sim": pf.get("best_sim") if pf.get("best_sim") is not None else "",
        "match_cache": r.get("match_cache") or "",
        "ms": ms,
    }
    for v in viewers:
        row[f"v{v}"] = grade(c, decision, v)
    return row


def load_raw(out):
    done = {}
    path = out / "raw.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                done[(item["id"], item["run"])] = item
    return done


def finish(out, rows, viewers, policy, args, health, suffix=""):
    meta = {"server": args.server, "model": health.get("gemini_model"), "drama": args.drama, "kb": args.kb,
            "episodes": f"{args.episodes[0]}-{args.episodes[-1]}", "prefilter": args.prefilter, "top_k": args.top_k,
            "prompt_version": getattr(args, "prompt_version", "") or "server",
            "repeat": args.repeat, "policy": policy.describe(), "cases": str(args.cases),
            "at": out.name}
    fields = CASE_FIELDS + RESULT_FIELDS + [f"v{v}" for v in viewers]
    write_csv(out / f"results{suffix}.csv", rows, fields)
    summary = summarize(rows, viewers)
    (out / f"summary{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"report{suffix}.html").write_text(render_report(rows, summary, viewers, meta), encoding="utf-8")
    print_summary(summary, viewers)
    print(f"\n결과: {out / f'report{suffix}.html'}")


def sample_cases(cases, n, seed):
    """유형(category)별로 고르게 n개 뽑기. 사람이 쓴 문장(source=HUMAN)은 항상 포함"""
    import random
    rng = random.Random(seed)
    human = [c for c in cases if (c.get("source") or "").upper() == "HUMAN"]
    rest = [c for c in cases if c not in human]
    groups = defaultdict(list)
    for c in rest:
        groups[c["category"]].append(c)
    for g in groups.values():
        rng.shuffle(g)
    picked, need = [], max(0, n - len(human))
    while need > 0 and any(groups.values()):
        for cat in sorted(groups):
            if groups[cat] and need > 0:
                picked.append(groups[cat].pop())
                need -= 1
    chosen = {id(c) for c in human + picked}
    result = [c for c in cases if id(c) in chosen]          # 원래 순서 유지
    print(f"표본 {len(result)}개 (사람 작성 {len(human)}개 포함): "
          + ", ".join(f"{k} {v}" for k, v in sorted(Counter(c['category'] for c in result).items())))
    return result


def cmd_run(args):
    if args.resume:
        out = Path(args.resume)
        if not (out / "cases.csv").is_file():
            raise SystemExit(f"{out} 에 이전 실행 기록이 없습니다.")
        saved = json.loads((out / "run.json").read_text(encoding="utf-8"))
        for k in ("drama", "kb", "prefilter", "repeat", "cases", "use_cache"):
            setattr(args, k, saved[k])
        args.top_k = saved.get("top_k", 20)
        args.episodes = saved["episodes"]
        cases = read_csv(out / "cases.csv")
    else:
        cases = read_csv(args.cases)
        for i, c in enumerate(cases, 1):
            c["id"] = (c.get("id") or "").strip() or f"C{i:03d}"
        validate_cases(cases)
        if args.author_episode is not None:
            for c in cases:
                if not str(c.get("author_episode") or "").strip():
                    c["author_episode"] = str(args.author_episode)
        if args.sample:
            cases = sample_cases(cases, args.sample, args.seed)
        if args.limit:
            cases = cases[: args.limit]
        out = OUT_DIR / "results" / (f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-pf_{args.prefilter}"
                                     f"-k{args.top_k}" + (f"-{args.tag}" if args.tag else ""))
        write_csv(out / "cases.csv", cases, CASE_FIELDS)
        (out / "run.json").write_text(json.dumps({
            "drama": args.drama, "kb": args.kb, "prefilter": args.prefilter, "repeat": args.repeat,
            "cases": str(args.cases), "episodes": args.episodes, "use_cache": args.use_cache, "top_k": args.top_k,
        }, ensure_ascii=False), encoding="utf-8")
    validate_cases(cases)
    policy = Policy(args)
    viewers = [int(v) for v in args.viewers.split(",")]
    indexed = args.episodes
    done = load_raw(out)
    interval = 60.0 / args.rpm if args.rpm > 0 else 0.0

    try:
        health = http_json("GET", f"{args.server}/health", timeout=10)
    except RuntimeError:
        health = {}
    todo = sum(1 for c in cases for rep in range(1, args.repeat + 1) if (c["id"], rep) not in done)
    print(f"서버 {args.server}  model={health.get('gemini_model')}  "
          f"drama={args.drama}  kb={args.kb}  episodes={indexed[0]}-{indexed[-1]}  prefilter={args.prefilter}")
    print(f"정책: {policy.describe()}")
    print(f"채팅 {len(cases)}개 × {args.repeat}회 · 남은 판정 {todo}개"
          + (f" · 분당 {args.rpm}회 제한 → 최대 약 {round(todo * interval / 60)}분" if interval else "")
          + (" · 캐시 사용" if args.use_cache else ""))
    print(f"기록: {out}\n")

    raw_file = (out / "raw.jsonl").open("a", encoding="utf-8")
    last_call = 0.0
    stopped = False
    try:
        for n, c in enumerate(cases, 1):
            for rep in range(1, args.repeat + 1):
                if (c["id"], rep) in done:
                    continue
                body = {
                    "message": c["message"], "message_id": n, "drama_id": args.drama, "kb_version": args.kb,
                    "episodes": indexed, "retrieval": "vector", "top_k": args.top_k, "include_evidence": True,
                    "reason_mode": "short", "prefilter_mode": args.prefilter,
                    "refresh": not args.use_cache, "debug": True,
                }
                if args.prompt_version:
                    body["prompt_version"] = args.prompt_version
                for attempt in range(1, args.retries + 2):
                    wait = interval - (time.monotonic() - last_call)
                    if wait > 0:
                        time.sleep(wait)
                    t = time.perf_counter()
                    try:
                        r = http_json("POST", f"{args.server}/judge", body)
                    except RuntimeError as e:
                        r = {"analysis_verdict": "ERROR", "error": str(e)}
                    ms = round((time.perf_counter() - t) * 1000)
                    if _used_gemini(r) or _is_quota(r):
                        last_call = time.monotonic()
                    if not _is_quota(r):
                        break
                    if attempt <= args.retries:
                        print(f"   Gemini 한도 초과 → {args.quota_wait}초 대기 후 다시 시도 ({attempt}/{args.retries})",
                              flush=True)
                        time.sleep(args.quota_wait)
                else:
                    stopped = True
                    break
                item = {"id": c["id"], "run": rep, "ms": ms, "response": r}
                raw_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                raw_file.flush()
                done[(c["id"], rep)] = item
                row = build_row(c, rep, r, ms, policy, viewers, indexed)
                marks = " ".join(f"{v}화:{row[f'v{v}']}" for v in viewers)
                print(f"[{n:>3}/{len(cases)}] {c['id']:<6} {c['category']:<3} {c['expected']:<12} "
                      f"→ {row['verdict']:<16} {row['decision']:<22} {marks}", flush=True)
                if args.sleep:
                    time.sleep(args.sleep)
            if stopped:
                break
    except KeyboardInterrupt:
        stopped = True
        print("\n중단했습니다.")
    finally:
        raw_file.close()

    rows = [build_row(c, rep, done[(c["id"], rep)]["response"], done[(c["id"], rep)]["ms"],
                      policy, viewers, indexed)
            for c in cases for rep in range(1, args.repeat + 1) if (c["id"], rep) in done]
    if stopped:
        print(f"\n{len(rows)}개까지 판정하고 멈췄습니다 (Gemini 한도 또는 중단). 나중에 이어서:")
        print(f'  python chat_test.py run --resume "{out}"')
    if rows:
        finish(out, rows, viewers, policy, args, health)


def cmd_rescore(args):
    """Gemini 호출 없이, 저장된 판정 결과를 다른 정책으로 다시 채점"""
    out = Path(args.resume or "")
    if not args.resume or not (out / "raw.jsonl").is_file():
        raise SystemExit("--resume 에 결과 폴더를 지정하세요. 예: --resume chat_test/results/20261030-...")
    saved = json.loads((out / "run.json").read_text(encoding="utf-8"))
    for k in ("drama", "kb", "prefilter", "repeat", "cases"):
        setattr(args, k, saved[k])
    args.episodes = saved["episodes"]
    args.top_k = saved.get("top_k", 20)
    cases = read_csv(out / "cases.csv")
    validate_cases(cases)
    policy = Policy(args)
    viewers = [int(v) for v in args.viewers.split(",")]
    done = load_raw(out)
    rows = [build_row(c, rep, done[(c["id"], rep)]["response"], done[(c["id"], rep)]["ms"],
                      policy, viewers, args.episodes)
            for c in cases for rep in range(1, args.repeat + 1) if (c["id"], rep) in done]
    suffix = "_" + ("nomatch-publish" if args.publish_no_match else "nomatch-hold") \
             + ("_no-unverified" if args.no_unverified else "") \
             + ("_no-author-guess" if args.no_author_guess else "")
    print(f"정책: {policy.describe()}")
    finish(out, rows, viewers, policy, args, {"gemini_model": None}, suffix)


# ------------------------------------------------------------------ 집계
def _rate(a, b):
    return round(100 * a / b, 1) if b else None


def _avg(xs):
    return round(sum(xs) / len(xs)) if xs else None


def _row_used_gemini(r):
    v = str(r.get("verdict") or "")
    return not v.startswith("SKIPPED") and not str(r.get("match_cache") or "").startswith("HIT")


def summarize(rows, viewers):
    s = {"total": len(rows), "viewers": {}, "categories": {}, "judge": {}, "stability": None}
    for v in viewers:
        g = Counter(r[f"v{v}"] for r in rows)
        eff = [effective_expected(r) for r in rows]
        must_hide = sum(1 for exp, ep in eff if exp != "SAFE" and v < ep)
        may_show = sum(1 for exp, ep in eff if exp == "SAFE" or (exp == "SPOILER" and v >= ep))
        s["viewers"][v] = {
            "ok": g["OK"] + g["HOLD_OK"], "leak": g["LEAK"], "overblock": g["OVERBLOCK"],
            "must_hide": must_hide, "may_show": may_show,
            "leak_rate": _rate(g["LEAK"], must_hide), "overblock_rate": _rate(g["OVERBLOCK"], may_show),
            "accuracy": _rate(g["OK"] + g["HOLD_OK"], len(rows)),
        }
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"] or "?"].append(r)
    def group_stats(items):
        cells = [r[f"v{v}"] for r in items for v in viewers]
        return {
            "count": len(items),
            "expected": Counter(r["expected"] for r in items).most_common(1)[0][0],
            "ok_rate": _rate(sum(1 for x in cells if x in ("OK", "HOLD_OK")), len(cells)),
            "leak": sum(1 for x in cells if x == "LEAK"),
            "overblock": sum(1 for x in cells if x == "OVERBLOCK"),
            "avg_ms": round(sum(int(r["ms"]) for r in items) / len(items)),
            "gemini_calls": sum(1 for r in items if _row_used_gemini(r)),
        }

    for cat, items in sorted(by_cat.items()):
        s["categories"][cat] = {"name": CATEGORY_NAMES.get(cat, ""), **group_stats(items)}
    by_note = defaultdict(list)
    for r in rows:
        if (r.get("note") or "").strip():
            by_note[r["note"].strip()].append(r)
    s["groups"] = {k: group_stats(v) for k, v in sorted(by_note.items())} if len(by_note) <= 20 else {}
    # 추측(GUESS) 문장은 작성자 진도에 따라 정답이 바뀌므로 판정 품질 지표에서는 제외
    spoilers = [r for r in rows if r["expected"] == "SPOILER"]
    matched = [r for r in spoilers if r["decision_kind"] == "FROM"]
    hidden = [r for r in spoilers if r["decision_kind"] in ("FROM", "HOLD")]
    exact = [r for r in spoilers if r["decision_kind"] == "FROM"
             and str(r["decision_episode"]) == str(r["episode"])]
    with_id = [r for r in spoilers if r["event_id"]]
    id_hit = [r for r in with_id if r["event_id"] in (r["matched_event_ids"] or "").split(";")]
    safe = [r for r in rows if r["expected"] == "SAFE"]
    s["judge"] = {
        "spoilers": len(spoilers),
        "match_rate": _rate(len(matched), len(spoilers)),
        "hidden_rate": _rate(len(hidden), len(spoilers)),
        "exact_episode_rate": _rate(len(exact), len(spoilers)),
        "event_id_rate": _rate(len(id_hit), len(with_id)),
        "safe": len(safe),
        "safe_published_rate": _rate(sum(1 for r in safe if r["decision_kind"] == "ALL"), len(safe)),
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "avg_ms": round(sum(int(r["ms"]) for r in rows) / len(rows)) if rows else 0,
        "max_ms": max((int(r["ms"]) for r in rows), default=0),
        "gemini_calls": sum(1 for r in rows if _row_used_gemini(r)),
        "skipped_avg_ms": _avg([int(r["ms"]) for r in rows if str(r["verdict"]).startswith("SKIPPED")]),
        "gemini_avg_ms": _avg([int(r["ms"]) for r in rows if _row_used_gemini(r)]),
    }
    runs = defaultdict(set)
    for r in rows:
        runs[r["id"]].add(r["decision"])
    if any(r["run"] > 1 for r in rows):
        unstable = [cid for cid, d in runs.items() if len(d) > 1]
        s["stability"] = {"unstable": len(unstable), "ids": unstable, "rate": _rate(len(unstable), len(runs))}
    return s


def print_summary(s, viewers):
    print("\n===== 요약 =====")
    for v in viewers:
        x = s["viewers"][v]
        print(f"  {v}화 시청자: 정확도 {_pct(x['accuracy'])}  스포 노출 {x['leak']}/{x['must_hide']} "
              f"({_pct(x['leak_rate'])})  과잉 차단 {x['overblock']}/{x['may_show']} ({_pct(x['overblock_rate'])})")
    j = s["judge"]
    print(f"  Gemini 호출 {j['gemini_calls']}회 · 평균 응답 {j['avg_ms']}ms "
          f"(1차 필터에서 끝남 {j['skipped_avg_ms'] or '-'}ms / Gemini {j['gemini_avg_ms'] or '-'}ms)")
    print(f"  스포 문장 사건 매칭 {_pct(j['match_rate'])} (보류 포함 가림 {_pct(j['hidden_rate'])})  "
          f"회차 정확 {_pct(j['exact_episode_rate'])}  사건 일치 {_pct(j['event_id_rate'])}  평균 {j['avg_ms']}ms")
    if s["stability"]:
        print(f"  반복 판정 불일치 {s['stability']['unstable']}건")


# ------------------------------------------------------------------ 리포트
def _pct(v):
    return "-" if v is None else f"{v}%"


def render_report(rows, s, viewers, meta):
    e = html.escape
    def badge(g):
        cls = {"OK": "ok", "HOLD_OK": "ok", "LEAK": "bad", "OVERBLOCK": "warn"}.get(g, "")
        label = {"OK": "정상", "HOLD_OK": "보류(정상)", "LEAK": "스포 노출", "OVERBLOCK": "과잉 차단"}.get(g, g)
        return f'<span class="b {cls}">{label}</span>'

    vcards = "".join(
        f'<div class="card"><div class="k">{v}화 시청자</div><div class="big">{_pct(x["accuracy"])}</div>'
        f'<div class="sub">스포 노출 <b class="{"bad" if x["leak"] else ""}">{x["leak"]}</b> / {x["must_hide"]}'
        f' ({_pct(x["leak_rate"])})<br>과잉 차단 <b>{x["overblock"]}</b> / {x["may_show"]} ({_pct(x["overblock_rate"])})</div></div>'
        for v, x in s["viewers"].items())
    j = s["judge"]
    jcards = (
        f'<div class="card"><div class="k">스포 문장 사건 매칭</div><div class="big">{_pct(j["match_rate"])}</div>'
        f'<div class="sub">스포 {j["spoilers"]}개 중 · 보류 포함 가림 {_pct(j["hidden_rate"])}</div></div>'
        f'<div class="card"><div class="k">공개 회차 정확</div><div class="big">{_pct(j["exact_episode_rate"])}</div>'
        f'<div class="sub">사건 ID 일치 {_pct(j["event_id_rate"])}</div></div>'
        f'<div class="card"><div class="k">정상 채팅 공개</div><div class="big">{_pct(j["safe_published_rate"])}</div>'
        f'<div class="sub">SAFE {j["safe"]}개 중</div></div>'
        f'<div class="card"><div class="k">응답 시간</div><div class="big">{j["avg_ms"]}ms</div>'
        f'<div class="sub">최대 {j["max_ms"]}ms · 1차 필터에서 끝남 {j["skipped_avg_ms"] or "-"}ms · Gemini {j["gemini_avg_ms"] or "-"}ms</div></div>'
        f'<div class="card"><div class="k">Gemini 호출</div><div class="big">{j["gemini_calls"]}</div>'
        f'<div class="sub">전체 {s["total"]}개 중 · 1차 필터로 {s["total"] - j["gemini_calls"]}개 절약</div></div>')
    stab = ""
    if s["stability"]:
        st = s["stability"]
        stab = (f'<p class="note">반복 판정에서 결과가 달라진 문장: <b>{st["unstable"]}</b>개 ({_pct(st["rate"])})'
                + (f' — {e(", ".join(st["ids"][:20]))}' if st["ids"] else "") + "</p>")
    def stat_cells(x):
        return (f'<td>{x["count"]}</td><td>{e(x["expected"])}</td>'
                f'<td class="num">{_pct(x["ok_rate"])}</td><td class="num {"bad" if x["leak"] else ""}">{x["leak"]}</td>'
                f'<td class="num">{x["overblock"]}</td><td class="num">{x["avg_ms"]}ms</td>')
    cat_rows = "".join(f'<tr><td>{e(c)}</td><td>{e(x["name"])}</td>{stat_cells(x)}</tr>'
                       for c, x in s["categories"].items())
    group_names = {"HARD_LEAK": "계속 새던 스포", "PREFILTER_SKIP": "1차 필터가 넘긴 스포",
                   "UNSTABLE": "실행마다 결과 다름", "ERROR_ID": "사건 ID 오류", "OVERBLOCK": "과잉 차단",
                   "E2_HIDDEN": "맞는 추측(가려짐)", "E2_SHOWN": "맞는 추측(공개)",
                   "GUESS_KNOWN": "이미 본 회차를 추측처럼", "GUESS_UNKNOWN": "안 본 회차 추측",
                   "RELABELED": "정답 수정(스포 아님)", "REGRESSION": "잘 되던 스포",
                   "SAFE_AI": "스포 아님(AI 작성)", "SAFE_REAL": "스포 아님(실제 말투)"}
    group_html = ""
    if s.get("groups"):
        group_rows = "".join(f'<tr><td>{e(g)}</td><td>{e(group_names.get(g, ""))}</td>{stat_cells(x)}</tr>'
                             for g, x in s["groups"].items())
        group_html = ('<h2>선정 이유별 (note)</h2><div class="scroll"><table><tr><th>그룹</th><th>설명</th><th>개수</th>'
                      '<th>정답</th><th class="num">정상 비율</th><th class="num">스포 노출</th><th class="num">과잉 차단</th>'
                      f'<th class="num">평균 응답</th></tr>{group_rows}</table></div>')
    verdicts = ", ".join(f"{e(str(k))} {v}" for k, v in sorted(j["verdicts"].items(), key=lambda kv: -kv[1]))

    def row_html(r):
        grades = "".join(f"<td>{badge(r[f'v{v}'])}</td>" for v in viewers)
        bad = any(r[f"v{v}"] in ("LEAK", "OVERBLOCK") for v in viewers)
        exp = r["expected"] + (f' · {r["episode"]}화' if r["episode"] else "")
        if author_of(r) is not None:
            exp += f' · 작성자 {author_of(r)}화'
            if r["expected"] == "GUESS":
                exp += " → " + effective_expected(r)[0]
        detail = e(r["reason"] or "")
        if r.get("guess_description"):
            detail += f'<div class="m">추측 사건: {e(r["guess_description"])}</div>'
        if r["matched_description"]:
            detail += f'<div class="m">매칭: {e(r["matched_description"])}</div>'
        if r["prefilter_stage"]:
            sim = f' · 유사도 {r["prefilter_sim"]}' if r["prefilter_sim"] != "" else ""
            detail += f'<div class="m">1차 필터: {e(str(r["prefilter_stage"]))}{sim}</div>'
        run = "" if r["run"] == 1 else f' <small>#{r["run"]}</small>'
        return (f'<tr class="{"fail" if bad else ""}" data-fail="{int(bad)}"><td>{e(r["id"])}{run}</td>'
                f'<td>{e(r["category"])}</td><td class="msg">{e(r["message"])}</td><td>{e(exp)}</td>'
                f'<td>{e(str(r["verdict"]))}</td><td>{e(r["decision"])}</td>{grades}<td class="why">{detail}</td></tr>')

    body_rows = "".join(row_html(r) for r in rows)
    vh = "".join(f"<th>{v}화</th>" for v in viewers)
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>스포 판정 테스트</title>
<style>
:root{{--bg:#f6f7f9;--fg:#1d2330;--mute:#667085;--card:#fff;--line:#e4e7ec;--ok:#067647;--okb:#ecfdf3;--bad:#b42318;--badb:#fef3f2;--warn:#b54708;--warnb:#fffaeb;--accent:#3e4fd6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#11141b;--fg:#e6e8ee;--mute:#98a2b3;--card:#191d27;--line:#2a3040;--ok:#47cd89;--okb:#0e2a1d;--bad:#f97066;--badb:#3a1512;--warn:#fdb022;--warnb:#33240a;--accent:#8a98ff}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Segoe UI","Malgun Gothic",sans-serif;padding:24px 16px}}
.wrap{{max-width:1280px;margin:0 auto}}h1{{font-size:22px;margin:0 0 4px}}h2{{font-size:16px;margin:28px 0 10px}}
.meta{{color:var(--mute);font-size:13px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}}.k{{color:var(--mute);font-size:12px}}
.big{{font-size:26px;font-weight:700;margin:2px 0}}.sub{{font-size:12px;color:var(--mute)}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;font-size:13px}}th{{color:var(--mute);font-weight:600;background:var(--bg)}}
.num{{text-align:right}}.msg{{min-width:220px}}.why{{color:var(--mute);min-width:200px}}.m{{font-size:12px;margin-top:3px}}
.b{{display:inline-block;padding:1px 7px;border-radius:99px;font-size:12px;white-space:nowrap}}.b.ok{{background:var(--okb);color:var(--ok)}}
.b.bad{{background:var(--badb);color:var(--bad)}}.b.warn{{background:var(--warnb);color:var(--warn)}}b.bad,td.bad{{color:var(--bad)}}
tr.fail td{{background:color-mix(in srgb,var(--badb) 45%,transparent)}}.note{{color:var(--mute);font-size:13px}}
.scroll{{overflow-x:auto}}label{{font-size:13px;color:var(--mute);cursor:pointer}}
</style></head><body><div class="wrap">
<h1>스포일러 판정 테스트</h1>
<div class="meta">{e(meta["at"])} · 드라마 {meta["drama"]} · {e(meta["kb"])} · {e(meta["episodes"])}화 · 1차 필터 {e(meta["prefilter"])} · 후보 {meta.get("top_k", 20)}개 · 반복 {meta["repeat"]}회 · {e(str(meta["model"]))}<br>정책: {e(meta["policy"])}</div>
<h2>시청 진도별 결과</h2><div class="grid">{vcards}</div>
<p class="note">스포 노출 = 아직 안 본 회차의 내용이 보임 (가장 중요) · 과잉 차단 = 봐도 되는 채팅이 가려짐</p>
<h2>판정 품질</h2><div class="grid">{jcards}</div>
<p class="note">판정 분포: {verdicts}</p>{stab}
<h2>유형별</h2><div class="scroll"><table><tr><th>유형</th><th>설명</th><th>개수</th><th>정답</th><th class="num">정상 비율</th><th class="num">스포 노출</th><th class="num">과잉 차단</th><th class="num">평균 응답</th></tr>{cat_rows}</table></div>
{group_html}
<h2>전체 결과</h2><label><input type="checkbox" id="only"> 틀린 것만 보기</label>
<div class="scroll" style="margin-top:8px"><table id="t"><tr><th>ID</th><th>유형</th><th>채팅</th><th>정답</th><th>판정</th><th>처리</th>{vh}<th>이유</th></tr>{body_rows}</table></div>
</div><script>
document.getElementById('only').addEventListener('change',function(){{
  document.querySelectorAll('#t tr[data-fail]').forEach(function(tr){{tr.style.display=(this.checked&&tr.dataset.fail==='0')?'none':''}}.bind(this));
}});
</script></body></html>"""


# ------------------------------------------------------------------ main
def main(argv=None):
    load_env()
    p = argparse.ArgumentParser(description="SPOILOCK 채팅 스포일러 판정 테스트")
    p.add_argument("cmd", choices=["export", "draft", "run", "rescore"])
    p.add_argument("--server", default="http://127.0.0.1:8000")
    p.add_argument("--drama", type=int, default=1)
    p.add_argument("--kb", default="local-csy")
    p.add_argument("--episodes", default="1-6")
    p.add_argument("--title", default="Pioneer One", help="draft: 작품 제목")
    p.add_argument("--safe", type=int, default=30, help="draft: 스포 아닌 채팅 개수")
    p.add_argument("--cases", default=str(OUT_DIR / "cases.csv"))
    p.add_argument("--viewers", default="2,4,6")
    p.add_argument("--prefilter", default="on", choices=["on", "off", "shadow"])
    p.add_argument("--top-k", type=int, default=20, help="Gemini 에 보여줄 후보 사건 수 (Spring 기본 20)")
    p.add_argument("--prompt-version", default="", choices=["", "v1", "v2"],
                   help="판정 프롬프트 버전 (비우면 서버 설정 PROMPT_VERSION). v1/v2 비교용")
    p.add_argument("--tag", default="", help="결과 폴더 이름에 붙일 메모 (예: prompt2)")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sample", type=int, default=0, help="유형별로 고르게 N개만 뽑아 실행")
    p.add_argument("--seed", type=int, default=1, help="--sample 고정값 (같으면 같은 문장)")
    p.add_argument("--sleep", type=float, default=0.0, help="요청 사이 대기(초)")
    p.add_argument("--publish-no-match", action="store_true")
    p.add_argument("--no-finale", action="store_true",
                   help="보류를 마지막 회차 공개로 바꾸지 않음 (Spring hold-as-finale=false 와 같음)")
    p.add_argument("--no-unverified", action="store_true")
    p.add_argument("--author-episode", type=int, default=None,
                   help="author_episode 가 빈 문장의 작성자 시청 회차 (기본: 모름)")
    p.add_argument("--no-author-guess", action="store_true",
                   help="작성자 진도로 추측을 판단하지 않음 (이전 정책과 비교용)")
    p.add_argument("--rpm", type=float, default=14, help="Gemini 분당 호출 제한 (0 = 제한 없음)")
    p.add_argument("--retries", type=int, default=3, help="한도 초과 시 다시 시도 횟수")
    p.add_argument("--quota-wait", type=int, default=65, help="한도 초과 시 대기(초)")
    p.add_argument("--resume", help="이전 결과 폴더에서 이어서 실행 / rescore 대상")
    p.add_argument("--use-cache", action="store_true", help="이전 판정 캐시 재사용 (Gemini 호출 절약)")
    args = p.parse_args(argv)
    args.server = args.server.rstrip("/")
    args.episodes = parse_episodes(args.episodes)
    {"export": cmd_export, "draft": cmd_draft, "run": cmd_run, "rescore": cmd_rescore}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
