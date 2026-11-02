"""SPOILOCK local analysis API. Run: python -m uvicorn spoiler_server:app --port 8000.

This is a draft-event demo API, not a verified production spoiler classifier.
"""
import copy
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
# Process environment wins for server options. Gemini key/model keep their existing precedence.
load_dotenv(ROOT / ".env", encoding="utf-8-sig", override=False)
# Allow first-time model downloads; set to 1 explicitly when using only a cached model.
os.environ.setdefault("HF_HUB_OFFLINE", "0")

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import prefilter
GEMINI_TIMEOUT_S = float(os.getenv("GEMINI_TIMEOUT_S", "8"))
# 장문 메시지: 이 길이 이상이면 문장 단위로 나눠 사건을 추가 검색
LONG_MESSAGE_CHARS = int(os.getenv("LONG_MESSAGE_CHARS", "60"))
CHUNK_TOP_K = int(os.getenv("CHUNK_TOP_K", "8"))          # 문장 하나당 추가로 찾을 후보 수
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "30"))   # Gemini 에 보낼 후보 최대 개수
# 1차 필터 모드: off(사용 안 함) / shadow(판단만 기록, 검사는 전부 수행) / on(통과 채팅은 Gemini 생략)
PREFILTER_MODE = os.getenv("PREFILTER_MODE", "shadow")
# 유사도 통과 임계값. calibrate_prefilter.py 결과로 정한 뒤 설정. 비우면 유사도 통과 미사용
_sim = os.getenv("PREFILTER_SIM_THRESHOLD", "").strip()
PREFILTER_SIM_THRESHOLD = float(_sim) if _sim else None
MEMORY_CACHE_SIZE = 5000
TIMING_LOG = ROOT / "data" / "tests" / "server_timing.jsonl"

S: dict = {}                       # 서버 시작 시 준비되는 자원
_mem_cache: OrderedDict = OrderedDict()
_mem_lock = threading.Lock()
_log_lock = threading.Lock()
_gemini_pool = ThreadPoolExecutor(max_workers=16)


# ---------------------------------------------------------------- 시작 준비
@asynccontextmanager
async def lifespan(app: FastAPI):
    steps = {}
    t = time.perf_counter()
    import event_store as base
    import gemini_client
    steps["import"] = time.perf_counter() - t

    t = time.perf_counter()
    model = base.embedding_model()
    model.encode(["query: warmup"], normalize_embeddings=True, show_progress_bar=False)
    steps["embedding_model"] = time.perf_counter() - t

    t = time.perf_counter()
    col = base.get_collection()
    steps["chromadb"] = time.perf_counter() - t

    t = time.perf_counter()
    client, gemini_model = gemini_client.get_client()
    steps["gemini_client"] = time.perf_counter() - t

    S.update(base=base, client=client, gemini_model=gemini_model, col=col, model=model)
    # 자막 분석·사건 관리 작업이 판정 서버와 같은 Chroma 컬렉션·임베딩 모델을 쓰도록 연결
    try:
        import ingest_jobs
        ingest_jobs.bind(lambda: S)
    except ImportError:
        pass
    print("[startup] " + ", ".join(f"{k}={v:.1f}s" for k, v in steps.items()), flush=True)
    print(f"[startup] ready: model={gemini_model}, events={col.count()}", flush=True)
    print(f"[startup] prefilter mode={PREFILTER_MODE}, sim_threshold={PREFILTER_SIM_THRESHOLD}, "
          f"keywords={len(prefilter._KEYWORDS)}", flush=True)
    yield
    _gemini_pool.shutdown(wait=False)


app = FastAPI(title="SPOILOCK spoiler judge", lifespan=lifespan)

# 관리자 영상 분석 API (video_api.py / video_pipeline.py 가 있고 python-multipart 가 설치된 경우만 켜짐)
try:
    import video_api
    import video_pipeline
    app.include_router(video_api.router)
    video_pipeline.recover_interrupted_jobs()
    print(f"[startup] video api enabled: whisper={video_pipeline.WHISPER_MODEL}/"
          f"{video_pipeline.WHISPER_DEVICE}", flush=True)
except Exception as e:  # 영상 기능이 없어도 스포일러 판정 서버는 정상 동작
    print(f"[startup] video api disabled: {type(e).__name__}: {e}", flush=True)

# 관리자 자막 분석·사건 관리 API (ingest_*.py / events_*.py)
try:
    import ingest_api
    import ingest_jobs
    import events_api
    app.include_router(ingest_api.router)
    app.include_router(events_api.router)
    ingest_jobs.recover_interrupted_jobs()
    print("[startup] subtitle ingest / event admin api enabled", flush=True)
except Exception as e:
    print(f"[startup] subtitle ingest / event admin api disabled: {type(e).__name__}: {e}", flush=True)

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()


# ---------------------------------------------------------------- 요청/응답
class JudgeRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    message_id: int | None = Field(None, description="Spring chat_messages.id (로그용)")
    drama_id: int = 1
    kb_version: str = "kb1"
    episodes: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    retrieval: str = Field("vector", pattern="^(vector|all)$")
    top_k: int = Field(20, ge=1, le=100)
    include_evidence: bool = Field(True, description="False면 프롬프트에서 evidence 인용문 제외(실험용)")
    refresh: bool = Field(False, description="True면 캐시 무시하고 Gemini 재판정")
    watched_episode: int | None = Field(
        None, ge=0, description="데모용. 실서비스에서는 Spring이 required_episode_number로 사용자별 판정")
    prefilter_mode: str | None = Field(
        None, pattern="^(off|shadow|on)$", description="이 요청만 1차 필터 모드 변경(테스트용)")
    debug: bool = Field(False, description="True면 매칭 사건 설명 포함(테스트 페이지용, 사용자에게 노출 금지)")
    reason_mode: str = Field(
        "full", pattern="^(full|short|none)$",
        description="Gemini 판단 이유 길이. full=기존, short=15자 이내, none=생성 안 함(출력 토큰↓ → 속도↑)")


# ---------------------------------------------------------------- 판별
def _log(row: dict):
    TIMING_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock, TIMING_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


REASON_SUFFIX = {
    "full": "",
    "short": "\nKeep the reason extremely short: at most 15 Korean characters.",
    "none": '\nDo not explain. Always return reason as an empty string "".',
}


# 판정 보강 (사건 DB가 해당 회차의 줄거리를 모두 담고 있다는 전제)
#  - 바꿔 말한 문장도 핵심 사실이 같으면 MATCH → UNCERTAIN 감소
#  - reveals_plot: 줄거리 정보를 '사실로' 드러내는가 (감상·질문·추측은 false)
PLOT_RULES_SUFFIX = """

Additional rules:
- Treat the event candidates as the complete list of plot facts for these episodes.
- Paraphrases, summaries, translations, and partial descriptions that convey the same core fact
  of a candidate (who did what, what happened, an identity, a relationship, a death, an outcome)
  count as MATCH. For each fact, pick the closest candidate.
- A message may reveal several different facts, possibly from different episodes. Then return
  MATCH, put the most important one in event_id, and list EVERY matching candidate in
  matched_event_ids (including event_id). Revealing several facts is NOT a reason for UNCERTAIN.
- matched_event_ids must be [] unless verdict is MATCH.
- Use UNCERTAIN only when two or more candidates fit one fact equally well, or when the message
  states a plot fact but is too vague to tell which candidate it refers to.
- Judge by the information a reader learns, not by the tone. A reaction or emotion that
  describes what happened still reveals the fact and is a MATCH.
  e.g. "CCTV에 그림자 보이는 거 소름이었어" reveals that a shadow appears on the CCTV,
  so it matches a candidate about a shadowy figure caught on CCTV.
  e.g. "걔 죽는 장면 진짜 슬펐다" reveals the death → MATCH.
- Pure reactions with no story information ("소름", "대박", "이번 화 미쳤다", "연기 좋다"),
  questions and guesses/predictions ("~일까?", "~인 것 같아", "I think", "maybe") are NOT a MATCH,
  even if they name a character or the show; return NO_MATCH for them.

Also set reveals_plot (boolean) for the chat message itself:
- true if the message lets a reader learn story information: what happened, what was seen in a
  scene, who did what, identities, relationships, deaths, outcomes, or the ending - even when it
  is wrapped in a reaction or emotion ("~해서 소름", "~하는 장면 슬펐어").
- false only for reactions without story content, opinions, praise/criticism, questions, guesses,
  predictions, greetings, and unrelated small talk."""


def _system_prompt(reason_mode):
    return S["base"].MATCH_SYSTEM + REASON_SUFFIX[reason_mode] + PLOT_RULES_SUFFIX


def _match_schema():
    """기존 MATCH_SCHEMA 에 reveals_plot(boolean), matched_event_ids(string[]) 를 필수로 추가한 복사본"""
    schema = copy.deepcopy(S["base"].MATCH_SCHEMA)
    props = schema.setdefault("properties", {})
    props["reveals_plot"] = {"type": "boolean"}
    props["matched_event_ids"] = {"type": "array", "items": {"type": "string"}}
    required = schema.setdefault("required", [])
    for field in ("reveals_plot", "matched_event_ids"):
        if field not in required:
            required.append(field)
    return schema


def _split_answer(raw):
    """Gemini 응답에서 추가 필드를 분리. 기존 validate_match_answer 는 원래 필드만 허용하므로"""
    if not isinstance(raw, dict):
        return raw, None
    core = {k: v for k, v in raw.items() if k not in EXTRA_FIELDS}
    reveals = raw.get("reveals_plot")
    return core, (reveals if isinstance(reveals, bool) else None)


EXTRA_FIELDS = ("reveals_plot", "matched_event_ids")


def _matched_ids(raw):
    """Gemini 가 고른 여러 사건 ID (문자열 목록만 인정)"""
    ids = raw.get("matched_event_ids") if isinstance(raw, dict) else None
    if not isinstance(ids, list):
        return []
    return [i for i in ids if isinstance(i, str) and i]


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


def _retrieve_candidates(args, req):
    """
    기본: 메시지 전체로 top_k 검색 (기존과 동일).
    장문: 문장마다 CHUNK_TOP_K 개를 추가 검색해 합침 → 여러 사건이 섞인 글에서 정답 사건이 빠지지 않게.
    반환: (후보 목록, 나눈 구간 수)
    """
    base = S["base"]
    candidates = list(base.candidates_for(args, collection=S["col"], model=S["model"]))
    chunks = _split_long_message(req.message) if req.retrieval == "vector" else []
    if not chunks:
        return candidates, 0

    seen = {c["event_id"] for c in candidates}
    extra = []
    for chunk in chunks:
        chunk_args = SimpleNamespace(**{**vars(args), "message": chunk, "top_k": CHUNK_TOP_K})
        for c in base.candidates_for(chunk_args, collection=S["col"], model=S["model"]):
            if c["event_id"] not in seen:
                seen.add(c["event_id"])
                extra.append(c)
    # 전체 검색 상위 10개 → 문장별로 새로 찾은 후보 → 전체 검색 나머지, 최대 MAX_CANDIDATES 개
    merged = candidates[:10] + extra + candidates[10:]
    return merged[:MAX_CANDIDATES], len(chunks)


def _call_gemini(prompt_candidates, message, system):
    base = S["base"]
    return S["client"].models.generate_content(
        model=S["gemini_model"],
        contents=json.dumps({"message": message, "candidates": prompt_candidates}, ensure_ascii=False),
        config={
            "system_instruction": system,
            "temperature": 0,
            "response_mime_type": "application/json",
            "response_json_schema": _match_schema(),
        },
    )


def _best_similarity(req, episodes):
    """등록된 사건(시청 회차와 무관하게 전체 검색 범위)과의 최고 코사인 유사도"""
    vec = S["model"].encode(["query: " + req.message], normalize_embeddings=True,
                            show_progress_bar=False).tolist()[0]
    where = {"$and": [{"drama_id": req.drama_id}, {"kb_version": req.kb_version},
                      {"source_episode": {"$in": episodes}},
                      {"verification_status": {"$ne": "REJECTED"}}]}
    d = S["col"].query(query_embeddings=[vec], where=where, n_results=1, include=["distances"])
    if not d["ids"][0]:
        return None, None
    return round(1 - d["distances"][0][0], 4), d["ids"][0][0]   # cosine space


@app.get("/")
def chat_tester():
    """채팅 형식 테스트 페이지: http://localhost:8000"""
    return FileResponse(ROOT / "web" / "chat_tester.html")


@app.get("/health")
def health():
    return {"ready": "base" in S, "gemini_model": S.get("gemini_model"),
            "events": S["col"].count() if "col" in S else None,
            "prefilter_mode": PREFILTER_MODE, "prefilter_sim_threshold": PREFILTER_SIM_THRESHOLD}


@app.post("/admin/reload-keywords")
def reload_keywords(x_admin_token: str | None = Header(None)):
    """config/prefilter_keywords.txt 수정 후 서버 재시작 없이 다시 읽기"""
    if ADMIN_TOKEN and x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="관리자 토큰이 올바르지 않습니다.")
    return {"keywords": prefilter.reload_keywords()}


@app.post("/judge")
def judge(req: JudgeRequest):
    """메시지 1건 판별. sync 함수라 FastAPI가 스레드풀에서 동시 처리합니다."""
    base = S["base"]
    tm = {}
    t_all = time.perf_counter()
    episodes = sorted(set(req.episodes))
    args = SimpleNamespace(
        message=req.message, drama_id=req.drama_id, kb_version=req.kb_version,
        episodes=episodes, retrieval=req.retrieval, top_k=req.top_k,
    )

    # 0) 1차 필터
    mode = req.prefilter_mode or PREFILTER_MODE
    pf = None
    if mode != "off":
        t = time.perf_counter()
        best_sim, best_id = None, None
        if prefilter.rule_pass(req.message) is None:          # 규칙 통과면 유사도 계산 생략
            best_sim, best_id = _best_similarity(req, episodes)
        pf = prefilter.decide(req.message, best_sim, PREFILTER_SIM_THRESHOLD)
        pf.update(mode=mode, best_sim=best_sim, best_event_id=best_id)
        tm["prefilter"] = time.perf_counter() - t

        if mode == "on" and pf["decision"] == "SKIP":
            tm["total"] = time.perf_counter() - t_all
            result = {
                "message_id": req.message_id,
                # SKIPPED_GENERAL: 일반 대화 확인 / SKIPPED_UNRELATED: 등록 사건과 관련 낮음
                "analysis_verdict": "SKIPPED_GENERAL" if pf["stage"] in ("rule", "daily") else "SKIPPED_UNRELATED",
                "matched_event_id": None,
                "required_episode_number": None,
                "episode_basis": "UNVERIFIED_SOURCE_EPISODE_FOR_DEMO_ONLY",
                "needs_review": False,
                "reason": None, "error": None,
                "reveals_plot": False if pf["stage"] in ("rule", "daily") else None,
                # Spring 호환용 (Spring 은 about_story=false 를 '공개 가능'으로 해석)
                "about_story": False if pf["stage"] in ("rule", "daily") else None,
                "gemini_model": None, "candidate_count": 0, "match_cache": "NONE",
                "prefilter": pf,
            }
            if req.watched_episode is not None:
                result["watched_episode_number"] = req.watched_episode
                result["visibility"] = "VISIBLE"
            result["timings_ms"] = {k: round(v * 1000, 1) for k, v in tm.items()}
            _log({"at": datetime.now().isoformat(timespec="seconds"), "message": req.message, **result})
            return result

    # 1) 후보 사건 검색 (원본 candidates_for 그대로 사용)
    t = time.perf_counter()
    try:
        candidates, chunk_count = _retrieve_candidates(args, req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    tm["retrieval"] = time.perf_counter() - t
    if chunk_count:
        tm["retrieval_chunks"] = chunk_count

    # 1-1) 이 범위에 등록된 사건이 하나도 없음 → 비교 대상이 없으므로 NO_MATCH (Spring 은 보류 처리)
    if not candidates:
        tm["total"] = time.perf_counter() - t_all
        result = {
            "message_id": req.message_id, "analysis_verdict": "NO_MATCH",
            "matched_event_id": None, "matched_event_ids": [], "matched_episodes": [],
            "required_episode_number": None, "episode_basis": base.UNVERIFIED_BASIS,
            "needs_review": True, "reason": "등록된 사건 없음", "reveals_plot": None,
            "about_story": None, "error": None, "gemini_model": None, "candidate_count": 0,
            "match_cache": "NONE", "prefilter": pf,
            "timings_ms": {k: (round(v * 1000, 1) if isinstance(v, float) else v) for k, v in tm.items()},
        }
        _log({"at": datetime.now().isoformat(timespec="seconds"), "message": req.message, **result})
        return result

    prompt_candidates = []
    for e in candidates:
        item = {"event_id": e["event_id"], "description": e["description"],
                "fact_status": e["fact_status"]}
        if req.include_evidence:
            item["evidence"] = json.loads(e["evidence_json"])
            # 자막 + 영상 분석 사건: 화면 근거(장면 설명)도 함께 → 장면으로만 드러난 사건 판정에 사용
            visual = json.loads(e.get("visual_evidence_json") or "[]")
            if visual:
                item["visual_evidence"] = [v.get("description", "") for v in visual]
        prompt_candidates.append(item)
    if req.retrieval == "all":
        prompt_candidates.sort(key=lambda item: item["event_id"])

    # 2) 기존 판별 캐시 키 형식 유지
    key_content = {
        "message": base.compact(req.message),
        "drama_id": req.drama_id,
        "source_episode": "MULTI:" + ",".join(map(str, episodes)),
        "kb_version": req.kb_version,
        "retrieval": req.retrieval,
        "top_k": req.top_k if req.retrieval == "vector" else None,
        "model": S["gemini_model"],
        "system": _system_prompt(req.reason_mode),   # 기존 full 프롬프트 유지
        "schema": _match_schema(),
        "candidates": prompt_candidates,
    }
    key_text = json.dumps(key_content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = sha256(key_text.encode("utf-8")).hexdigest()
    cache_file = ROOT / "data" / "events" / "demo_match_cache" / f"{digest}.json"
    id_map = {e["event_id"]: e for e in candidates}

    answer, cache_status, error, reveals_plot = None, "MISS", None, None
    extra_ids = []
    if not req.refresh:
        with _mem_lock:
            if digest in _mem_cache:
                _mem_cache.move_to_end(digest)
                answer, reveals_plot = _split_answer(_mem_cache[digest])
                extra_ids = _matched_ids(_mem_cache[digest])
                cache_status = "HIT_MEMORY"
        if answer is None and cache_file.is_file():
            try:
                raw = json.loads(cache_file.read_text(encoding="utf-8")).get("answer")
                answer, reveals_plot = _split_answer(raw)
                extra_ids = _matched_ids(raw)
                base.validate_match_answer(answer, id_map)
                cache_status = "HIT_FILE"
            except Exception:
                answer, reveals_plot = None, None

    # 3) Gemini 판정 (시간 초과·오류 → fail-closed)
    if answer is None:
        t = time.perf_counter()
        try:
            fut = _gemini_pool.submit(_call_gemini, prompt_candidates, req.message,
                                      _system_prompt(req.reason_mode))
            response = fut.result(timeout=GEMINI_TIMEOUT_S)
            if not response.text:
                raise ValueError("Gemini returned an empty response")
            raw_answer = json.loads(response.text)
            answer, reveals_plot = _split_answer(raw_answer)
            extra_ids = _matched_ids(raw_answer)
            base.validate_match_answer(answer, id_map)
            u = getattr(response, "usage_metadata", None)
            if u is not None:
                tm["tokens_in"] = getattr(u, "prompt_token_count", None)
                tm["tokens_out"] = getattr(u, "candidates_token_count", None)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_name(f"{digest}.{uuid.uuid4().hex}.tmp")
            tmp.write_text(json.dumps({"answer": raw_answer}, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                tmp.replace(cache_file)
            except OSError:            # 같은 문장이 동시에 들어와 이미 저장된 경우
                tmp.unlink(missing_ok=True)
            cache_status = "REFRESHED" if req.refresh else "MISS_SAVED"
        except FutureTimeout:
            error = f"Gemini timeout > {GEMINI_TIMEOUT_S}s"
            answer, reveals_plot = None, None
        except Exception as e:
            error = f"{type(e).__name__}: {e}"[:300]
            answer, reveals_plot = None, None
        tm["gemini"] = time.perf_counter() - t

    # 4) 결과 정리
    if error is not None:
        verdict, matched = "ERROR", None
        cache_status = "NONE"
    else:
        with _mem_lock:
            _mem_cache[digest] = {**answer, "reveals_plot": reveals_plot, "matched_event_ids": extra_ids}
            if len(_mem_cache) > MEMORY_CACHE_SIZE:
                _mem_cache.popitem(last=False)
        verdict = answer["verdict"]
        matched = None

    # MATCH: Gemini 가 고른 사건 전부 중 '가장 늦은 공개 회차' 기준 (여러 회차가 섞인 장문 대응)
    #  공개 회차 = 검수된 사건은 first_reveal_episode, 미검수는 source_episode
    matched_list = []
    if error is None and verdict == "MATCH":
        ids = [answer["event_id"]] + [i for i in extra_ids if i != answer["event_id"]]
        matched_list = [id_map[i] for i in dict.fromkeys(ids) if i in id_map]
        if matched_list:
            matched = max(matched_list, key=base.effective_episode)

    # MATCH 이외(NO_MATCH/UNCERTAIN/ERROR)는 회차를 정할 수 없으므로 fail-closed
    required_episode = base.effective_episode(matched) if matched else None
    # 일치한 사건이 모두 검수된 경우에만 검증된 기준. 하나라도 미검수면 보수적으로 UNVERIFIED
    all_verified = bool(matched_list) and all(base.is_verified(e) for e in matched_list)
    episode_basis = base.VERIFIED_BASIS if all_verified else base.UNVERIFIED_BASIS
    result = {
        "message_id": req.message_id,
        "analysis_verdict": verdict,                  # MATCH / NO_MATCH / UNCERTAIN / ERROR
        "matched_event_id": matched["event_id"] if matched else None,   # 가장 늦은 회차의 사건
        "matched_event_ids": [e["event_id"] for e in matched_list],      # 일치한 사건 전부
        "matched_episodes": sorted({base.effective_episode(e) for e in matched_list}),
        "required_episode_number": required_episode,  # chat_messages.required_episode_number
        "episode_basis": episode_basis,
        "needs_review": matched is None,              # True면 Spring은 HIDDEN/대기 처리
        "reason": answer.get("reason") if answer else None,   # 관리자용. 사용자에게 보내지 말 것
        # 줄거리 정보를 사실로 드러내는지 (Gemini 판단). false = 감상·질문·추측·잡담
        "reveals_plot": reveals_plot,
        # Spring 호환용 별칭: Spring 은 about_story=false 를 '공개 가능'으로 해석
        "about_story": reveals_plot,
        "error": error,
        "gemini_model": S["gemini_model"],
        "candidate_count": len(candidates),
        "match_cache": cache_status,
        "prefilter": pf,       # shadow 모드: 'SKIP이었을 텐데 실제 판정은?'을 비교하는 기록
    }
    if req.debug and matched:
        result["matched_event_description"] = matched["description"]
        result["matched_event_descriptions"] = [
            f'{base.effective_episode(e)}화'
            f'{"" if base.is_verified(e) else "(미검수)"}: {e["description"]}' for e in matched_list]
    if req.watched_episode is not None:   # 데모 화면용 공개 판정 (원본 규칙 그대로)
        visibility, _ = base.decide_visibility(matched, req.watched_episode)
        result["watched_episode_number"] = req.watched_episode
        result["visibility"] = visibility

    tm["total"] = time.perf_counter() - t_all
    result["timings_ms"] = {k: (round(v * 1000, 1) if isinstance(v, float) else v) for k, v in tm.items()}
    _log({"at": datetime.now().isoformat(timespec="seconds"), "message": req.message, **result})
    return result