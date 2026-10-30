"""
SPOILOCK 자막 분석 작업 관리 (video_pipeline.py 와 같은 구조)

상태 흐름: QUEUED → PARSING → EXTRACTING → INDEXING → DONE / FAILED
  - 작업은 한 번에 하나씩 (Gemini 추출은 판정용 _gemini_pool 과 별도 스레드에서 실행 → 채팅 판정이 밀리지 않음)
  - 상태: data/ingest_jobs/<job_id>/status.json  (서버 재시작 후에도 조회)
  - 추출 체크포인트: data/ingest_jobs/<job_id>/extract_progress.json
    → 실패/재시작 후 retry 하면 끝난 배치는 건너뛰고 이어서 진행
  - 완료 시 data/kb/d{drama_id}/ep{NN}/ 에 source.srt, chunks.json, events_draft.json 최신본 저장

영상 함께 분석 (자막 + 영상 화면)
  create_job(..., video_path=...) 로 영상을 주면 배치마다 그 시간대 영상 구간을 Gemini 가 함께 보고
  화면에서만 드러나는 사건도 추출 (ingest_video.VideoBatchExtractor). 사건에 화면 근거(MM:SS)가 붙음.
  영상 사본은 작업이 끝나면(DONE) 지우고, 실패 시에는 재시도를 위해 남겨둠.

CLI (서버 없이 로컬 적재·점검)
  python -m ingest_jobs run --srt data/raw/ep01.srt --drama-id 1 --episode-id 1 --episode 1 [--replace]
  python -m ingest_jobs check --srt data/raw/ep01.srt      # 파싱·청크만 확인 (Gemini 호출 없음)

환경변수 (.env)
  KB_VERSION=kb1                  # 기본 KB 버전
  EXTRACT_GEMINI_MODEL=           # 비우면 판정과 같은 모델 사용
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

import ingest_core as core

ROOT = Path(__file__).resolve().parent
# CLI 로 직접 실행할 때도 .env 를 읽도록 (spoiler_server 와 같은 규칙: 프로세스 환경변수 우선)
load_dotenv(ROOT / ".env", encoding="utf-8-sig", override=False)
JOBS_DIR = ROOT / "data" / "ingest_jobs"
DEFAULT_KB_VERSION = os.getenv("KB_VERSION", "kb1").strip() or "kb1"
EXTRACT_GEMINI_MODEL = os.getenv("EXTRACT_GEMINI_MODEL", "").strip() or None
VIDEO_RESOLUTION = os.getenv("INGEST_VIDEO_RESOLUTION", "low").strip() or "low"
VIDEO_CLIP_MODE = os.getenv("INGEST_VIDEO_CLIP_MODE", "api").strip() or "api"
VIDEO_BATCH_SLEEP = float(os.getenv("INGEST_VIDEO_BATCH_SLEEP", "4"))

STATUS_QUEUED = "QUEUED"
STATUS_PARSING = "PARSING"
STATUS_EXTRACTING = "EXTRACTING"
STATUS_INDEXING = "INDEXING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"
ACTIVE_STATUSES = {STATUS_QUEUED, STATUS_PARSING, STATUS_EXTRACTING, STATUS_INDEXING}

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest-job")
_lock = threading.Lock()          # status.json 쓰기
_create_lock = threading.Lock()   # 같은 회차 중복 작업 검사 + 생성
_provider = None


class JobConflict(Exception):
    """같은 회차 작업이 진행 중이거나 이미 사건이 있음 (HTTP 409)"""


# ---------------------------------------------------------------- 공유 자원
def bind(provider):
    """
    spoiler_server 의 lifespan 에서 호출: ingest_jobs.bind(lambda: S)
    provider() 는 col(사건 컬렉션), model(임베딩), client, gemini_model 키를 가진 dict 반환.
    판정 서버와 같은 Chroma 컬렉션·임베딩 모델 인스턴스를 공유하기 위함.
    """
    global _provider
    _provider = provider


def _resources() -> dict:
    if _provider is not None:
        res = _provider()
        if "col" in res:
            return res
    # CLI 등 서버 밖에서 실행할 때
    import event_store as base
    import gemini_client
    client, gemini_model = gemini_client.get_client()
    return {"col": base.get_collection(), "model": base.embedding_model(),
            "client": client, "gemini_model": gemini_model}


# ---------------------------------------------------------------- 상태 파일
def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _write_status(job_id: str, /, **fields) -> dict:
    with _lock:
        path = _job_dir(job_id) / "status.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        data.update(fields)
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return data


def get_job(job_id: str) -> dict | None:
    if not job_id or any(c in job_id for c in "/\\.") or len(job_id) > 64:
        return None
    path = _job_dir(job_id) / "status.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_jobs(limit: int = 50, drama_id: int | None = None, episode: int | None = None) -> list[dict]:
    if not JOBS_DIR.is_dir():
        return []
    jobs = []
    for d in JOBS_DIR.iterdir():
        st = d / "status.json"
        if not st.is_file():
            continue
        try:
            job = json.loads(st.read_text(encoding="utf-8"))
        except Exception:
            continue
        if drama_id is not None and job.get("drama_id") != drama_id:
            continue
        if episode is not None and job.get("episode") != episode:
            continue
        jobs.append(job)
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs[:limit]


def _active_job_for(drama_id, episode, kb_version, exclude=None) -> dict | None:
    for job in list_jobs(limit=10_000, drama_id=drama_id, episode=episode):
        if job.get("kb_version") == kb_version and job.get("status") in ACTIVE_STATUSES \
                and job.get("job_id") != exclude:
            return job
    return None


# ---------------------------------------------------------------- 작업 생성·복구·재시도
def create_job(raw_srt: bytes, *, drama_id: int, episode_id: int, episode: int,
               kb_version: str = DEFAULT_KB_VERSION, language: str = "en",
               original_name: str = "", replace: bool = False,
               video_path: Path | None = None, characters: list[str] | None = None,
               source_mode: str | None = None) -> dict:
    """
    입력 검증(인코딩·SRT 형식)은 즉시 수행 → 잘못된 파일은 IngestError 로 바로 거절.
    Gemini 추출·색인은 백그라운드 작업으로 넘김.

    video_path: 자막과 시간이 맞는 회차 영상. 주면 작업 폴더로 옮겨 자막 + 영상 화면으로 분석
    characters: 주요 인물 이름 (영상 분석 시 화면 속 인물과 이름 연결에 사용)
    source_mode: 색인에 남길 분석 원본. 비우면 영상 유무로 결정 (SUBTITLE / SUBTITLE_VIDEO)
    """
    text = core.decode_srt(raw_srt)
    rows, skipped = core.parse_srt_text(text)

    with _create_lock:
        active = _active_job_for(drama_id, episode, kb_version)
        if active:
            raise JobConflict(f"같은 회차의 분석 작업이 진행 중입니다: {active['job_id']}")
        if not replace and core.existing_events(_resources()["col"], drama_id, episode, kb_version):
            raise JobConflict("이미 사건이 등록된 회차입니다. 교체하려면 replace 옵션을 켜 주세요.")

        job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        d = _job_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        # 인코딩을 UTF-8 로 통일해 저장 (원본 파일명은 경로에 쓰지 않음)
        # 줄바꿈을 \n 으로 통일하고 바이트로 저장
        # (write_text 는 Windows 에서 \n 을 \r\n 으로 바꿔 원본의 \r\n 이 \r\r\n 이 됨)
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        (d / "source.srt").write_bytes(normalized.encode("utf-8"))

        video_file = None
        if video_path is not None:
            suffix = Path(video_path).suffix.lower() if Path(video_path).suffix else ".mp4"
            video_file = d / f"source_video{suffix}"
            shutil.move(str(video_path), video_file)

        mode = source_mode or ("SUBTITLE_VIDEO" if video_file else "SUBTITLE")
        if mode not in core.SOURCE_MODES:
            raise core.IngestError(f"알 수 없는 분석 원본입니다: {mode}")
        now = datetime.now().isoformat(timespec="seconds")
        _write_status(
            job_id, job_id=job_id, status=STATUS_QUEUED, progress=0.0,
            drama_id=drama_id, episode_id=episode_id, episode=episode, kb_version=kb_version,
            language=language, replace=replace, original_name=original_name,
            size_bytes=len(raw_srt), subtitle_count=len(rows), skipped_blocks=skipped,
            source_mode=mode, video_file=video_file.name if video_file else None,
            characters=[c for c in (characters or []) if c][:30],
            created_at=now, error=None,
        )
    _executor.submit(_run_job, job_id)
    return get_job(job_id)


def recover_interrupted_jobs():
    """서버가 작업 도중 꺼졌던 경우 FAILED 로 표시. retry 하면 체크포인트부터 이어서 진행"""
    for job in list_jobs(limit=10_000):
        if job.get("status") in ACTIVE_STATUSES:
            _write_status(job["job_id"], status=STATUS_FAILED,
                          error="서버 재시작으로 작업이 중단되었습니다. 재시도하면 이어서 진행합니다.")


def retry_job(job_id: str) -> dict | None:
    with _create_lock:
        job = get_job(job_id)
        if job is None or job.get("status") != STATUS_FAILED:
            return job
        active = _active_job_for(job["drama_id"], job["episode"], job["kb_version"], exclude=job_id)
        if active:
            raise JobConflict(f"같은 회차의 다른 작업이 진행 중입니다: {active['job_id']}")
        _write_status(job_id, status=STATUS_QUEUED, error=None)
    _executor.submit(_run_job, job_id)
    return get_job(job_id)


# ---------------------------------------------------------------- 작업 실행
def _fingerprint(chunks, gemini_model, video=None) -> str:
    payload = json.dumps({"chunks": chunks, "model": gemini_model, "video": video,
                          "extract": core.DEFAULT_EXTRACT_SETTINGS}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _default_video_extractor(client, model, video_path: Path, workdir: Path):
    import ingest_video
    return ingest_video.VideoBatchExtractor(
        client, model, video_path, clip_mode=VIDEO_CLIP_MODE, resolution=VIDEO_RESOLUTION,
        fps=None, workdir=workdir)


# 테스트에서 가짜 영상 분석기로 바꿔 끼울 수 있게 분리
_video_extractor_factory = _default_video_extractor


def _run_job(job_id: str):
    job = get_job(job_id)
    d = _job_dir(job_id)
    drama_id, episode_id, episode = job["drama_id"], job["episode_id"], job["episode"]
    kb_version, language = job["kb_version"], job["language"]
    try:
        started = time.perf_counter()
        _write_status(job_id, status=STATUS_PARSING, progress=0.0)

        # 1) SRT → 청크
        rows, _ = core.parse_srt_text((d / "source.srt").read_bytes().decode("utf-8"))
        chunks = core.build_chunks(rows, drama_id=drama_id, episode_id=episode_id, episode=episode,
                                   kb_version=kb_version, language=language,
                                   source_name=job.get("original_name") or "source.srt")
        core.save_json(d / "chunks.json", chunks)

        # 2) Gemini 사건 추출 (체크포인트 재개)
        res = _resources()
        gemini_model = EXTRACT_GEMINI_MODEL or res["gemini_model"]
        video_file = d / job["video_file"] if job.get("video_file") else None
        if video_file is not None and not video_file.is_file():
            raise core.IngestError("작업 폴더에 영상 파일이 없습니다. 영상과 함께 다시 분석을 요청해 주세요.")
        fingerprint = _fingerprint(chunks, gemini_model,
                                   {"file": job.get("video_file"), "resolution": VIDEO_RESOLUTION,
                                    "characters": job.get("characters")} if video_file else None)
        progress_path = d / "extract_progress.json"
        state = None
        if progress_path.is_file():
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            if saved.get("fingerprint") == fingerprint:
                state = saved["state"]
        state = state or core.new_extract_state()
        total = len(core.batch_ranges(len(chunks), core.DEFAULT_EXTRACT_SETTINGS["batch_size"],
                                      core.DEFAULT_EXTRACT_SETTINGS["overlap"]))
        _write_status(job_id, status=STATUS_EXTRACTING, chunk_count=len(chunks),
                      total_batches=total, done_batches=state["next_batch"],
                      progress=round(0.9 * state["next_batch"] / total, 3), gemini_model=gemini_model)

        def on_batch_done(st, done, total_batches):
            core.save_json(progress_path, {"fingerprint": fingerprint, "state": st})
            _write_status(job_id, done_batches=done, progress=round(0.9 * done / total_batches, 3),
                          candidate_count=len(st["events"]), rejected_quotes=len(st["rejected"]))

        extractor = None
        try:
            batch_fn, between = None, 0
            if video_file is not None and state["next_batch"] < total:
                # 영상은 Gemini 에 한 번만 올리고, 배치마다 그 시간대 구간만 지정해서 보게 함
                _write_status(job_id, status=STATUS_EXTRACTING, video_status="UPLOADING")
                extractor = _video_extractor_factory(res["client"], gemini_model, video_file, d)
                extractor.prepare()
                _write_status(job_id, video_status="READY")
                characters = job.get("characters") or []

                def batch_fn(batch):
                    (accepted, rejected), _ = extractor.extract(episode, batch, characters)
                    return accepted, rejected
                between = VIDEO_BATCH_SLEEP

            state = core.extract_events(res["client"], gemini_model, chunks, episode=episode,
                                        language=language, state=state, on_batch_done=on_batch_done,
                                        batch_fn=batch_fn, between_batches=between)
        finally:
            if extractor is not None:
                usage = getattr(extractor, "usage", None)
                extractor.close()
                if usage:
                    _write_status(job_id, video_usage=usage)
        draft = {
            "status": "DRAFT_NEEDS_REVIEW", "drama_id": drama_id, "episode_id": episode_id,
            "episode_number": episode, "kb_version": kb_version, "model": gemini_model,
            "job_id": job_id, "source_mode": job.get("source_mode", "SUBTITLE"),
            "source_chunks": len(chunks), "event_count": len(state["events"]),
            "events": state["events"], "rejected": state["rejected"],
        }
        core.save_json(d / "events_draft.json", draft)

        # 3) Chroma 색인 (검수·수동 사건 보존)
        _write_status(job_id, status=STATUS_INDEXING, progress=0.95)
        result = core.index_extracted_events(
            res["col"], res["model"], state["events"], drama_id=drama_id, episode_id=episode_id,
            episode=episode, kb_version=kb_version, run_tag=job_id[-6:], replace=job["replace"],
            source_mode=job.get("source_mode", "SUBTITLE"))

        # 4) 회차 폴더에 최신본 보관 (근거 검증·관리 화면용)
        ep_dir = core.episode_dir(drama_id, episode, kb_version)
        ep_dir.mkdir(parents=True, exist_ok=True)
        for name in ("source.srt", "chunks.json", "events_draft.json"):
            shutil.copyfile(d / name, ep_dir / f"{name}.tmp")
            (ep_dir / f"{name}.tmp").replace(ep_dir / name)

        # 5) 영상 사본 정리 (끝난 작업은 재시도하지 않으므로 디스크 공간 확보)
        if video_file is not None:
            video_file.unlink(missing_ok=True)

        _write_status(job_id, status=STATUS_DONE, progress=1.0,
                      event_count=result["indexed"], replaced_events=result["replaced"],
                      kept_events=result["kept"], duplicate_events=result["skipped_as_duplicate"],
                      rejected_quotes=len(state["rejected"]),
                      elapsed_sec=round(time.perf_counter() - started, 1))
    except Exception as e:
        _write_status(job_id, status=STATUS_FAILED, error=f"{type(e).__name__}: {e}"[:500],
                      traceback=traceback.format_exc()[-2000:])


# ---------------------------------------------------------------- CLI
def _cli():
    parser = argparse.ArgumentParser(description="SPOILOCK 자막 분석 (서버 없이 실행)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="파싱·청크만 확인 (Gemini 호출 없음)")
    p.add_argument("--srt", required=True)

    p = sub.add_parser("run", help="추출 + 색인까지 실행")
    p.add_argument("--srt", required=True)
    p.add_argument("--drama-id", type=int, required=True, help="Spring dramas.id")
    p.add_argument("--episode-id", type=int, required=True, help="Spring episodes.id")
    p.add_argument("--episode", type=int, required=True, help="회차 번호")
    p.add_argument("--kb-version", default=DEFAULT_KB_VERSION)
    p.add_argument("--language", default="en")
    p.add_argument("--replace", action="store_true", help="기존 미검수 사건 교체")
    args = parser.parse_args()

    raw = Path(args.srt).read_bytes()
    if args.command == "check":
        rows, skipped = core.parse_srt_text(core.decode_srt(raw))
        chunks = core.build_chunks(rows, drama_id=0, episode_id=0, episode=0, kb_version="check",
                                   language="-", source_name=Path(args.srt).name)
        print(json.dumps({
            "subtitles": len(rows), "skipped_blocks": skipped, "chunks": len(chunks),
            "batches": len(core.batch_ranges(len(chunks), **core.DEFAULT_EXTRACT_SETTINGS)),
            "first": core.format_ms(rows[0]["start_ms"]), "last": core.format_ms(rows[-1]["end_ms"]),
            "max_chunk_chars": max(len(c["document"]) for c in chunks),
        }, ensure_ascii=False, indent=2))
        return

    # 동기 실행: executor 대신 현재 스레드에서 바로 처리
    global _executor

    class _Inline:
        def submit(self, fn, *a):
            fn(*a)
    _executor = _Inline()
    job = create_job(raw, drama_id=args.drama_id, episode_id=args.episode_id, episode=args.episode,
                     kb_version=args.kb_version, language=args.language,
                     original_name=Path(args.srt).name, replace=args.replace)
    job = get_job(job["job_id"])
    print(json.dumps(job, ensure_ascii=False, indent=2))
    sys.exit(0 if job.get("status") == STATUS_DONE else 1)


if __name__ == "__main__":
    _cli()
