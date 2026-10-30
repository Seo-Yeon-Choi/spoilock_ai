"""
SPOILOCK 영상 분석: 영상 → 대사 텍스트(SRT) → 사건 후보

흐름
  업로드된 영상 → faster-whisper 음성 인식(영상 파일에서 음성을 직접 읽음, ffmpeg 별도 설치 불필요)
  → 대사 + 시간 정보 → SRT 파일 / 세그먼트 JSON 저장
  → episode_id(Spring episodes.id)가 있으면 생성된 SRT 로 자막 분석 작업(ingest_jobs) 시작
     (청크 → Gemini 사건 추출 → Chroma 색인, 자막 업로드와 같은 흐름)
  → episode_id 가 없으면 (선택) VIDEO_POST_COMMAND 실행 (예전 방식)

작업은 한 번에 하나씩 순서대로 처리(음성 인식은 CPU를 많이 씀).
작업 상태는 data/video_jobs/<job_id>/status.json 에 저장되어 서버를 재시작해도 조회 가능.

환경변수 (.env)
  WHISPER_MODEL=small            # tiny / base / small / medium / large-v3 (클수록 정확, 느림)
  WHISPER_DEVICE=cpu             # GPU 가 있으면 cuda
  WHISPER_COMPUTE_TYPE=int8      # cpu: int8 / cuda: float16
  WHISPER_LANGUAGE=              # 비우면 자동 감지 (예: en, ko)
  VIDEO_POST_COMMAND=            # SRT 생성 후 실행할 명령 (아래 run_post_command 참고)
"""
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / "data" / "video_jobs"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "").strip() or None
VIDEO_POST_COMMAND = os.getenv("VIDEO_POST_COMMAND", "").strip()

# 상태 흐름
#   QUEUED → TRANSCRIBING → SRT_READY → INGESTING(사건 추출) → DONE / FAILED
#   (episode_id 없이 VIDEO_POST_COMMAND 사용 시: SRT_READY → POST_PROCESSING → DONE)
STATUS_QUEUED = "QUEUED"
STATUS_TRANSCRIBING = "TRANSCRIBING"
STATUS_SRT_READY = "SRT_READY"
STATUS_INGESTING = "INGESTING"
STATUS_POST_PROCESSING = "POST_PROCESSING"
STATUS_DONE = "DONE"
STATUS_FAILED = "FAILED"
# 이 모듈의 스레드가 처리 중인 상태 (서버 재시작 시 FAILED 로 표시)
ACTIVE_STATUSES = {STATUS_QUEUED, STATUS_TRANSCRIBING, STATUS_POST_PROCESSING}
# INGESTING 은 ingest_jobs 가 처리 (재시작 복구도 ingest_jobs 가 담당)
BUSY_STATUSES = ACTIVE_STATUSES | {STATUS_INGESTING}

# 진행률 배분: 사건 추출까지 이어서 하면 음성 인식 0~50%, 사건 추출 50~100%
TRANSCRIBE_SHARE = 0.5

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-job")
_lock = threading.Lock()
_model = None
_model_lock = threading.Lock()

# 테스트에서 가짜 인식기로 바꿔 끼울 수 있게 함수로 분리
_transcriber_factory = None


# ---------------------------------------------------------------- 음성 인식 모델
def _load_model():
    """faster-whisper 모델을 처음 쓸 때 한 번만 로딩 (서버 시작을 느리게 하지 않음)"""
    global _model
    with _model_lock:
        if _model is None:
            if _transcriber_factory is not None:
                _model = _transcriber_factory()
            else:
                from faster_whisper import WhisperModel
                _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE,
                                      compute_type=WHISPER_COMPUTE_TYPE)
        return _model


# ---------------------------------------------------------------- SRT
def _srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments) -> str:
    """[{start, end, text}] → SRT 문자열 (빈 대사는 제외, 번호는 1부터 연속)"""
    blocks = []
    n = 0
    for seg in segments:
        text = " ".join(str(seg["text"]).split())
        if not text:
            continue
        n += 1
        blocks.append(f"{n}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n{text}\n")
    return "\n".join(blocks)


# ---------------------------------------------------------------- 작업 상태
def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _write_status(job_id: str, /, **fields):
    """status.json 갱신 (임시 파일 → 교체로 원자적 저장)"""
    with _lock:
        path = _status_path(job_id)
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        data.update(fields)
        data["updated_at"] = datetime.now().isoformat(timespec="seconds")
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return data


def _read_job(job_id: str) -> dict | None:
    if not job_id or any(c in job_id for c in "/\\.") or len(job_id) > 64:
        return None
    path = _status_path(job_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def get_job(job_id: str) -> dict | None:
    """작업 상태. 사건 추출 단계(INGESTING)면 자막 분석 작업 상태를 합쳐서 돌려줌"""
    job = _read_job(job_id)
    if job is None or job.get("status") != STATUS_INGESTING:
        return job
    return _merge_ingest(job)


def _merge_ingest(job: dict) -> dict:
    """INGESTING 영상 작업 + ingest_jobs 작업 → 하나의 상태. 끝났으면 영상 작업에도 기록"""
    import ingest_jobs

    ingest = ingest_jobs.get_job(job.get("ingest_job_id") or "")
    if ingest is None:
        return _write_status(job["job_id"], status=STATUS_FAILED,
                             error="사건 추출 작업 기록을 찾을 수 없습니다. 재시도해 주세요.")

    if ingest.get("status") == ingest_jobs.STATUS_DONE:
        return _write_status(
            job["job_id"], status=STATUS_DONE, progress=1.0, ingest_status=ingest["status"],
            event_count=ingest.get("event_count"), kept_events=ingest.get("kept_events"),
            replaced_events=ingest.get("replaced_events"),
            candidate_count=ingest.get("candidate_count"), error=None)

    if ingest.get("status") == ingest_jobs.STATUS_FAILED:
        return _write_status(
            job["job_id"], status=STATUS_FAILED, ingest_status=ingest["status"],
            error=f"사건 추출 실패: {ingest.get('error') or '알 수 없는 오류'}"[:500])

    # 진행 중: 저장하지 않고 계산만 (ingest 진행률이 계속 바뀜)
    merged = dict(job)
    merged["ingest_status"] = ingest.get("status")
    merged["progress"] = round(TRANSCRIBE_SHARE + (1 - TRANSCRIBE_SHARE)
                               * float(ingest.get("progress") or 0.0), 3)
    for key in ("done_batches", "total_batches", "candidate_count"):
        if key in ingest:
            merged[key] = ingest[key]
    return merged


def list_jobs(limit: int = 50) -> list[dict]:
    if not JOBS_DIR.is_dir():
        return []
    jobs = []
    for d in JOBS_DIR.iterdir():
        st = d / "status.json"
        if st.is_file():
            try:
                jobs.append(json.loads(st.read_text(encoding="utf-8")))
            except Exception:
                continue
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return [get_job(j["job_id"]) if j.get("status") == STATUS_INGESTING else j
            for j in jobs[:limit]]


def new_job_dir(original_name: str) -> tuple[str, Path]:
    """업로드 저장용 작업 폴더 생성. 반환: (job_id, 영상 저장 경로)"""
    job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_name or "").suffix.lower()
    if suffix not in {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mp3", ".wav", ".m4a", ".aac", ".flac"}:
        suffix = ".bin"
    return job_id, d / f"source{suffix}"


def create_job(job_id: str, video_path: Path, *, drama_id: int, episode: int,
               original_name: str, size_bytes: int, language: str | None,
               episode_id: int | None = None, replace: bool = False,
               kb_version: str | None = None) -> dict:
    """
    episode_id(Spring episodes.id)를 주면 음성 인식 후 사건 추출까지 이어서 진행.
    replace=True 면 이미 사건이 있는 회차도 교체 (검수·수동 사건은 보존)
    """
    now = datetime.now().isoformat(timespec="seconds")
    _write_status(
        job_id,
        job_id=job_id, status=STATUS_QUEUED, progress=0.0,
        drama_id=drama_id, episode=episode, episode_id=episode_id, replace=bool(replace),
        kb_version=kb_version,   # 사건을 넣을 칸 (환경별 분리). None 이면 ingest_jobs 기본값
        original_name=original_name, size_bytes=size_bytes,
        video_path=str(video_path), language=language or WHISPER_LANGUAGE,
        whisper_model=WHISPER_MODEL, created_at=now, error=None,
    )
    _executor.submit(_run_job, job_id)
    return get_job(job_id)


def recover_interrupted_jobs():
    """서버가 작업 도중 꺼졌던 경우: 진행 중 상태를 FAILED 로 표시 (다시 업로드하거나 재시도)"""
    if not JOBS_DIR.is_dir():
        return
    for d in JOBS_DIR.iterdir():
        job = _read_job(d.name)
        if job is None:
            continue
        status = job.get("status")
        # SRT 는 만들었는데 사건 추출을 시작하기 전에 꺼진 경우도 포함
        stuck_before_ingest = (status == STATUS_SRT_READY and job.get("episode_id")
                               and not job.get("ingest_job_id"))
        if status in ACTIVE_STATUSES or stuck_before_ingest:
            _write_status(job["job_id"], status=STATUS_FAILED,
                          error="서버 재시작으로 작업이 중단되었습니다. 재시도해 주세요.")


def retry_job(job_id: str) -> dict | None:
    """
    실패한 단계부터 다시 진행:
      - 사건 추출이 실패했으면 그 작업을 재시도 (끝난 Gemini 배치는 건너뜀)
      - SRT 는 있는데 사건 추출을 시작하지 못했으면 SRT 로 바로 시작 (음성 인식 생략)
      - 그 외에는 음성 인식부터 다시
    같은 회차 작업이 진행 중이면 ingest_jobs.JobConflict 발생
    """
    job = get_job(job_id)
    if job is None or job.get("status") in BUSY_STATUSES:
        return job
    if job.get("status") == STATUS_DONE:
        return job

    srt_ready = bool(job.get("srt_path")) and Path(job["srt_path"]).is_file()

    if job.get("episode_id") and srt_ready and job.get("ingest_job_id"):
        import ingest_jobs
        ingest = ingest_jobs.retry_job(job["ingest_job_id"])
        if ingest is not None and ingest.get("status") in ingest_jobs.ACTIVE_STATUSES:
            _write_status(job_id, status=STATUS_INGESTING, error=None)
            return get_job(job_id)
        # 사건 추출 작업 기록이 없거나 재시도할 수 없음 → SRT 로 새로 시작

    if job.get("episode_id") and srt_ready:
        _start_ingest(job_id)
        return get_job(job_id)

    _write_status(job_id, status=STATUS_QUEUED, progress=0.0, error=None)
    _executor.submit(_run_job, job_id)
    return get_job(job_id)


# ---------------------------------------------------------------- 작업 실행
def _run_job(job_id: str):
    job = get_job(job_id)
    try:
        started = time.perf_counter()
        _write_status(job_id, status=STATUS_TRANSCRIBING, progress=0.0)
        model = _load_model()

        segments_iter, info = model.transcribe(
            job["video_path"],
            language=job.get("language") or None,
            vad_filter=True,            # 무음·배경음 구간 건너뜀 → 속도·정확도 향상
            beam_size=5,
        )
        duration = float(getattr(info, "duration", 0) or 0)
        segments = []
        last_report = 0.0
        for seg in segments_iter:      # 제너레이터: 인식이 진행되면서 하나씩 나옴
            text = " ".join(seg.text.split())
            if text:                   # 빈 대사는 버림 (SRT 줄 수와 일치)
                segments.append({"start": float(seg.start), "end": float(seg.end), "text": text})
            if duration > 0:
                progress = min(0.99, float(seg.end) / duration)
                if job.get("episode_id"):
                    progress *= TRANSCRIBE_SHARE
                if progress - last_report >= 0.02:
                    last_report = progress
                    _write_status(job_id, progress=round(progress, 3))

        d = _job_dir(job_id)
        srt_path = d / f"ep{int(job['episode']):02d}.srt"
        srt_path.write_text(segments_to_srt(segments), encoding="utf-8")
        (d / "segments.json").write_text(
            json.dumps({"language": getattr(info, "language", None),
                        "duration": duration, "segments": segments},
                       ensure_ascii=False, indent=2), encoding="utf-8")

        _write_status(
            job_id, status=STATUS_SRT_READY,
            progress=TRANSCRIBE_SHARE if job.get("episode_id") else 1.0,
            srt_path=str(srt_path), segment_count=len(segments),
            detected_language=getattr(info, "language", None),
            duration_sec=round(duration, 1),
            transcribe_sec=round(time.perf_counter() - started, 1),
        )

        if job.get("episode_id"):
            _start_ingest(job_id)
        elif VIDEO_POST_COMMAND:
            run_post_command(job_id)
    except Exception as e:
        _write_status(job_id, status=STATUS_FAILED,
                      error=f"{type(e).__name__}: {e}"[:500],
                      traceback=traceback.format_exc()[-2000:])


def _start_ingest(job_id: str):
    """생성된 SRT 로 자막 분석 작업 시작 (Gemini 추출은 ingest_jobs 전용 스레드에서 실행)"""
    import ingest_core
    import ingest_jobs

    job = _read_job(job_id)
    if not job.get("segment_count"):
        _write_status(job_id, status=STATUS_FAILED,
                      error="영상에서 인식된 대사가 없습니다. 음성이 있는 영상인지 확인해 주세요.")
        return

    raw = Path(job["srt_path"]).read_bytes()
    try:
        ingest = ingest_jobs.create_job(
            raw,
            drama_id=int(job["drama_id"]),
            episode_id=int(job["episode_id"]),
            episode=int(job["episode"]),
            kb_version=job.get("kb_version") or ingest_jobs.DEFAULT_KB_VERSION,
            language=job.get("language") or job.get("detected_language") or "en",
            original_name=f"video-{job_id}.srt",
            replace=bool(job.get("replace")),
            source_mode="VIDEO_STT",   # 음성 인식 자막으로 만든 사건임을 표시
        )
    except (ingest_core.IngestError, ingest_jobs.JobConflict) as e:
        _write_status(job_id, status=STATUS_FAILED, error=f"사건 추출을 시작할 수 없습니다: {e}"[:500])
        return

    _write_status(job_id, status=STATUS_INGESTING, ingest_job_id=ingest["job_id"],
                  progress=TRANSCRIBE_SHARE, error=None)


def run_post_command(job_id: str):
    """
    SRT 생성 후 기존 파이프라인(전처리·사건 추출)을 이어서 실행.
    VIDEO_POST_COMMAND 예:
      {python} -m tools.events extract --episode {episode} --srt "{srt}"
    사용 가능한 값: {python} {srt} {episode} {drama_id} {job_dir}
    """
    job = get_job(job_id)
    cmd_text = VIDEO_POST_COMMAND.format(
        python=sys.executable, srt=job["srt_path"], episode=job["episode"],
        drama_id=job["drama_id"], job_dir=str(_job_dir(job_id)))
    _write_status(job_id, status=STATUS_POST_PROCESSING, post_command=cmd_text)
    started = time.perf_counter()
    # Windows 는 명령 문자열을 그대로 넘겨야 따옴표가 올바르게 해석됨
    args = cmd_text if os.name == "nt" else shlex.split(cmd_text)
    proc = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    (_job_dir(job_id) / "post_command.log").write_text(log, encoding="utf-8")
    if proc.returncode != 0:
        _write_status(job_id, status=STATUS_FAILED, post_exit_code=proc.returncode,
                      error=f"후처리 명령 실패 (exit {proc.returncode}). post_command.log 확인",
                      post_sec=round(time.perf_counter() - started, 1))
    else:
        _write_status(job_id, status=STATUS_DONE, post_exit_code=0,
                      post_sec=round(time.perf_counter() - started, 1))
