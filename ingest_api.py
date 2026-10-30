"""
SPOILOCK 관리자 자막 분석 API (FastAPI 라우터)

  POST /admin/subtitles/jobs                 SRT 업로드 + 분석 시작
       multipart: file, drama_id, episode_id, episode, kb_version?, language?, replace?,
                  video?(자막과 시간이 맞는 회차 영상 → 자막 + 영상 화면으로 분석), characters?(쉼표 구분)
  GET  /admin/subtitles/jobs?drama_id=&episode=   작업 목록
  GET  /admin/subtitles/jobs/{job_id}        작업 상태·진행률
  POST /admin/subtitles/jobs/{job_id}/retry  실패 작업 재시도 (끝난 배치는 건너뜀)

drama_id / episode_id 는 Spring 의 dramas.id / episodes.id, episode 는 회차 번호.
보안: video_api 와 같은 ADMIN_TOKEN → 요청 헤더 X-Admin-Token (비우면 검사 안 함, 운영 시 반드시 설정)
"""
import os
import re
import uuid

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

import ingest_core as core
import ingest_jobs as jobs

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()
SRT_MAX_MB = int(os.getenv("SRT_MAX_MB", "5"))
VIDEO_MAX_MB = int(os.getenv("VIDEO_MAX_MB", "4096"))
VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
CHUNK = 8 * 1024 * 1024
_KB_VERSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,20}$")
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")

router = APIRouter(tags=["admin-subtitle"])


def _check_token(token: str | None):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="관리자 토큰이 올바르지 않습니다.")


def _public(job: dict | None) -> dict:
    if job is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return {k: v for k, v in job.items() if k != "traceback"}


@router.post("/admin/subtitles/jobs", status_code=202)
def create_subtitle_job(
    file: UploadFile = File(..., description="SRT 자막 (UTF-8 / UTF-16 / CP949)"),
    drama_id: int = Form(..., ge=1, description="Spring dramas.id"),
    episode_id: int = Form(..., ge=1, description="Spring episodes.id"),
    episode: int = Form(..., ge=1, description="회차 번호"),
    kb_version: str = Form(jobs.DEFAULT_KB_VERSION),
    language: str = Form("en", description="자막 언어 (en, ko ...)"),
    replace: bool = Form(False, description="기존 미검수 사건 교체 (검수·수동 사건은 보존)"),
    video: UploadFile | None = File(None, description="자막과 시간이 맞는 회차 영상 (선택)"),
    characters: str = Form("", description="주요 인물 이름, 쉼표 구분 (선택)"),
    x_admin_token: str | None = Header(None),
):
    _check_token(x_admin_token)
    name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not name.lower().endswith(".srt"):
        raise HTTPException(status_code=400, detail="SRT 파일만 업로드할 수 있습니다.")
    if not _KB_VERSION_RE.fullmatch(kb_version):
        raise HTTPException(status_code=400, detail="kb_version 형식이 올바르지 않습니다.")
    if not _LANGUAGE_RE.fullmatch(language):
        raise HTTPException(status_code=400, detail="language 형식이 올바르지 않습니다. (예: en, ko)")

    limit = SRT_MAX_MB * 1024 * 1024
    try:
        raw = file.file.read(limit + 1)
    finally:
        file.file.close()
    if len(raw) > limit:
        raise HTTPException(status_code=413, detail=f"자막 파일이 너무 큽니다 (최대 {SRT_MAX_MB}MB)")

    video_tmp = _save_video(video) if video is not None and video.filename else None
    names = [c.strip()[:50] for c in characters.split(",") if c.strip()]
    try:
        job = jobs.create_job(raw, drama_id=drama_id, episode_id=episode_id, episode=episode,
                              kb_version=kb_version, language=language, original_name=name,
                              replace=replace, video_path=video_tmp, characters=names)
    except core.IngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except jobs.JobConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    finally:
        # 작업 폴더로 옮겨졌으면 이미 없음. 실패했으면 임시 파일 정리
        if video_tmp is not None and video_tmp.exists():
            video_tmp.unlink(missing_ok=True)
    return _public(job)


def _save_video(upload: UploadFile):
    """큰 영상을 메모리에 올리지 않고 임시 파일로 저장 (작업 생성 시 작업 폴더로 이동)"""
    suffix = "." + (upload.filename or "").rsplit(".", 1)[-1].lower() if "." in (upload.filename or "") else ""
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="영상은 mp4, mkv, mov, webm 파일만 가능합니다.")
    incoming = jobs.JOBS_DIR / "_incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    path = incoming / f"{uuid.uuid4().hex}{suffix}"
    limit, size = VIDEO_MAX_MB * 1024 * 1024, 0
    try:
        with path.open("wb") as out:
            while True:
                chunk = upload.file.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"영상이 너무 큽니다 (최대 {VIDEO_MAX_MB}MB)")
                out.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        upload.file.close()
    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="빈 영상 파일입니다.")
    return path


@router.get("/admin/subtitles/jobs")
def list_subtitle_jobs(drama_id: int | None = None, episode: int | None = None, limit: int = 50,
                       x_admin_token: str | None = Header(None)):
    _check_token(x_admin_token)
    return [_public(j) for j in jobs.list_jobs(limit=min(max(limit, 1), 200),
                                                drama_id=drama_id, episode=episode)]


@router.get("/admin/subtitles/jobs/{job_id}")
def get_subtitle_job(job_id: str, x_admin_token: str | None = Header(None)):
    _check_token(x_admin_token)
    return _public(jobs.get_job(job_id))


@router.post("/admin/subtitles/jobs/{job_id}/retry", status_code=202)
def retry_subtitle_job(job_id: str, x_admin_token: str | None = Header(None)):
    _check_token(x_admin_token)
    try:
        job = jobs.retry_job(job_id)
    except jobs.JobConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    if job is not None and job.get("status") not in (jobs.STATUS_FAILED, *jobs.ACTIVE_STATUSES):
        raise HTTPException(status_code=409, detail=f"재시도할 수 없는 상태입니다: {job.get('status')}")
    return _public(job)
