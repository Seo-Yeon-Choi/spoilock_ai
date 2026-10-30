"""
SPOILOCK 관리자 영상 분석 API (FastAPI 라우터)

  POST /admin/videos                영상 업로드 + 분석 시작
                                    (multipart: file, drama_id, episode, language, episode_id?, replace?)
                                    episode_id(Spring episodes.id)를 주면 음성 인식 후 사건 추출까지 진행
  GET  /admin/videos                작업 목록
  GET  /admin/videos/{job_id}       작업 상태·진행률
  GET  /admin/videos/{job_id}/srt   생성된 SRT 다운로드
  POST /admin/videos/{job_id}/retry 실패한 작업 재시도
  GET  /admin/videos-ui             업로드 테스트 페이지

보안: .env 에 ADMIN_TOKEN 을 넣으면 요청 헤더 X-Admin-Token 이 일치해야 함 (비우면 검사 안 함)
필요 패키지: pip install python-multipart faster-whisper
"""
import os
import re
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

import video_pipeline as vp

ROOT = Path(__file__).resolve().parent
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()
VIDEO_MAX_MB = int(os.getenv("VIDEO_MAX_MB", "4096"))
CHUNK = 8 * 1024 * 1024

router = APIRouter(tags=["admin-video"])


def _check_token(token: str | None):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="관리자 토큰이 올바르지 않습니다.")


def _public(job: dict) -> dict:
    """응답에서 서버 내부 경로·긴 오류 추적은 숨김"""
    hidden = {"video_path", "traceback"}
    out = {k: v for k, v in job.items() if k not in hidden}
    if "srt_path" in out:
        out["srt_file"] = Path(out.pop("srt_path")).name
    return out


@router.post("/admin/videos", status_code=202)
def upload_video(
    file: UploadFile = File(..., description="회차 영상 (mp4, mkv, mov 등) 또는 음성 파일"),
    drama_id: int = Form(..., ge=1, description="Python 사건 데이터의 drama_id"),
    episode: int = Form(..., ge=1, description="회차 번호"),
    language: str | None = Form(None, description="음성 언어 (예: en, ko). 비우면 자동 감지"),
    episode_id: int | None = Form(None, ge=1, description="Spring episodes.id. 주면 사건 추출까지 진행"),
    replace: bool = Form(False, description="이미 사건이 있는 회차 교체 (검수·수동 사건은 보존)"),
    kb_version: str | None = Form(None, description="사건을 넣을 칸 (환경별 분리). 비우면 서버 기본값"),
    x_admin_token: str | None = Header(None),
):
    _check_token(x_admin_token)
    language = (language or "").strip() or None
    if language is not None and not re.fullmatch(r"[a-z]{2,3}", language):
        raise HTTPException(status_code=400, detail="language 형식이 올바르지 않습니다. (예: en, ko)")

    kb_version = (kb_version or "").strip() or None
    if kb_version is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,20}", kb_version):
        raise HTTPException(status_code=400, detail="kb_version 형식이 올바르지 않습니다.")

    # 같은 회차 사건 추출이 진행 중이면 업로드 전에 거절 (긴 음성 인식 후에 실패하지 않도록)
    if episode_id is not None:
        import ingest_jobs
        active = ingest_jobs._active_job_for(drama_id, episode,
                                             kb_version or ingest_jobs.DEFAULT_KB_VERSION)
        if active or _active_video_job_for(drama_id, episode):
            raise HTTPException(status_code=409, detail="같은 회차의 분석 작업이 진행 중입니다.")

    job_id, video_path = vp.new_job_dir(file.filename)

    # 큰 파일을 메모리에 올리지 않고 조금씩 디스크에 저장
    size = 0
    limit = VIDEO_MAX_MB * 1024 * 1024
    try:
        with video_path.open("wb") as out:
            while True:
                chunk = file.file.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413,
                                        detail=f"파일이 너무 큽니다 (최대 {VIDEO_MAX_MB}MB)")
                out.write(chunk)
    except Exception:
        shutil.rmtree(video_path.parent, ignore_errors=True)
        raise
    finally:
        file.file.close()

    if size == 0:
        shutil.rmtree(video_path.parent, ignore_errors=True)
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    job = vp.create_job(job_id, video_path, drama_id=drama_id, episode=episode,
                        original_name=file.filename or "", size_bytes=size,
                        language=language, episode_id=episode_id, replace=replace,
                        kb_version=kb_version)
    return _public(job)


def _active_video_job_for(drama_id: int, episode: int) -> bool:
    return any(j.get("drama_id") == drama_id and j.get("episode") == episode
               and j.get("status") in vp.BUSY_STATUSES
               for j in vp.list_jobs(limit=10_000))


@router.get("/admin/videos")
def list_videos(limit: int = 50, x_admin_token: str | None = Header(None)):
    _check_token(x_admin_token)
    return [_public(j) for j in vp.list_jobs(limit=min(max(limit, 1), 200))]


@router.get("/admin/videos/{job_id}")
def get_video_job(job_id: str, x_admin_token: str | None = Header(None)):
    _check_token(x_admin_token)
    job = vp.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return _public(job)


@router.get("/admin/videos/{job_id}/srt")
def download_srt(job_id: str, x_admin_token: str | None = Header(None)):
    _check_token(x_admin_token)
    job = vp.get_job(job_id)
    if job is None or not job.get("srt_path") or not Path(job["srt_path"]).is_file():
        raise HTTPException(status_code=404, detail="SRT 가 아직 없습니다.")
    return FileResponse(job["srt_path"], media_type="application/x-subrip",
                        filename=Path(job["srt_path"]).name)


@router.post("/admin/videos/{job_id}/retry", status_code=202)
def retry_video_job(job_id: str, x_admin_token: str | None = Header(None)):
    _check_token(x_admin_token)
    import ingest_jobs
    try:
        job = vp.retry_job(job_id)
    except ingest_jobs.JobConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    if job is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return _public(job)


@router.get("/admin/videos-ui")
def video_upload_page():
    return FileResponse(ROOT / "web" / "video_upload.html")
