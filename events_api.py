"""
SPOILOCK 관리자 사건 API (FastAPI 라우터)

  GET    /admin/events?drama_id=&episode=&status=&kb_version=   회차별 사건 목록
  POST   /admin/events                                         수동 사건 등록
  PATCH  /admin/events/{event_id}                              수정·승인·반려·최초 공개 회차 지정
  DELETE /admin/events/{event_id}                              삭제
  GET    /kb/episodes?drama_id=&kb_version=                    판정에 쓰이는 회차 목록 (Spring 매핑용)

승인 예시 (PATCH body):  {"verification_status": "VERIFIED", "first_reveal_episode": 3}
반려 예시:               {"verification_status": "REJECTED"}   → 판정 후보에서 제외
보안: ADMIN_TOKEN → 요청 헤더 X-Admin-Token (video_api / ingest_api 와 동일)
"""
import os

from fastapi import APIRouter, Header, HTTPException, Response
from pydantic import BaseModel, Field

import events_admin as admin
import ingest_core as core
import ingest_jobs

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "").strip()

router = APIRouter(tags=["admin-events"])


def _check_token(token: str | None):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="관리자 토큰이 올바르지 않습니다.")


def _res() -> dict:
    return ingest_jobs._resources()


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except admin.EventNotFound:
        raise HTTPException(status_code=404, detail="사건을 찾을 수 없습니다.")
    except admin.EventConflict as e:
        raise HTTPException(status_code=409, detail=str(e))
    except core.IngestError as e:
        raise HTTPException(status_code=400, detail=str(e))


class CreateEventRequest(BaseModel):
    drama_id: int = Field(..., ge=1, description="Spring dramas.id")
    episode_id: int = Field(..., ge=1, description="Spring episodes.id")
    episode: int = Field(..., ge=1, description="사건이 나온 회차 번호")
    description: str = Field(..., min_length=1, max_length=admin.MAX_DESCRIPTION)
    event_type: str = "MAJOR_EVENT"
    fact_status: str = "DIRECT_EVENT"
    kb_version: str = ingest_jobs.DEFAULT_KB_VERSION


class UpdateEventRequest(BaseModel):
    description: str | None = Field(None, min_length=1, max_length=admin.MAX_DESCRIPTION)
    event_type: str | None = None
    fact_status: str | None = None
    verification_status: str | None = None
    first_reveal_episode: int | None = Field(None, ge=0, description="null 또는 0 = 미지정")


@router.get("/admin/events")
def list_events(drama_id: int, episode: int | None = None, status: str | None = None,
                kb_version: str = ingest_jobs.DEFAULT_KB_VERSION,
                x_admin_token: str | None = Header(None)):
    _check_token(x_admin_token)
    if status is not None and status not in admin.STATUSES:
        raise HTTPException(status_code=400, detail=f"status 는 {admin.STATUSES} 중 하나여야 합니다.")
    return admin.list_events(_res()["col"], drama_id=drama_id, kb_version=kb_version,
                             episode=episode, status=status)


@router.post("/admin/events", status_code=201)
def create_event(req: CreateEventRequest, x_admin_token: str | None = Header(None)):
    _check_token(x_admin_token)
    res = _res()
    return _call(admin.create_manual, res["col"], res["model"], drama_id=req.drama_id,
                 episode_id=req.episode_id, episode=req.episode, kb_version=req.kb_version,
                 description=req.description, event_type=req.event_type, fact_status=req.fact_status)


@router.patch("/admin/events/{event_id}")
def update_event(event_id: str, req: UpdateEventRequest, x_admin_token: str | None = Header(None)):
    _check_token(x_admin_token)
    changes = req.model_dump(exclude_unset=True)   # 보내지 않은 항목은 그대로 둠
    if not changes:
        raise HTTPException(status_code=400, detail="수정할 항목이 없습니다.")
    for key in ("description", "event_type", "fact_status", "verification_status"):
        if key in changes and changes[key] is None:
            raise HTTPException(status_code=400, detail=f"{key} 는 null 로 지정할 수 없습니다.")
    res = _res()
    return _call(admin.update_event, res["col"], res["model"], event_id, changes)


@router.delete("/admin/events/{event_id}", status_code=204)
def delete_event(event_id: str, x_admin_token: str | None = Header(None)):
    _check_token(x_admin_token)
    _call(admin.delete_event, _res()["col"], event_id)
    return Response(status_code=204)


@router.get("/kb/episodes")
def kb_episodes(drama_id: int, kb_version: str = ingest_jobs.DEFAULT_KB_VERSION,
                x_admin_token: str | None = Header(None)):
    _check_token(x_admin_token)
    rows = admin.kb_episodes(_res()["col"], drama_id=drama_id, kb_version=kb_version)
    return {"drama_id": drama_id, "kb_version": kb_version,
            "episodes": [r["episode"] for r in rows if r["active"]], "detail": rows}
