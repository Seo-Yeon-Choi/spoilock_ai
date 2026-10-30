"""
SPOILOCK 사건 관리 로직 (Chroma 사건 컬렉션 = 사건의 원본 저장소)

  list_events     회차별 사건 목록 (관리자 화면)
  create_manual   수동 사건 등록 (MANUAL, PENDING)
  update_event    설명/분류 수정, 승인(VERIFIED)·반려(REJECTED)·대기(PENDING), 최초 공개 회차 지정
  delete_event    삭제
  kb_episodes     작품별 판정에 쓰이는 회차 목록 (Spring 작품 매핑 대체용)

기존 kb1 데이터(origin·first_reveal_episode 없음)도 그대로 읽고 수정할 수 있게 기본값으로 처리.
"""
import json
import uuid

import ingest_core as core

STATUSES = ("PENDING", "VERIFIED", "REJECTED")
EDITABLE = {"description", "event_type", "fact_status", "verification_status", "first_reveal_episode"}
MAX_DESCRIPTION = 500


class EventNotFound(LookupError):
    pass


class EventConflict(Exception):
    pass


def _check_no_active_job(meta):
    """같은 회차 자막 분석이 진행 중이면 수정 금지 (색인 단계에서 교체되며 수정이 사라질 수 있음)"""
    import ingest_jobs
    active = ingest_jobs._active_job_for(meta["drama_id"], meta["source_episode"], meta["kb_version"])
    if active:
        raise EventConflict(f"이 회차의 자막 분석 작업이 진행 중입니다: {active['job_id']}")


def to_public(event_id, document, meta) -> dict:
    """Chroma 레코드 → 관리자 API 응답 (기존 데이터 기본값 보정)"""
    def parse(value):
        try:
            return json.loads(value) if value else []
        except (TypeError, ValueError):
            return []
    first = int(meta.get("first_reveal_episode") or 0)
    return {
        "event_id": event_id,
        "description": document,
        "drama_id": meta.get("drama_id"),
        "episode_id": meta.get("episode_id"),
        "source_episode": meta.get("source_episode"),
        "kb_version": meta.get("kb_version"),
        "event_type": meta.get("event_type", "MAJOR_EVENT"),
        "fact_status": meta.get("fact_status", "CHARACTER_CLAIM"),
        "verification_status": meta.get("verification_status", "PENDING"),
        "first_reveal_episode": first or None,
        "origin": meta.get("origin", "LEGACY"),
        "evidence": parse(meta.get("evidence_json")),
        "source_chunk_ids": parse(meta.get("source_chunk_ids_json")),
        # 화면 근거 [{timestamp: "MM:SS", description}] (자막 + 영상 분석에서만)
        "visual_evidence": parse(meta.get("visual_evidence_json")),
        # DIALOGUE / VISUAL / BOTH
        "evidence_type": meta.get("evidence_type", "DIALOGUE" if meta.get("origin") != "MANUAL" else None),
        # SUBTITLE / SUBTITLE_VIDEO / VIDEO_STT (수동 등록·기존 데이터는 None)
        "source_mode": meta.get("source_mode"),
    }


def _where(drama_id, kb_version, episode=None, status=None):
    conds = [{"drama_id": int(drama_id)}, {"kb_version": kb_version}]
    if episode is not None:
        conds.append({"source_episode": int(episode)})
    if status is not None:
        conds.append({"verification_status": status})
    return {"$and": conds}


def list_events(col, *, drama_id, kb_version, episode=None, status=None) -> list[dict]:
    data = col.get(where=_where(drama_id, kb_version, episode, status), include=["documents", "metadatas"])
    events = [to_public(i, d, m) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])]
    events.sort(key=lambda e: (e["source_episode"] or 0, e["event_id"]))
    return events


def get_event(col, event_id) -> tuple[str, dict]:
    data = col.get(ids=[event_id], include=["documents", "metadatas"])
    if not data["ids"]:
        raise EventNotFound(event_id)
    return data["documents"][0], dict(data["metadatas"][0])


def _clean_description(text) -> str:
    if not isinstance(text, str) or not text.strip():
        raise core.IngestError("사건 설명이 비어 있습니다.")
    text = core.compact(text)
    if len(text) > MAX_DESCRIPTION:
        raise core.IngestError(f"사건 설명은 {MAX_DESCRIPTION}자 이하로 입력해 주세요.")
    return text


def _validate_first_reveal(value, source_episode):
    """최초 공개 회차: 1 이상, 사건이 나온 회차(source_episode) 이하. None/0 = 미지정"""
    if value in (None, 0):
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise core.IngestError("최초 공개 회차는 1 이상의 정수여야 합니다.")
    if value > int(source_episode):
        raise core.IngestError(
            f"최초 공개 회차({value})는 사건이 나온 회차({source_episode}) 이후일 수 없습니다.")
    return value


def _encode(model, text):
    return model.encode(["passage: " + text], normalize_embeddings=True,
                        show_progress_bar=False).tolist()


def create_manual(col, model, *, drama_id, episode_id, episode, kb_version, description,
                  event_type="MAJOR_EVENT", fact_status="DIRECT_EVENT") -> dict:
    description = _clean_description(description)
    if event_type not in core.TYPES or fact_status not in core.FACT_STATUSES:
        raise core.IngestError("사건 유형 또는 사실 구분 값이 올바르지 않습니다.")
    meta = {
        "drama_id": int(drama_id), "episode_id": int(episode_id), "source_episode": int(episode),
        "kb_version": kb_version, "event_type": event_type, "fact_status": fact_status,
        "verification_status": "PENDING", "first_reveal_episode": 0, "origin": "MANUAL",
        "run_tag": "manual", "evidence_json": "[]", "source_chunk_ids_json": "[]",
    }
    _check_no_active_job(meta)
    event_id = f"d{drama_id}-e{episode_id}-{kb_version}-m{uuid.uuid4().hex[:8]}"
    col.upsert(ids=[event_id], documents=[description], metadatas=[meta],
               embeddings=_encode(model, description))
    return to_public(event_id, description, meta)


def update_event(col, model, event_id, changes: dict) -> dict:
    unknown = set(changes) - EDITABLE
    if unknown:
        raise core.IngestError(f"수정할 수 없는 항목입니다: {sorted(unknown)}")
    document, meta = get_event(col, event_id)
    _check_no_active_job(meta)

    new_doc = document
    if "description" in changes:
        new_doc = _clean_description(changes["description"])
    if "event_type" in changes:
        if changes["event_type"] not in core.TYPES:
            raise core.IngestError("사건 유형 값이 올바르지 않습니다.")
        meta["event_type"] = changes["event_type"]
    if "fact_status" in changes:
        if changes["fact_status"] not in core.FACT_STATUSES:
            raise core.IngestError("사실 구분 값이 올바르지 않습니다.")
        meta["fact_status"] = changes["fact_status"]
    if "first_reveal_episode" in changes:
        meta["first_reveal_episode"] = _validate_first_reveal(changes["first_reveal_episode"],
                                                              meta["source_episode"])
    if "verification_status" in changes:
        if changes["verification_status"] not in STATUSES:
            raise core.IngestError(f"검수 상태는 {STATUSES} 중 하나여야 합니다.")
        meta["verification_status"] = changes["verification_status"]

    meta.setdefault("first_reveal_episode", 0)
    meta.setdefault("origin", "LEGACY")
    if meta["verification_status"] == "VERIFIED" and int(meta["first_reveal_episode"] or 0) < 1:
        raise core.IngestError("승인하려면 최초 공개 회차를 함께 지정해 주세요.")

    if new_doc != document:
        # 설명이 바뀌면 검색 벡터도 다시 계산
        col.update(ids=[event_id], documents=[new_doc], metadatas=[meta],
                   embeddings=_encode(model, new_doc))
    else:
        col.update(ids=[event_id], metadatas=[meta])
    return to_public(event_id, new_doc, meta)


def delete_event(col, event_id):
    _, meta = get_event(col, event_id)
    _check_no_active_job(meta)
    col.delete(ids=[event_id])


def kb_episodes(col, *, drama_id, kb_version) -> list[dict]:
    """판정에 쓰이는(REJECTED 제외) 사건이 있는 회차와 상태별 개수"""
    data = col.get(where=_where(drama_id, kb_version), include=["metadatas"])
    summary = {}
    for meta in data["metadatas"]:
        ep = int(meta["source_episode"])
        row = summary.setdefault(ep, {"episode": ep, "PENDING": 0, "VERIFIED": 0, "REJECTED": 0})
        row[meta.get("verification_status", "PENDING")] = row.get(meta.get("verification_status", "PENDING"), 0) + 1
    rows = sorted(summary.values(), key=lambda r: r["episode"])
    for r in rows:
        r["active"] = r["PENDING"] + r["VERIFIED"] > 0
    return rows
