"""
SPOILOCK 사건 DB(Chroma) 칸(kb_version) 관리 도구

사건마다 kb_version 이름표가 붙어 있고, Spring 은 환경마다 다른 칸을 씀
  AWS 배포본: kb1 (기본값)
  각자 로컬:  .env 의 SPOILER_KB_VERSION (예: local)

명령 (모두 Python 서버를 끈 상태에서 실행. 기본은 미리보기, --apply 를 붙여야 실제로 바뀜)
  list                                   칸·작품·회차별 사건 수
  copy   --from kb1 --to local           사건을 다른 칸으로 복사 (원본 유지)
  move   --from kb1 --to local           옮기기 (복사 후 원본 삭제)
  delete --kb local                      삭제

  --drama 5 7    특정 작품만 (여러 개 가능)
  --episode 1    특정 회차만 (--drama 와 함께)

예
  python kb_tool.py list
  python kb_tool.py copy --from kb1 --to local                    # 미리보기
  python kb_tool.py copy --from kb1 --to local --apply            # 실제 복사
  python kb_tool.py delete --kb kb1 --drama 4 5 --apply           # kb1 에서 테스트 작품만 삭제

실행 전에 chroma_data 폴더를 백업해 두세요.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_KB_RE_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
BATCH = 200


def _where(kb, dramas=None, episode=None):
    conds = [{"kb_version": kb}]
    if dramas:
        conds.append({"drama_id": {"$in": [int(d) for d in dramas]}} if len(dramas) > 1
                     else {"drama_id": int(dramas[0])})
    if episode is not None:
        conds.append({"source_episode": int(episode)})
    return conds[0] if len(conds) == 1 else {"$and": conds}


def new_id(old_id: str, src: str, dst: str) -> str:
    """사건 ID 는 칸 이름을 포함 (d5-e15-kb1-xxxx-f0001) → 칸 이름만 바꿔 겹치지 않게"""
    marker = f"-{src}-"
    if marker in old_id:
        return old_id.replace(marker, f"-{dst}-", 1)
    return f"{dst}:{old_id}"


def summarize(col):
    data = col.get(include=["metadatas"])
    by_kb = Counter()
    by_drama = Counter()
    for m in data["metadatas"]:
        kb = m.get("kb_version", "?")
        by_kb[kb] += 1
        by_drama[(kb, m.get("drama_id"), m.get("source_episode"),
                  m.get("verification_status", "PENDING"))] += 1
    return by_kb, by_drama


def cmd_list(col, out=print):
    by_kb, by_drama = summarize(col)
    if not by_kb:
        out("사건이 없습니다.")
        return
    out(f"전체 사건 {sum(by_kb.values())}건")
    for kb, n in sorted(by_kb.items()):
        out(f"\n[{kb}] {n}건")
        rows = {}
        for (k, drama, ep, status), c in by_drama.items():
            if k != kb:
                continue
            rows.setdefault((drama, ep), Counter())[status] += c
        for (drama, ep), statuses in sorted(rows.items(), key=lambda r: (r[0][0] or 0, r[0][1] or 0)):
            detail = ", ".join(f"{s} {c}" for s, c in sorted(statuses.items()))
            out(f"  작품 {drama} / {ep}화: {sum(statuses.values())}건 ({detail})")


def _select(col, kb, dramas, episode):
    data = col.get(where=_where(kb, dramas, episode), include=["documents", "metadatas", "embeddings"])
    rows = list(zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"]))
    return rows


def cmd_copy(col, src, dst, dramas=None, episode=None, apply=False, remove_source=False, out=print):
    if src == dst:
        raise SystemExit("--from 과 --to 가 같습니다.")
    rows = _select(col, src, dramas, episode)
    verb = "옮기기" if remove_source else "복사"
    if not rows:
        out(f"[{src}] 에 대상 사건이 없습니다.")
        return 0
    targets = [(new_id(i, src, dst), d, {**m, "kb_version": dst}, e) for i, d, m, e in rows]
    already = set(col.get(ids=[t[0] for t in targets], include=[])["ids"])
    counts = Counter((m.get("drama_id"), m.get("source_episode")) for _, _, m, _ in rows)
    out(f"{verb}: [{src}] → [{dst}] 사건 {len(rows)}건"
        + (f" (이미 [{dst}] 에 있는 {len(already)}건은 덮어씀)" if already else ""))
    for (drama, ep), c in sorted(counts.items(), key=lambda r: (r[0][0] or 0, r[0][1] or 0)):
        out(f"  작품 {drama} / {ep}화: {c}건")
    if not apply:
        out("\n미리보기입니다. 실제로 실행하려면 --apply 를 붙이세요.")
        return 0

    for start in range(0, len(targets), BATCH):
        part = targets[start:start + BATCH]
        col.upsert(ids=[t[0] for t in part], documents=[t[1] for t in part],
                   metadatas=[t[2] for t in part], embeddings=[list(t[3]) for t in part])
    copied = set(col.get(ids=[t[0] for t in targets], include=[])["ids"])
    if copied != {t[0] for t in targets}:
        raise SystemExit("일부 사건이 복사되지 않았습니다. 원본은 지우지 않았습니다.")
    if remove_source:
        col.delete(ids=[r[0] for r in rows])
    out(f"완료: {len(targets)}건 {verb}")
    return len(targets)


def cmd_delete(col, kb, dramas=None, episode=None, apply=False, out=print):
    rows = col.get(where=_where(kb, dramas, episode), include=["metadatas"])
    ids, metas = rows["ids"], rows["metadatas"]
    if not ids:
        out(f"[{kb}] 에 대상 사건이 없습니다.")
        return 0
    counts = Counter((m.get("drama_id"), m.get("source_episode")) for m in metas)
    out(f"삭제: [{kb}] 사건 {len(ids)}건")
    for (drama, ep), c in sorted(counts.items(), key=lambda r: (r[0][0] or 0, r[0][1] or 0)):
        out(f"  작품 {drama} / {ep}화: {c}건")
    if not apply:
        out("\n미리보기입니다. 실제로 실행하려면 --apply 를 붙이세요.")
        return 0
    col.delete(ids=ids)
    out(f"완료: {len(ids)}건 삭제")
    return len(ids)


def _kb_arg(value):
    if not value or not set(value) <= _KB_RE_CHARS or len(value) > 20:
        raise argparse.ArgumentTypeError("칸 이름은 영문·숫자·-·_ 20자 이내")
    return value


def main(argv=None, col=None):
    p = argparse.ArgumentParser(description="사건 DB 칸(kb_version) 관리. 기본은 미리보기")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("copy", "move"):
        c = sub.add_parser(name)
        c.add_argument("--from", dest="src", required=True, type=_kb_arg)
        c.add_argument("--to", dest="dst", required=True, type=_kb_arg)
    d = sub.add_parser("delete")
    d.add_argument("--kb", required=True, type=_kb_arg)
    for c in (sub.choices["copy"], sub.choices["move"], d):
        c.add_argument("--drama", type=int, nargs="+")
        c.add_argument("--episode", type=int)
        c.add_argument("--apply", action="store_true", help="실제로 실행 (없으면 미리보기)")
    args = p.parse_args(argv)
    if getattr(args, "episode", None) is not None and not args.drama:
        p.error("--episode 는 --drama 와 함께 써야 합니다.")

    if col is None:
        import event_store
        col = event_store.get_collection()

    if args.cmd == "list":
        cmd_list(col)
    elif args.cmd in ("copy", "move"):
        cmd_copy(col, args.src, args.dst, args.drama, args.episode, args.apply,
                 remove_source=args.cmd == "move")
    else:
        cmd_delete(col, args.kb, args.drama, args.episode, args.apply)


if __name__ == "__main__":
    sys.exit(main())
