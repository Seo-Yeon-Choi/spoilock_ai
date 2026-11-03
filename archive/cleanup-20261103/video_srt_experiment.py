"""
SPOILOCK 실험: 영상 화면 + 공식 SRT 를 함께 Gemini 에 보내 사건 후보 추출

목적
  자막만으로 추출한 사건(ingest_jobs)과 비교해서, 화면 정보를 더하면
  대사에 드러나지 않는 사건(장면으로만 보이는 반전 등)을 더 잡는지 확인한다.

흐름
  SRT → 청크(ingest_core, 자막 분석과 동일) → 배치(청크 6개, 2개 겹침)
  영상은 Gemini Files API 에 한 번만 업로드
  배치마다: 그 시간대 영상 구간 + 그 구간 자막 → Gemini → 사건 후보
  근거 검증
    - 대사 근거(DIALOGUE): 인용문이 자막 원문에 그대로 있어야 함 (기존과 동일)
    - 화면 근거(VISUAL)  : 시각이 그 구간 안이어야 함 (관리자가 그 시각을 직접 확인)
  결과: experiments/<실행시각>/events.json, report.md

안전
  Chroma·Spring DB 에 아무것도 쓰지 않는다 (읽기 전용 실험).
  중간에 실패하면 같은 --out 으로 다시 실행해 끝난 배치를 건너뛴다.

실행 예
  .venv\\Scripts\\python.exe video_srt_experiment.py ^
      --video data\\raw\\tears_of_steel_480p.mp4 --srt data\\raw\\tears_of_steel_en.srt ^
      --episode 1 --characters "Thom,Celia,Barley,Captain" ^
      --compare data\\kb\\d2\\ep07\\events_draft.json
  처음엔 --max-batches 1 로 1배치만 돌려 연결을 확인하는 걸 권장.

구간 지정 방식 (--clip-mode)
  api    : 업로드한 영상 1개에 시작·끝 시각만 지정 (기본, 업로드 1회)
  ffmpeg : 배치마다 ffmpeg 로 구간을 잘라 따로 업로드 (api 모드가 400/500 오류를 낼 때)
"""
import argparse
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import ingest_core as core

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"

from ingest_video import (   # noqa: F401  (테스트·보고서에서 사용)
    EVIDENCE_TYPES, SCHEMA, SYSTEM, VideoBatchExtractor as Extractor,
    mmss, parse_mmss, segment_for, validate_events,
)

# ================================================================ 실행
def check_inputs(args):
    """시작 전에 파일 경로 확인 (업로드 단계에서 늦게 실패하지 않도록)"""
    missing = [f"  --{name}: {Path(value).resolve()}"
               for name, value in (("video", args.video), ("srt", args.srt), ("compare", args.compare))
               if value and not Path(value).is_file()]
    if missing:
        raise SystemExit("파일을 찾을 수 없습니다. 경로와 파일 이름을 확인해 주세요.\n" + "\n".join(missing)
                         + f"\n(현재 실행 위치: {Path.cwd()})")


def run(args, client=None, model=None, sleep=time.sleep):
    check_inputs(args)
    rows, skipped = core.parse_srt_text(core.decode_srt(Path(args.srt).read_bytes()))
    chunks = core.build_chunks(rows, drama_id=0, episode_id=0, episode=args.episode, kb_version="exp",
                               language=args.language, source_name=Path(args.srt).name)
    ranges = core.batch_ranges(len(chunks), args.batch_size, args.overlap)
    characters = [c.strip() for c in (args.characters or "").split(",") if c.strip()]

    out = Path(args.out) if args.out else EXPERIMENTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.json"
    settings = {"srt": Path(args.srt).name, "video": Path(args.video).name, "episode": args.episode,
                "batch_size": args.batch_size, "overlap": args.overlap, "clip_mode": args.clip_mode,
                "resolution": args.resolution, "fps": args.fps}
    state = {"settings": settings, "next_batch": 0, "events": [], "rejected": [], "segments": []}
    if progress_path.is_file():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved.get("settings") != settings:
            raise SystemExit(f"{out} 의 이전 실행과 설정이 다릅니다. 다른 --out 을 쓰세요.")
        state = saved
        print(f"이어서 진행: 배치 {state['next_batch'] + 1}/{len(ranges)}부터", flush=True)

    if client is None:
        import gemini_client
        client, default_model = gemini_client.get_client()
        model = args.model or default_model
    print(f"자막 {len(rows)}줄 → 청크 {len(chunks)}개 → 배치 {len(ranges)}개, 모델 {model}", flush=True)

    extractor = Extractor(client, model, Path(args.video), clip_mode=args.clip_mode,
                          resolution=args.resolution, fps=args.fps,
                          workdir=Path(tempfile.mkdtemp(prefix="spoilock_clips_")))
    if state["next_batch"] < len(ranges):
        extractor.prepare()
    else:
        print("모든 배치가 이미 끝났습니다. Gemini 호출 없이 보고서만 다시 만듭니다.", flush=True)
    seen = {core.compact(e["description"]).lower() for e in state["events"]}
    done_now = 0
    try:
        for b in range(state["next_batch"], len(ranges)):
            if args.max_batches and done_now >= args.max_batches:
                print(f"--max-batches {args.max_batches} 에서 멈춤. 같은 --out 으로 다시 실행하면 이어서 진행합니다.")
                break
            s, e = ranges[b]
            batch = chunks[s:e]
            for attempt in range(1, 4):
                try:
                    (accepted, rejected), (seg_start, seg_end) = extractor.extract(
                        args.episode, batch, characters)
                    break
                except Exception as exc:
                    if attempt == 3:
                        raise
                    print(f"  배치 {b + 1} 실패({type(exc).__name__}: {str(exc)[:120]}), 재시도 {attempt}/2",
                          flush=True)
                    sleep(5 * attempt)
            added = 0
            for ev in accepted:
                key = core.compact(ev["description"]).lower()
                if key in seen:
                    continue
                seen.add(key)
                ev["batch"] = b + 1
                ev["segment"] = [mmss(seg_start), mmss(seg_end)]
                state["events"].append(ev)
                added += 1
            state["rejected"].extend({"batch": b + 1, **r} for r in rejected)
            state["segments"].append({"batch": b + 1, "start": mmss(seg_start), "end": mmss(seg_end)})
            state["next_batch"] = b + 1
            core.save_json(progress_path, state)
            done_now += 1
            print(f"배치 {b + 1}/{len(ranges)} [{mmss(seg_start)}~{mmss(seg_end)}] "
                  f"사건 {added}건 추가 (버림 {len(rejected)}건)", flush=True)
            if b + 1 < len(ranges):
                sleep(args.sleep)
    finally:
        extractor.close(args.keep_files)

    result = {"settings": settings, "model": model, "finished": state["next_batch"] >= len(ranges),
              "subtitles": len(rows), "skipped_blocks": skipped, "chunks": len(chunks),
              "batches_done": state["next_batch"], "batches_total": len(ranges),
              "usage_this_run": extractor.usage, "events": state["events"], "rejected": state["rejected"]}
    core.save_json(out / "events.json", result)
    compare = load_compare(args.compare) if args.compare else None
    (out / "report.md").write_text(render_report(result, compare), encoding="utf-8")
    print(f"\n결과: {out / 'events.json'}\n보고서: {out / 'report.md'}")
    return result


def load_compare(path):
    """자막만으로 추출한 결과 (ingest_jobs 의 events_draft.json 또는 예전 epXX_events_draft.json)"""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    events = data.get("events", []) if isinstance(data, dict) else data
    return [e.get("description", "") for e in events]


def render_report(result, compare):
    evs = result["events"]
    by_type = {t: [e for e in evs if e["evidence_type"] == t] for t in EVIDENCE_TYPES}
    lines = [
        "# 영상 + 자막 사건 추출 실험",
        "",
        f"- 영상: `{result['settings']['video']}` / 자막: `{result['settings']['srt']}` / 모델: `{result['model']}`",
        f"- 자막 {result['subtitles']}줄, 청크 {result['chunks']}개, 배치 {result['batches_done']}/{result['batches_total']}"
        + ("" if result["finished"] else " **(미완료)**"),
        f"- 사건 {len(evs)}건: 대사+화면 {len(by_type['BOTH'])}, 대사만 {len(by_type['DIALOGUE'])}, "
        f"**화면만 {len(by_type['VISUAL'])}**, 버림 {len(result['rejected'])}",
        f"- 이번 실행 토큰: 입력 {result['usage_this_run']['prompt_tokens']:,} / "
        f"출력 {result['usage_this_run']['output_tokens']:,} ({result['usage_this_run']['calls']}회 호출)",
        "",
        "## 사건 목록 (시간순)",
        "",
        "| 시각 | 근거 | 사건 | 화면 근거 | 대사 근거 |",
        "|---|---|---|---|---|",
    ]

    def first_time(e):
        # 화면 근거 시각, 없으면 인용한 자막 청크의 시작 시각
        ts = [v["time_ms"] for v in e["visual_evidence"]]
        ts += [d["time_ms"] for d in e["dialogue_evidence"] if d.get("time_ms") is not None]
        return min(ts) if ts else parse_mmss(e["segment"][0]) or 0

    for e in sorted(evs, key=first_time):
        visual = "<br>".join(f"{v['timestamp']} {v['description_ko']}" for v in e["visual_evidence"]) or "-"
        dialogue = "<br>".join(f"\"{d['quote']}\"" for d in e["dialogue_evidence"]) or "-"
        lines.append(f"| {mmss(first_time(e))} | {e['evidence_type']} | {e['description']} | "
                     f"{visual} | {dialogue} |")

    if compare is not None:
        lines += ["", f"## 비교: 자막만으로 추출한 사건 ({len(compare)}건)", ""]
        lines += [f"- {d}" for d in compare]
        lines += ["", "> 화면만(VISUAL) 사건 중 위 목록에 없는 것이 '영상을 더해서 새로 잡은 사건'입니다.",
                  "> 관리자가 보고서의 시각으로 영상을 직접 확인해 맞는지 판단하세요."]

    if result["rejected"]:
        lines += ["", "## 버린 사건 (근거 검증 실패)", ""]
        for r in result["rejected"]:
            desc = (r.get("event") or {}).get("description_ko", "")
            lines.append(f"- 배치 {r['batch']}: {desc} — {r['reason']}")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="영상 화면 + SRT 사건 추출 실험 (Chroma 에 쓰지 않음)")
    p.add_argument("--video", required=True)
    p.add_argument("--srt", required=True, help="영상과 시간이 맞는 자막")
    p.add_argument("--episode", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--characters", default="", help="주요 인물 이름, 쉼표 구분 (예: Thom,Celia)")
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--overlap", type=int, default=2)
    p.add_argument("--clip-mode", choices=["api", "ffmpeg"], default="api")
    p.add_argument("--resolution", choices=["low", "medium", "high"], default="low")
    p.add_argument("--fps", type=float, default=None, help="초당 프레임 (기본 1, 예: 0.5)")
    p.add_argument("--model", default=None, help="비우면 gemini_client 기본 모델")
    p.add_argument("--max-batches", type=int, default=0, help="이번 실행에서 처리할 배치 수 (0=전부)")
    p.add_argument("--sleep", type=float, default=4.0, help="배치 사이 대기(초), 분당 한도 대비")
    p.add_argument("--out", default=None, help="결과 폴더 (같은 폴더로 다시 실행하면 이어서 진행)")
    p.add_argument("--compare", default=None, help="자막만 추출 결과 events_draft.json")
    p.add_argument("--keep-files", action="store_true", help="Gemini 에 올린 영상을 지우지 않음")
    args = p.parse_args()
    if not (1 <= args.overlap < args.batch_size):
        p.error("1 <= overlap < batch-size")
    run(args)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", encoding="utf-8-sig", override=False)
    sys.exit(main())
