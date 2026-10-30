r"""
1차 필터 보정 도구 (서버를 켠 상태에서 실행)

A) 테스트 세트로 임계값 정하기
   .\.venv\Scripts\python.exe -m tools.calibrate_prefilter --csv tests/fixtures/prefilter_cases.csv

   prefilter_cases.csv 열: text, kind
     kind = CONTENT : 작품 내용(사건·인물 행동)을 언급 → 반드시 검사해야 함 (스포 여부 무관)
     kind = GENERAL : 인사·잡담·감상 등 내용 언급 없음 → 건너뛰어도 됨

B) shadow 모드로 쌓인 실제 로그 점검
   .\.venv\Scripts\python.exe -m tools.calibrate_prefilter --from-log
   → '필터는 SKIP이라 했는데 Gemini는 MATCH'인 채팅(=누출 위험)을 보여줌
"""
import argparse
import csv
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "tests" / "server_timing.jsonl"


def call(url, text):
    body = json.dumps({"message": text, "prefilter_mode": "shadow"}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))


def from_csv(args):
    with open(args.csv, encoding="utf-8-sig") as f:
        cases = [r for r in csv.DictReader(f) if r["text"].strip()]
    rows = []
    for i, c in enumerate(cases, 1):
        res = call(args.url, c["text"])
        pf = res["prefilter"]
        rows.append({**c, "stage": pf["stage"], "decision": pf["decision"],
                     "sim": pf["best_sim"], "verdict": res["analysis_verdict"]})
        print(f"\r{i}/{len(cases)}", end="", flush=True)
    print()

    content = [r for r in rows if r["kind"] == "CONTENT"]
    general = [r for r in rows if r["kind"] == "GENERAL"]

    print("\n=== 1) 규칙 필터 ===")
    rule_leak = [r for r in content if r["stage"] == "rule"]
    rule_skip = [r for r in general if r["stage"] == "rule"]
    print(f"내용 채팅이 규칙으로 통과(누출) : {len(rule_leak)}/{len(content)}   <- 반드시 0")
    for r in rule_leak:
        print(f"   [누출] {r['text']}")
    print(f"일반 채팅이 규칙으로 통과      : {len(rule_skip)}/{len(general)}")

    print("\n=== 2) 강제 검사(must_check) ===")
    forced_g = [r for r in general if r["stage"] == "must_check"]
    print(f"일반 채팅인데 강제 검사됨       : {len(forced_g)}/{len(general)}  (안전하지만 비용)")
    for r in forced_g[:10]:
        print(f"   {r['text']}")

    print("\n=== 3) 유사도 분포 (규칙·강제검사에 안 걸린 채팅만) ===")
    c_sims = sorted(r["sim"] for r in content if r["stage"] not in ("rule",) and r["sim"] is not None)
    g_sims = sorted(r["sim"] for r in general
                    if r["stage"] not in ("rule", "must_check") and r["sim"] is not None)
    if c_sims:
        print(f"내용 채팅 유사도 : 최소 {c_sims[0]:.4f} / 중앙 {c_sims[len(c_sims) // 2]:.4f}")
    if g_sims:
        print(f"일반 채팅 유사도 : 최대 {g_sims[-1]:.4f} / 중앙 {g_sims[len(g_sims) // 2]:.4f}")
    if c_sims and g_sims:
        th = round(c_sims[0] - args.margin, 4)
        saved = sum(s < th for s in g_sims)
        print(f"\n권장 PREFILTER_SIM_THRESHOLD = {th}  (내용 채팅 최소값 - 여유 {args.margin})")
        print(f"→ 이 값이면 유사도로 건너뛰는 일반 채팅: {saved}/{len(g_sims)}, 이 표본에서 내용 채팅 누출: 0 (다른 문장에 대한 보장 아님)")
        lowest = sorted((r for r in content if r["sim"] is not None and r["stage"] != "rule"),
                        key=lambda r: r["sim"])[:5]
        print("임계값을 정한 내용 채팅(유사도 낮은 순):")
        for r in lowest:
            print(f"   {r['sim']:.4f}  {r['text']}")
        if saved == 0:
            print("※ 일반/내용 채팅 유사도가 겹쳐서 유사도 필터 효과가 없습니다. 규칙 필터만 쓰세요.")

    total_skip_rule = len(rule_skip)
    print(f"\n=== 요약: 일반 채팅 {len(general)}개 중 규칙만으로 {total_skip_rule}개 생략 ===")


def from_log(_args):
    if not LOG.is_file():
        print(f"로그 없음: {LOG}")
        return
    rows = [json.loads(l) for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("prefilter") and r["prefilter"].get("mode") == "shadow"]
    would_skip = [r for r in rows if r["prefilter"]["decision"] == "SKIP"]
    danger = [r for r in would_skip if r["analysis_verdict"] in ("MATCH", "UNCERTAIN")]
    print(f"shadow 기록 {len(rows)}건 중 필터가 SKIP 판단: {len(would_skip)}건 "
          f"({len(would_skip) / max(1, len(rows)) * 100:.0f}% Gemini 호출 절감 예상)")
    print(f"SKIP 판단인데 Gemini가 MATCH/UNCERTAIN: {len(danger)}건   <- 0건이어도 안전 보장은 아님; 별도 검증 필요")
    for r in danger:
        pf = r["prefilter"]
        print(f"   [{r['analysis_verdict']}] {pf['stage']} sim={pf['best_sim']}  {r['message']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--from-log", action="store_true")
    ap.add_argument("--url", default="http://localhost:8000/judge")
    ap.add_argument("--margin", type=float, default=0.02)
    a = ap.parse_args()
    from_log(a) if a.from_log else from_csv(a) if a.csv else ap.print_help()
