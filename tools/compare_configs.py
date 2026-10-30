r"""
Gemini 설정별 속도·판정 비교 (서버를 켠 상태에서 실행)

  .\.venv\Scripts\python.exe -m tools.compare_configs --csv tests/fixtures/prefilter_cases.csv --kind CONTENT

- 같은 문장들을 여러 설정으로 보내(캐시 무시, 1차 필터 off) Gemini 시간과 토큰을 비교합니다.
- 기준 설정(A)과 판정·매칭 사건이 달라진 문장을 모두 보여줍니다.
  → 빨라졌는데 판정이 바뀌면 그 설정은 쓰면 안 됩니다.
- 문장 수 × 설정 수만큼 Gemini를 호출합니다. (20문장 × 5설정 = 100회)
"""
import argparse
import csv
import json
import statistics
import urllib.request

CONFIGS = {
    "A 기준(현재)":          dict(top_k=20, include_evidence=True,  reason_mode="full"),
    "B evidence 제외":       dict(top_k=20, include_evidence=False, reason_mode="full"),
    "C 이유 짧게":           dict(top_k=20, include_evidence=True,  reason_mode="short"),
    "D top10·evid제외·짧게": dict(top_k=10, include_evidence=False, reason_mode="short"),
    "E top5·evid제외·이유X": dict(top_k=5,  include_evidence=False, reason_mode="none"),
}


def call(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="tests/fixtures/prefilter_cases.csv")
    ap.add_argument("--kind", default="CONTENT", help="이 kind의 문장만 사용 (빈 값이면 전부)")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--url", default="http://localhost:8000/judge")
    a = ap.parse_args()

    with open(a.csv, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["text"].strip()]
    if a.kind:
        rows = [r for r in rows if r.get("kind") == a.kind]
    texts = [r["text"] for r in rows][: a.limit]
    print(f"문장 {len(texts)}개 × 설정 {len(CONFIGS)}개 = Gemini {len(texts) * len(CONFIGS)}회 호출\n")

    results = {}
    for name, cfg in CONFIGS.items():
        out = []
        for i, t in enumerate(texts, 1):
            body = {"message": t, "refresh": True, "prefilter_mode": "off", **cfg}
            out.append(call(a.url, body))
            print(f"\r{name}: {i}/{len(texts)}", end="", flush=True)
        results[name] = out
        print()

    base_name = next(iter(CONFIGS))
    base = results[base_name]
    print(f"\n{'설정':22s} {'평균':>7s} {'중앙':>7s} {'최대':>7s} {'입력tok':>8s} {'출력tok':>8s} {'판정일치':>8s} {'오류':>4s}")
    for name, out in results.items():
        g = [r["timings_ms"].get("gemini", 0) for r in out]
        tin = [r["timings_ms"].get("tokens_in") or 0 for r in out]
        tout = [r["timings_ms"].get("tokens_out") or 0 for r in out]
        same = sum(r["analysis_verdict"] == b["analysis_verdict"] and r["matched_event_id"] == b["matched_event_id"]
                   for r, b in zip(out, base))
        errs = sum(r["analysis_verdict"] == "ERROR" for r in out)
        print(f"{name:22s} {statistics.mean(g)/1000:6.2f}s {pct(g, .5)/1000:6.2f}s {max(g)/1000:6.2f}s "
              f"{statistics.mean(tin):8.0f} {statistics.mean(tout):8.0f} {same:>4d}/{len(out):<3d} {errs:>4d}")

    print(f"\n=== 기준({base_name})과 판정이 달라진 문장 ===")
    any_diff = False
    for name, out in results.items():
        if name == base_name:
            continue
        for t, r, b in zip(texts, out, base):
            if r["analysis_verdict"] != b["analysis_verdict"] or r["matched_event_id"] != b["matched_event_id"]:
                any_diff = True
                print(f"[{name}] {t}\n    기준: {b['analysis_verdict']} {b['matched_event_id']}  →  "
                      f"이 설정: {r['analysis_verdict']} {r['matched_event_id']}")
    if not any_diff:
        print("없음")
    print("\n※ 같은 설정으로 다시 돌려도 Gemini 판정이 가끔 흔들릴 수 있습니다. "
          "달라진 문장은 테스트 페이지에서 직접 확인해 보세요.")


if __name__ == "__main__":
    main()
