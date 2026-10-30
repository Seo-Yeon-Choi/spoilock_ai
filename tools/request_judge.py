r"""
서버 동작·속도 확인용 클라이언트 (서버를 켠 상태에서 다른 터미널에서 실행)

  .\.venv\Scripts\python.exe -m tools.request_judge "러시아 측이 오래전에 실종된 우주비행사들을 돌려달라고 요구했네" --watched-episode 4 --refresh
  .\.venv\Scripts\python.exe -m tools.request_judge "..." --refresh --repeat 5          # 반복 속도
  .\.venv\Scripts\python.exe -m tools.request_judge "..." --refresh --top-k 5 --no-evidence   # 입력 줄이기 실험
"""
import argparse
import json
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("message")
p.add_argument("--url", default="http://localhost:8000/judge")
p.add_argument("--watched-episode", type=int)
p.add_argument("--top-k", type=int, default=20)
p.add_argument("--no-evidence", action="store_true")
p.add_argument("--refresh", action="store_true")
p.add_argument("--repeat", type=int, default=1)
a = p.parse_args()

body = {"message": a.message, "top_k": a.top_k, "include_evidence": not a.no_evidence,
        "refresh": a.refresh}
if a.watched_episode is not None:
    body["watched_episode"] = a.watched_episode

totals = []
for i in range(a.repeat):
    req = urllib.request.Request(a.url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=90) as r:
        res = json.loads(r.read().decode("utf-8"))
    totals.append(time.perf_counter() - t)
    if i == 0:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[{i + 1}] {res['analysis_verdict']:9s} {res['match_cache']:10s} "
          f"왕복 {totals[-1]:.2f}s  서버 {res['timings_ms']}")

if a.repeat > 1:
    print(f"평균 왕복 {sum(totals) / len(totals):.2f}s, 최대 {max(totals):.2f}s")
