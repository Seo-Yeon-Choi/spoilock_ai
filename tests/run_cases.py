"""Run draft-based 30-case evaluation. No gold-label spoiler accuracy claim."""
import argparse
import json
import urllib.request
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def grade(case, result):
    checks = {
        'verdict': result.get('analysis_verdict') == case['expected_verdict'],
        'visibility': result.get('visibility') == case['expected_visibility'],
    }
    if case['expected_verdict'] == 'MATCH':
        checks['event'] = result.get('matched_event_id') in case['acceptable_event_ids']
        checks['source_episode'] = result.get('required_episode_number') == case['expected_source_episode']
    else:
        checks['event'] = result.get('matched_event_id') is None
        checks['source_episode'] = result.get('required_episode_number') is None
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://localhost:8000/judge')
    p.add_argument('--retrieval', choices=['all', 'vector'], default='vector')
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--refresh-match', action='store_true')
    p.add_argument('--limit', type=int, default=30)
    args = p.parse_args()
    if not 1 <= args.limit <= 30 or args.top_k < 1:
        p.error('Require limit 1..30 and top-k >= 1')
    cases = json.loads((ROOT/'tests/fixtures/test_cases_30.json').read_text(encoding='utf-8-sig'))['cases'][:args.limit]
    rows=[]
    for case in cases:
        body = dict(message=case['message'], watched_episode=case['watched_episode'],
                    retrieval=args.retrieval, top_k=args.top_k, refresh=args.refresh_match,
                    prefilter_mode='off', reason_mode='full', include_evidence=True)
        start=time.perf_counter()
        row={'case':case}
        try:
            request = urllib.request.Request(args.url,
                data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=90) as response:
                result=json.loads(response.read().decode('utf-8'))
            checks=grade(case,result)
            row.update(result=result,checks=checks,passed=all(checks.values()))
            failed=','.join(k for k,v in checks.items() if not v)
            print(f"{case['id']} {'PASS' if row['passed'] else 'FAIL'} | "
                  f"{result['analysis_verdict']} / {result['visibility']}"
                  + (f' | mismatch={failed}' if failed else ''),flush=True)
        except (ValueError,OSError,RuntimeError) as exc:
            row.update(passed=False,error=str(exc))
            print(f"{case['id']} ERROR | {exc}",flush=True)
        row['elapsed_request_seconds']=round(time.perf_counter()-start,3)
        rows.append(row)
        # Save after each completed case to retain progress on interruption.
        save_report(rows,args)
    summary=summary_for(rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(f'Report: data/tests/report_30_{args.retrieval}.json')


def summary_for(rows):
    valid=[r for r in rows if 'result' in r]
    positives=[r for r in rows if r['case']['expected_verdict']=='MATCH']
    negatives=[r for r in rows if r['case']['expected_verdict']=='NO_MATCH']
    def pct(n,d):return round(100*n/d,1) if d else None
    return {
        'evaluation':'DRAFT_BASED_NOT_VERIFIED_SPOILER_ACCURACY',
        'total':len(rows),'completed':len(valid),'errors':len(rows)-len(valid),
        'strict_pass_count':sum(r['passed'] for r in rows),
        'strict_pass_rate_including_errors_pct':pct(sum(r['passed'] for r in rows),len(rows)),
        'verdict_agreement_completed_pct':pct(sum(r['checks']['verdict'] for r in valid),len(valid)),
        'expected_event_hit_rate_including_errors_pct':pct(sum(r.get('checks',{}).get('event',False) for r in positives),len(positives)),
        'general_false_match_count':sum(r.get('result',{}).get('analysis_verdict')=='MATCH' for r in negatives),
        'review_count':sum(r['result']['visibility']=='REVIEW' for r in valid),
        'cache_hits':sum(str(r['result'].get('match_cache','')).startswith('HIT') for r in valid),
    }


def save_report(rows,args):
    directory=ROOT/'data/tests';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'report_30_{args.retrieval}.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps({'created_at':datetime.now().isoformat(),
        'config':vars(args),'summary':summary_for(rows),'results':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


if __name__=='__main__':main()
