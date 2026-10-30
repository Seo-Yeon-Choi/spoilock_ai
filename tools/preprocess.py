"""SRT preprocessing for draft event extraction (Python 3.11+)."""
import argparse
import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIME = r'(\d{2}:\d{2}:\d{2},\d{3})'
BLOCK = re.compile(r'^(\d+)\n' + TIME + r'\s*-->\s*' + TIME + r'\n([\s\S]+)$')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def millis(value):
    h, m, s, ms = map(int, re.split('[:,]', value))
    if m >= 60 or s >= 60:
        raise ValueError(f'Invalid timestamp: {value}')
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def timestamp(ms):
    s, rem = divmod(ms, 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f'{h:02}:{m:02}:{s:02}.{rem:03}'


def parse_srt(path):
    text = path.read_text(encoding='utf-8-sig').replace('\r\n', '\n').replace('\r', '\n')
    rows, seen = [], set()
    for pos, block in enumerate(re.split(r'\n\s*\n', text.strip()), 1):
        match = BLOCK.fullmatch(block.strip())
        if not match:
            raise ValueError(f'Invalid SRT block at position {pos}; not skipped silently.')
        index, start, end, raw = match.groups()
        index, start, end = int(index), millis(start), millis(end)
        if index in seen or end < start or (rows and start < rows[-1]['start_ms']):
            raise ValueError(f'Duplicate index or invalid time order: {index}')
        seen.add(index)
        cleaned = html.unescape(re.sub(r'<[^>]+>', '', raw))
        cleaned = '\n'.join(re.sub(r'[ \t]+', ' ', line).strip() for line in cleaned.splitlines()).strip()
        if not cleaned:
            raise ValueError(f'Empty subtitle: {index}')
        rows.append(dict(subtitle_index=index, start_ms=start, end_ms=end, raw_text=raw, text=cleaned))
    return rows


def make_chunks(rows, window, overlap, max_chars, gap_ms):
    if window < 1 or not 0 <= overlap < window or max_chars < 1 or gap_ms < 0:
        raise ValueError('Require window > overlap >= 0, max_chars > 0 and gap_ms >= 0')
    i = 0
    while i < len(rows):
        j, size, hard_gap = i, 0, False
        while j < len(rows) and j - i < window:
            if j > i and rows[j]['start_ms'] - rows[j-1]['end_ms'] > gap_ms:
                hard_gap = True
                break
            extra = len(rows[j]['text']) + (1 if j > i else 0)
            if j > i and size + extra > max_chars:
                break
            size += extra
            j += 1
        yield rows[i:j]
        if j == len(rows):
            break
        i = j if hard_gap else max(i + 1, j - overlap)


def preprocess(args):
    rows = parse_srt(Path(args.file))
    excluded_ids = set(args.exclude_index)
    missing = excluded_ids - {r['subtitle_index'] for r in rows}
    if missing:
        raise ValueError(f'Excluded indices do not exist: {sorted(missing)}')
    selected = [r for r in rows if r['subtitle_index'] not in excluded_ids]
    if not selected:
        raise ValueError('No subtitles retained')
    records = []
    for n, group in enumerate(make_chunks(selected, args.window, args.overlap, args.max_chars, args.gap_ms), 1):
        records.append({
            'id': f'd{args.drama_id}-e{args.episode_id}-{args.kb_version}-c{n:04}',
            'document': '\n'.join(r['text'] for r in group),
            'metadata': dict(drama_id=args.drama_id, episode_id=args.episode_id,
                episode_number=args.episode_number, kb_version=args.kb_version,
                language='en', source_file=Path(args.file).name, chunk_index=n,
                start_ms=group[0]['start_ms'], end_ms=max(r['end_ms'] for r in group),
                subtitle_start=group[0]['subtitle_index'], subtitle_end=group[-1]['subtitle_index']),
            'subtitle_indices': [r['subtitle_index'] for r in group]
        })
    out = Path(args.out)
    dump(out / 'subtitles.json', rows)
    dump(out / 'excluded.json', [r for r in rows if r['subtitle_index'] in excluded_ids])
    dump(out / 'chunks.json', records)
    report = dict(parsed=len(rows), retained=len(selected), excluded=len(rows)-len(selected), chunks=len(records),
        first_timestamp=timestamp(rows[0]['start_ms']), last_dialogue_timestamp=timestamp(selected[-1]['end_ms']),
        source_sha256=hashlib.sha256(Path(args.file).read_bytes()).hexdigest(),
        settings=dict(window=args.window, overlap=args.overlap, max_chars=args.max_chars, gap_ms=args.gap_ms),
        note='IDs are local practice values until matched to MySQL. No LLM extraction or spoiler decision performed.')
    dump(out / 'report.json', report)
    preview = []
    for r in records:
        m = r['metadata']
        preview.append(f"[{r['id']}] subtitles {m['subtitle_start']}-{m['subtitle_end']} | "
            f"{timestamp(m['start_ms'])} ~ {timestamp(m['end_ms'])}\n{r['document']}")
    (out / 'chunks_preview.txt').write_text('\n\n'.join(preview), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    p = parser
    p.add_argument('--file', default=str(ROOT / 'data/raw/ep01.srt'))
    p.add_argument('--out', default=str(ROOT / 'data/processed/ep01'))
    p.add_argument('--drama-id', type=int, default=1)
    p.add_argument('--episode-id', type=int, default=101)
    p.add_argument('--episode-number', type=int, default=1)
    p.add_argument('--kb-version', default='kb1')
    p.add_argument('--window', type=int, default=15)
    p.add_argument('--overlap', type=int, default=3)
    p.add_argument('--max-chars', type=int, default=1000)
    p.add_argument('--gap-ms', type=int, default=15000)
    p.add_argument('--exclude-index', type=int, nargs='*', default=[])
    p.set_defaults(run=preprocess)
    args = parser.parse_args()
    args.run(args)


if __name__ == '__main__':
    main()
