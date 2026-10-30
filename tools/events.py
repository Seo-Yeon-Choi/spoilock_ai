"""Validate, index or list event drafts: python -m tools.events COMMAND."""
import argparse
from types import SimpleNamespace

import event_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'index', 'list'])
    parser.add_argument('--episodes', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--drama-id', type=int, default=1)
    parser.add_argument('--kb-version', default='kb1')
    args = parser.parse_args()
    args.episodes = sorted(set(args.episodes))
    if min(args.episodes) < 1 or args.drama_id < 1:
        parser.error('Episode numbers and drama-id must be positive.')

    if args.command in ('validate', 'index'):
        total = 0
        # Validate all files before making any database updates.
        for episode in args.episodes:
            draft, _ = event_store.load_draft(episode)
            if draft.get('drama_id') != args.drama_id or draft.get('kb_version') != args.kb_version:
                raise ValueError(f'Episode {episode}: drama-id / kb-version mismatch')
            events = event_store.verify_draft_events(draft, episode)
            total += len(events)
            print(f'Episode {episode}: {len(events)} events, '
                  f'skipped batches={draft.get("skipped_batch_count", 0)}')
        print(f'Validated {total} draft events against source quotes. First-reveal accuracy is NOT verified.')
        if args.command == 'validate':
            return
        for episode in args.episodes:
            event_store.index_events(SimpleNamespace(
                episode=episode, drama_id=args.drama_id, kb_version=args.kb_version))
        return

    args.retrieval = 'all'
    rows = event_store.candidates_for(args)
    for episode in args.episodes:
        subset = sorted((r for r in rows if r['source_episode'] == episode), key=lambda r: r['event_id'])
        print(f'\nEpisode {episode}: {len(subset)} events')
        for row in subset:
            print(f"{row['event_id']} [{row['fact_status']}] {row['description']}")


if __name__ == '__main__':
    main()
