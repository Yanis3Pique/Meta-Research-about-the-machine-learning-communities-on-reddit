#!/usr/bin/env python3
"""Install accepted Task F domain mappings without counting suspect parser artifacts.

From repository root (PowerShell):
    python finish_task_f.py --apply
    python 20_links.py
    python finish_task_f.py --check

Put both accepted files in results/ first:
    f_domain_work_4635_MERGED_ACCEPTED.csv
    f_domain_work_4635_MERGED_ACCEPTED_with_audit.csv

By default, the program only previews changes. --apply backs up and updates
config/domain_categories.csv. --check reports current post/link coverage after re-running
20_links.py (or the current Task F pipeline). Does not modify research outputs.
"""
import argparse
import csv
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

CATEGORIES = {
    'academic', 'code', 'models/hubs', 'blog/tutorial', 'company/product',
    'news/media', 'social', 'video', 'reddit-internal', 'reddit-media', 'other',
}


def load_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, reader.fieldnames or []


def mapping_of(rows, domain_key, category_key, label, allow_identical_duplicates=False):
    result = {}
    repeats = 0
    for i, row in enumerate(rows, 2):
        dom = row.get(domain_key, '').strip().lower()
        cat = row.get(category_key, '').strip()
        if not dom or cat not in CATEGORIES:
            raise ValueError(f'{label}: invalid domain/category at CSV line {i}: {dom!r} {cat!r}')
        if dom in result:
            if not allow_identical_duplicates or result[dom] != cat:
                raise ValueError(f'{label}: duplicate/conflicting domain: {dom}')
            repeats += 1
        result[dom] = cat
    return result, repeats


def run_install(root, accepted_path, audit_path, apply):
    conf = root / 'config' / 'domain_categories.csv'
    if not conf.exists():
        raise FileNotFoundError(f'Existing reviewed taxonomy missing: {conf}')
    accepted_rows, _ = load_csv(accepted_path)
    accepted_map, _ = mapping_of(accepted_rows, 'domain', 'final_category', 'Accepted Work')
    if any(x.get('final_category', '') == 'unclassified' for x in accepted_rows):
        raise ValueError('Accepted Work still has unclassified categories')
    if len(accepted_map) != 4635:
        raise ValueError(f'Expected 4,635 accepted domains; got {len(accepted_map)}')

    audit_rows, _ = load_csv(audit_path)
    audited_map, _ = mapping_of(audit_rows, 'domain', 'final_category', 'Audit')
    if accepted_map != audited_map:
        raise ValueError('The audit and accepted CSV differ in domain/category mappings')
    excluded = {
        r['domain'].strip().lower()
        for r in audit_rows
        if r.get('link_validity', '').strip() == 'suspected_false_positive'
    }
    missing_validity = [x for x in excluded if x not in accepted_map]
    if missing_validity:
        raise ValueError(f'Suspect domains missing from mapping: {missing_validity[:5]}')

    existing_rows, existing_headers = load_csv(conf)
    current, identical_duplicates = mapping_of(
        existing_rows, 'domain', 'category', 'Existing taxonomy',
        allow_identical_duplicates=True,
    )
    conflicts = [
        (d, current[d], c) for d, c in accepted_map.items()
        if d in current and current[d] != c
    ]
    if conflicts:
        # Existing human-reviewed classifications take precedence.
        print(f'WARNING: {len(conflicts)} conflicts; existing taxonomy takes priority.')
        for d, prev, new in conflicts[:15]:
            print(f'   {d}: existing={prev}, proposed={new}')

    additions = {
        d: cat for d, cat in accepted_map.items()
        if d not in current and d not in excluded
    }
    merged = {**current, **additions}
    print(f'Existing taxonomy: {len(existing_rows)} rows / {len(current)} unique')
    print(f'Accepted Work: {len(accepted_map)} domains')
    print(f'Suspected false-positive links excluded from mapping: {len(excluded)}')
    print(f'New mappings: {len(additions)}')
    print(f'Preserved existing-category conflicts: {len(conflicts)}')
    print(f'Removed identical duplicate baseline rows: {identical_duplicates}')
    print(f'Final domain taxonomy: {len(merged)} unique domains')

    if apply:
        suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = conf.with_name(f'domain_categories.backup_{suffix}.csv')
        shutil.copy2(conf, backup)
        # Keep known metadata columns, including review_url, if the local project has any.
        fieldnames = existing_headers if existing_headers else ['domain', 'category']
        metadata = {r['domain'].strip().lower(): r for r in existing_rows}
        with conf.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for domain in sorted(merged):
                row = {key: '' for key in fieldnames}
                if domain in metadata:
                    row.update({k: v for k, v in metadata[domain].items() if k in row})
                row['domain'], row['category'] = domain, merged[domain]
                writer.writerow(row)
        print(f'BACKUP: {backup}')
        print(f'UPDATED: {conf}')
        print('Next: rerun python 20_links.py, then python finish_task_f.py --check')
    else:
        print('PREVIEW ONLY: run with --apply after checking these counts.')


def run_check(root):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError('pandas + pyarrow required for Parquet coverage check') from exc
    path = root / 'processed' / 'link_features.parquet'
    if not path.exists():
        raise FileNotFoundError(f'Rerun Task F first to create {path}')
    data = pd.read_parquet(path)
    required = {'has_link', 'fully_mapped'}
    if not required.issubset(data.columns):
        raise ValueError(f'Expected columns missing: {sorted(required - set(data.columns))}')
    linked = data.loc[data['has_link'].eq(1)]
    n = len(linked)
    mapped = int(linked['fully_mapped'].eq(1).sum())
    percent = mapped / n * 100 if n else 0
    print('\n-- ACTUAL POST COVERAGE (RECOMPUTED FEATURE FILE) --')
    print(f'Link-bearing posts: {n:,}')
    print(f'Fully mapped link-bearing posts: {mapped:,}')
    print(f'Coverage: {percent:.4f}%  =>  {"PASS" if percent >= 95 else "FAIL"} >=95%')

    if 'subreddit' in linked.columns:
        for name, sub in linked.groupby('subreddit', sort=True):
            sub_pct = sub['fully_mapped'].eq(1).mean() * 100
            print(f'  {name}: {sub_pct:.2f}% ({len(sub):,} linked posts)')

    # 20_links.py writes results/f_coverage.csv and processed/link_features.parquet.
    # processed/link_events.parquet is an older pipeline artifact and is NOT
    # regenerated by 20_links.py; do not report its "reviewed" share as current.
    coverage_file = root / 'results' / 'f_coverage.csv'
    if coverage_file.exists():
        table = pd.read_csv(coverage_file)
        needed = {'subreddit', 'period', 'metric', 'mapped', 'total', 'coverage_pct'}
        if not needed.issubset(table.columns):
            raise ValueError(f'Missing coverage table columns: {sorted(needed - set(table.columns))}')
        current = table.loc[table['subreddit'].eq('ALL') & table['period'].eq('all')]
        if current['metric'].duplicated().any():
            raise ValueError('Duplicate overall coverage metrics in f_coverage.csv')
        current = current.set_index('metric')
        metric_names = ('individual_links', 'primary_links', 'post_domains', 'fully_mapped_posts')
        if any(m not in current.index for m in metric_names):
            raise ValueError('Overall f_coverage.csv is missing one or more required metrics')
        row = current.loc['fully_mapped_posts']
        if int(row['mapped']) != mapped or int(row['total']) != n:
            raise RuntimeError('Stale/inconsistent f_coverage.csv: fully_mapped_posts counts '
                               'do not agree with processed/link_features.parquet')
        print('\n-- FOUR CURRENT COVERAGE MEASURES (results/f_coverage.csv) --')
        for metric in metric_names:
            row = current.loc[metric]
            if int(row['total']) < int(row['mapped']):
                raise ValueError(f'Invalid coverage totals for {metric}')
            calculated = 100 * int(row['mapped']) / int(row['total']) if int(row['total']) else float('nan')
            if abs(float(row['coverage_pct']) - calculated) > 0.0001:
                raise RuntimeError(f'Inconsistent percentage for {metric}')
            print(f"  {metric}: {int(row['mapped']):,}/{int(row['total']):,} = {calculated:.4f}%")
    else:
        print('results/f_coverage.csv missing: rerun python 20_links.py')
    print('NOTE: Historical processed/link_events.parquet is ignored (not regenerated by 20_links.py).')
    print('WARNING: Coverage measures taxonomy lookups, not independently verified site identities.')
    print('WARNING: Parser artifacts/tenant-level ambiguities still require a data-quality audit.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', type=Path, default=Path('.'), help='Local repository root')
    ap.add_argument('--accepted', type=Path, help='Path to accepted 4,635-domain CSV')
    ap.add_argument('--audit', type=Path, help='Path to full 30-column audit CSV')
    ap.add_argument('--apply', action='store_true', help='Back up and update config/domain_categories.csv')
    ap.add_argument('--check', action='store_true', help='Report coverage after running 20_links.py')
    args = ap.parse_args()
    root = args.project.resolve()
    if args.check:
        run_check(root)
    else:
        accepted = args.accepted or root / 'results' / 'f_domain_work_4635_MERGED_ACCEPTED.csv'
        audit = args.audit or root / 'results' / 'f_domain_work_4635_MERGED_ACCEPTED_with_audit.csv'
        run_install(root, accepted, audit, args.apply)


if __name__ == '__main__':
    main()
