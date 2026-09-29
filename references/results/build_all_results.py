"""Concatenate the per-paper result CSVs into references/results/all_results.csv and validate them."""
from __future__ import annotations
import argparse
import csv
import pathlib
import sys

COLUMNS = ['source_paper', 'source_arxiv', 'source_table', 'source_page', 'table_kind', 'method',
           'base_model', 'dataset', 'scene', 'psnr_db', 'ssim', 'lpips', 'size_mb',
           'size_unit_printed', 'num_gaussians', 'fps', 'train_time_s', 'encode_time_s',
           'decode_time_s', 'total_time_s', 'notes']
DATASETS = {'Mip-NeRF 360', 'Tanks and Temples', 'Deep Blending', 'NeRF Synthetic', 'BungeeNeRF',
            'N3DV', 'OMMO', 'DL3DV-GS', 'Other'}
KINDS = {'main', 'ablation', 'supplement'}
UNITS = {'MB', 'MiB', 'GB', 'KB', ''}

#---------------------------------------------------------------------
# validation
#---------------------------------------------------------------------


def check_row(row: dict, where: str) -> list[str]:
    """Return the list of validation problems in one row."""
    problems = []
    if row['table_kind'] not in KINDS:
        problems.append(f'{where}: table_kind {row["table_kind"]!r}')
    if row['dataset'] not in DATASETS:
        problems.append(f'{where}: dataset {row["dataset"]!r}')
    if row['size_unit_printed'] not in UNITS:
        problems.append(f'{where}: size_unit_printed {row["size_unit_printed"]!r}')
    if bool(row['size_mb']) != bool(row['size_unit_printed']):
        problems.append(f'{where}: size_mb and size_unit_printed disagree')
    for col in ('psnr_db', 'ssim', 'lpips', 'size_mb', 'num_gaussians', 'fps',
                'train_time_s', 'encode_time_s', 'decode_time_s', 'total_time_s'):
        if row[col] == '':
            continue
        try:
            float(row[col])
        except ValueError:
            problems.append(f'{where}: {col} {row[col]!r} is not numeric')
    if row['psnr_db'] and not 5 <= float(row['psnr_db']) <= 45:
        problems.append(f'{where}: psnr_db {row["psnr_db"]} out of range')
    if row['ssim'] and not 0 <= float(row['ssim']) <= 1:
        problems.append(f'{where}: ssim {row["ssim"]} out of range')
    if row['lpips'] and not 0 <= float(row['lpips']) <= 1:
        problems.append(f'{where}: lpips {row["lpips"]} out of range')
    return problems


def build_all_results(results_dir: str = 'references/results') -> None:
    """Merge every per-paper CSV under results_dir into all_results.csv, after validating it."""
    root = pathlib.Path(results_dir)
    paths = sorted(p for p in root.glob('*/*.csv') if p.parent.name != 'special')
    merged, problems = [], []
    for path in paths:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            problems.append(f'{path}: empty')
            continue
        if list(rows[0].keys()) != COLUMNS:
            problems.append(f'{path}: column mismatch {list(rows[0].keys())}')
            continue
        for ii, row in enumerate(rows, start=2):
            problems.extend(check_row(row, f'{path.name}:{ii}'))
        merged.extend(rows)
    out_path = root / 'all_results.csv'
    with out_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(merged)
    print(f'{len(paths)} files, {len(merged)} rows -> {out_path}')
    if problems:
        print(f'{len(problems)} validation problems:', file=sys.stderr)
        for problem in problems[:40]:
            print('  ' + problem, file=sys.stderr)
        sys.exit(1)
    print('validation passed')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_dir', default='references/results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build_all_results(**vars(args))
