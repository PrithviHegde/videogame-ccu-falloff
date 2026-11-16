"""Compress your existing verified Steam CSVs into GitHub-friendly Parquet files.

Place/extract this deployment kit inside your existing "Steam Project" folder:
  Steam Project/steam_cloud_deployment_kit/prepare_for_cloud.py
  Steam Project/processed/steam_games.csv  (etc.)

Run inside the deployment-kit folder:
  python -m pip install -r requirements.txt
  python prepare_for_cloud.py

Optional explicit location:
  python prepare_for_cloud.py --source "C:/path/to/Steam Project/processed"

Outputs ONLY into this kit's processed/ folder. Does not alter original CSVs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def convert_one(src: Path, dst: Path, mode: str) -> tuple[int, float]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit('Missing pyarrow. Run: python -m pip install -r requirements.txt') from exc

    if not src.is_file():
        raise SystemExit(f'Missing source file: {src}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + '.tmp')
    tmp.unlink(missing_ok=True)
    writer = None
    count = 0
    if mode == 'games':
        # This is only ~7,000 rows; preserve every metadata field unchanged.
        frames = [pd.read_csv(src, low_memory=False)]
    else:
        available = set(pd.read_csv(src, nrows=0).columns)
        columns = (
            ['appid', 'year_month', 'avg_ccu', 'peak_ccu', 'source']
            if mode == 'monthly' else
            ['appid', 'day_utc', 'daily_mean_ccu', 'daily_peak_ccu',
             'sample_coverage_ratio', 'observation_count']
        )
        required = {'appid','year_month','avg_ccu','peak_ccu'} if mode == 'monthly' else {
            'appid','day_utc','daily_mean_ccu','daily_peak_ccu','sample_coverage_ratio'}
        if absent := required - available:
            raise SystemExit(f'{src.name} missing required columns: {sorted(absent)}')
        columns = [c for c in columns if c in available]
        frames = pd.read_csv(src, usecols=columns, chunksize=120_000, low_memory=False)

    try:
        for frame in frames:
            if mode != 'games':
                frame['appid'] = pd.to_numeric(frame['appid'], errors='coerce').astype('Int64')
                if mode == 'monthly':
                    for col in ('year_month', 'source'):
                        if col in frame:
                            frame[col] = frame[col].astype('string')
                    numeric_cols = ('avg_ccu', 'peak_ccu')
                else:
                    frame['day_utc'] = frame['day_utc'].astype('string')
                    if 'observation_count' in frame:
                        frame['observation_count'] = pd.to_numeric(
                            frame['observation_count'], errors='coerce').astype('Int64')
                    numeric_cols = ('daily_mean_ccu', 'daily_peak_ccu', 'sample_coverage_ratio')
                for col in numeric_cols:
                    frame[col] = pd.to_numeric(frame[col], errors='coerce').astype('float64')
            arrow = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(tmp), arrow.schema, compression='zstd', use_dictionary=True)
            writer.write_table(arrow)
            count += len(frame)
            if mode == 'daily' and count % 600_000 == 0:
                print(f'  daily: {count:,} rows converted...')
    finally:
        if writer is not None:
            writer.close()

    if not tmp.is_file():
        raise SystemExit(f'No records were written for {src.name}')
    actual = pq.ParquetFile(str(tmp)).metadata.num_rows
    if actual != count:
        raise SystemExit(f'Row-count mismatch for {mode}: input {count}, parquet {actual}')
    size_mib = tmp.stat().st_size / 1024**2
    if size_mib >= 100:
        print(f'WARNING: {dst.name} is {size_mib:.1f} MiB (GitHub ordinary Git blocks >100 MiB).')
    elif size_mib >= 25:
        print(f'NOTICE: {dst.name} is {size_mib:.1f} MiB: upload via Git/GitHub Desktop, not browser.')
    tmp.replace(dst)
    return count, size_mib


def main() -> None:
    kit = Path(__file__).resolve().parent
    default_source = kit.parent / 'processed'
    p = argparse.ArgumentParser(description='Build compressed Steam data for Community Cloud')
    p.add_argument('--source', type=Path, default=default_source,
                   help='Directory containing the three verified .csv files')
    args = p.parse_args()
    src_dir = args.source.expanduser().resolve()
    dst_dir = kit / 'processed'
    if src_dir == dst_dir.resolve():
        raise SystemExit('Source and destination must differ: use the original processed CSV folder.')
    print('Source:', src_dir)
    print('Deploy data:', dst_dir)
    print('Your original CSVs will not be changed.\n')
    for stem, mode in [('steam_games','games'),('steam_monthly','monthly'),('steam_daily','daily')]:
        print(f'Converting {stem}.csv ...')
        rows, size = convert_one(src_dir / (stem + '.csv'), dst_dir / (stem + '.parquet'), mode)
        print(f'  OK: {rows:,} rows, {size:.1f} MiB\n')
    print('SUCCESS: all three Parquet files are ready.')
    print('Next: upload ONLY this deployment-kit folder contents to your GitHub repository.')


if __name__ == '__main__':
    main()
