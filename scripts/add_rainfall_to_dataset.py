"""
Merge downloaded rainfall cache into enhanced_training_data.csv.

Assigns each training sample the rainfall of its nearest ERA5 grid cell
(nearest-neighbour on the 0.25-deg grid).

Usage (run AFTER download_kerala_rainfall.py):
    python scripts/add_rainfall_to_dataset.py

Updates:
    data/enhanced_training_data.csv  (+2 columns: annual_rainfall_mm, extreme_rain_events)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from pathlib import Path

DATA_DIR    = Path('data')
DATASET_CSV = DATA_DIR / 'enhanced_training_data.csv'
CACHE_CSV   = DATA_DIR / 'kerala_rainfall_cache.csv'


def snap_to_era5(lat: float, lon: float) -> tuple:
    return (round(round(lat / 0.25) * 0.25, 4),
            round(round(lon / 0.25) * 0.25, 4))


def main():
    print('=' * 55)
    print('ADDING RAINFALL TO ENHANCED TRAINING DATA')
    print('=' * 55)

    if not CACHE_CSV.exists():
        print(f'❌  {CACHE_CSV} not found — run download_kerala_rainfall.py first')
        return

    df    = pd.read_csv(DATASET_CSV)
    cache = pd.read_csv(CACHE_CSV)
    print(f'\n✓  Dataset  : {len(df):,} rows, {len(df.columns)} features')
    print(f'✓  Cache    : {len(cache)} ERA5 grid cells')

    # Build lookup dict  (lat25, lon25) → rainfall values
    lookup = {}
    for _, row in cache.iterrows():
        key = (round(row['lat25'], 4), round(row['lon25'], 4))
        lookup[key] = {
            'annual_rainfall_mm':  row['annual_rainfall_mm'],
            'extreme_rain_events': int(row['extreme_rain_events']),
        }

    # Assign values per sample
    annual   = []
    extreme  = []
    miss     = 0

    for _, row in df.iterrows():
        key = snap_to_era5(row['latitude'], row['longitude'])
        if key in lookup:
            annual.append(lookup[key]['annual_rainfall_mm'])
            extreme.append(lookup[key]['extreme_rain_events'])
        else:
            # Shouldn't happen if download covered all cells
            annual.append(3000.0)
            extreme.append(12)
            miss += 1

    df['annual_rainfall_mm']  = annual
    df['extreme_rain_events'] = extreme
    df.to_csv(DATASET_CSV, index=False)

    if miss:
        print(f'⚠   {miss} samples used fallback values (missing grid cells)')

    print(f'\n{"="*55}')
    print('✅  DONE!')
    print(f'   Total features : {len(df.columns)}  (+2 rainfall columns)')
    print(f'   Annual rainfall: {df["annual_rainfall_mm"].min():.0f} – {df["annual_rainfall_mm"].max():.0f} mm')
    print(f'   Extreme events : {df["extreme_rain_events"].min()} – {df["extreme_rain_events"].max()}')
    print(f'   Saved          : {DATASET_CSV}')
    print(f'\n   Next: python ml/train_model.py')
    print(f'{"="*55}')


if __name__ == '__main__':
    main()
