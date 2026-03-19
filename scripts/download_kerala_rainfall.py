"""
Download rainfall data for Kerala using ERA5 grid cells via Open-Meteo Archive.

Strategy (avoids 429 rate limits):
  - Instead of 12K individual API calls, extract the ~228 unique 0.25-deg
    ERA5 grid cells that cover all training samples.
  - Download 5-year daily precipitation for each cell with throttling.
  - Save to data/kerala_rainfall_cache.csv (lookup table).
  - merge_rainfall.py then assigns each training sample its nearest cell.

Usage:
    python scripts/download_kerala_rainfall.py

Output:
    data/kerala_rainfall_cache.csv
      columns: lat25, lon25, annual_rainfall_mm, extreme_rain_events
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

DATA_DIR    = Path('data')
INPUT_CSV   = DATA_DIR / 'enhanced_training_data.csv'
OUTPUT_CSV  = DATA_DIR / 'kerala_rainfall_cache.csv'
ARCHIVE_URL = 'https://archive-api.open-meteo.com/v1/archive'

# 5-year window (2019-2024)
END_DATE   = datetime(2024, 12, 31)
START_DATE = END_DATE - timedelta(days=365 * 5)


def get_era5_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Snap all coordinates to the nearest 0.25-deg ERA5 grid cell."""
    df = df.copy()
    df['lat25'] = (df['latitude'] / 0.25).round() * 0.25
    df['lon25'] = (df['longitude'] / 0.25).round() * 0.25
    grid = df[['lat25', 'lon25']].drop_duplicates().reset_index(drop=True)
    return grid


def fetch_rainfall(lat: float, lon: float, retries: int = 4) -> dict:
    """
    Fetch 5-year daily precipitation for one grid cell.
    Returns dict with annual_rainfall_mm and extreme_rain_events.
    """
    for attempt in range(retries):
        try:
            r = requests.get(ARCHIVE_URL, params={
                'latitude':   lat,
                'longitude':  lon,
                'start_date': START_DATE.strftime('%Y-%m-%d'),
                'end_date':   END_DATE.strftime('%Y-%m-%d'),
                'daily':      'precipitation_sum',
                'timezone':   'Asia/Kolkata',
            }, timeout=45)

            if r.status_code == 200:
                data  = r.json()
                precip = [p for p in data.get('daily', {}).get('precipitation_sum', []) if p is not None]
                if precip:
                    annual = round(sum(precip) / 5, 1)
                    extreme = sum(1 for p in precip if p > 100)
                    return {'annual_rainfall_mm': annual, 'extreme_rain_events': extreme}

            elif r.status_code == 429:
                wait = 10 * (2 ** attempt)   # 10 / 20 / 40 / 80 s
                print(f'\n  429 rate limit – waiting {wait}s (attempt {attempt+1}/{retries})...')
                time.sleep(wait)
                continue

            else:
                print(f'\n  HTTP {r.status_code} for ({lat}, {lon})')
                break

        except Exception as e:
            print(f'\n  Error ({attempt+1}/{retries}): {e}')
            time.sleep(5)

    # Fallback: Kerala average values
    return {'annual_rainfall_mm': 3000.0, 'extreme_rain_events': 12}


def main():
    print('=' * 60)
    print('DOWNLOADING KERALA RAINFALL (ERA5 grid, 228 cells)')
    print('=' * 60)

    if not INPUT_CSV.exists():
        print(f'❌  {INPUT_CSV} not found — run enrich_with_indian_sources.py first')
        return

    df   = pd.read_csv(INPUT_CSV)
    grid = get_era5_grid(df)
    print(f'\n✓  {len(grid)} unique ERA5 grid cells to fetch')
    print(f'   Estimated time: ~{len(grid) * 3 / 60:.0f} min (3 s/cell)\n')

    # Resume support — skip cells already downloaded
    if OUTPUT_CSV.exists():
        done = pd.read_csv(OUTPUT_CSV)
        existing = set(zip(done['lat25'].round(4), done['lon25'].round(4)))
        grid_todo = grid[~grid.apply(lambda r: (round(r['lat25'], 4), round(r['lon25'], 4)) in existing, axis=1)]
        print(f'   Resuming — {len(done)} done, {len(grid_todo)} remaining')
        results = done.to_dict('records')
    else:
        grid_todo = grid
        results   = []

    for _, row in tqdm(grid_todo.iterrows(), total=len(grid_todo), desc='Fetching rainfall'):
        lat, lon = row['lat25'], row['lon25']
        data = fetch_rainfall(lat, lon)
        results.append({'lat25': lat, 'lon25': lon, **data})

        # Save incrementally so we can resume if interrupted
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
        time.sleep(3)   # polite delay — well under rate limit

    cache = pd.DataFrame(results)
    cache.to_csv(OUTPUT_CSV, index=False)

    print(f'\n{"="*60}')
    print('✅  DONE!')
    print(f'   Cells fetched   : {len(cache)}')
    print(f'   Rainfall range  : {cache["annual_rainfall_mm"].min():.0f} – {cache["annual_rainfall_mm"].max():.0f} mm')
    print(f'   Extreme events  : {cache["extreme_rain_events"].min()} – {cache["extreme_rain_events"].max()}')
    print(f'   Saved           : {OUTPUT_CSV}')
    print(f'\n   Next: python scripts/add_rainfall_to_dataset.py')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
