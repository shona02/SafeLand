"""
Rebuild balanced_training_data.csv from scratch using both approaches:

Option A — Proper multi-year raster intersection:
  Load all 3 rasters at once, sum pixel-by-pixel to get true flood counts.
  Samples from each count tier (0/1/2/3) to build balanced classes.

Option B — Environmental label refinement:
  Upgrade "never flooded but high-risk environment" points from Low → Medium
  and "repeatedly flooded + high-risk environment" from Medium → High.
  Criteria use elevation, KSDMA zone, annual rainfall from enhanced_training_data.csv.

Output:
    data/balanced_training_data.csv   (rebuilt)
    Then re-run: python scripts/enrich_with_indian_sources.py (adds env features)
"""

import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
import random

DATA_DIR    = Path('data')
RASTERS     = {
    '2018': 'kerala_flood_2018_raster.tif',
    '2019': 'kerala_flood_2019_raster.tif',
    '2021': 'kerala_flood_2021_raster.tif',
}
OUTPUT_CSV  = DATA_DIR / 'balanced_training_data.csv'

# Target samples per risk class
TARGET = {'Low': 4000, 'Medium': 4000, 'High': 4000}

# ──────────────────────────────────────────────────────────────────
# OPTION A: Multi-year raster intersection
# ──────────────────────────────────────────────────────────────────

def load_and_align_rasters():
    """
    Load all 3 flood rasters and align them to a common grid.
    Returns (frequency_array, transform, crs) where frequency_array[r,c]
    is the number of flood years (0-3) at that pixel.
    """
    arrays = {}
    ref_transform = ref_shape = ref_crs = None

    for year, fname in RASTERS.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"❌  {path} not found!")
            return None, None, None
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.uint8)
            arr = (arr > 0).astype(np.uint8)   # binary: flooded / not
            arrays[year] = arr
            if ref_transform is None:
                ref_transform = src.transform
                ref_shape     = arr.shape
                ref_crs       = src.crs

    print(f"✓  Rasters loaded: shape {ref_shape}")

    # Stack and sum → pixel value = number of years flooded (0-3)
    stack     = np.stack(list(arrays.values()), axis=0)
    frequency = stack.sum(axis=0).astype(np.uint8)

    dist = {int(v): int((frequency == v).sum()) for v in [0,1,2,3]}
    print(f"   Pixel distribution: {dist}")
    return frequency, ref_transform, ref_crs


def sample_points(frequency, transform, count_value, n_samples, label, year_arrays):
    """Sample n_samples random pixels with frequency == count_value."""
    rows, cols = np.where(frequency == count_value)
    if len(rows) == 0:
        print(f"  ⚠  No pixels with count={count_value}")
        return []

    take = min(n_samples, len(rows))
    idx  = np.random.choice(len(rows), take, replace=False)
    s_rows, s_cols = rows[idx], cols[idx]
    xs, ys = rasterio.transform.xy(transform, s_rows, s_cols)

    years = list(RASTERS.keys())
    records = []
    for x, y, r, c in zip(xs, ys, s_rows, s_cols):
        rec = {
            'latitude':           float(y),
            'longitude':          float(x),
            'flood_history_count': int(count_value),
            'risk':               label,
        }
        for yr, arr in zip(years, year_arrays):
            rec[f'flooded_{yr}'] = int(arr[r, c])
        records.append(rec)
    return records


# ──────────────────────────────────────────────────────────────────
# OPTION B: Environmental label refinement
# ──────────────────────────────────────────────────────────────────

def refine_labels(df):
    """
    Upgrade risk labels using environmental features where available.
    Rules (applied in order, most aggressive last):

    Low → Medium:
      Never flooded but clearly high-risk environment:
        elevation < 10 m  AND  (ksdma_zone >= 4  OR  annual_rainfall_mm > 3500)

    Medium → High:
      Flooded at least once AND very high-risk environment:
        elevation < 15 m  AND  ksdma_zone >= 4  AND  annual_rainfall_mm > 3500

    Low → High (skipped — too aggressive without actual flood evidence)
    """
    env_cols = ['elevation', 'ksdma_zone', 'annual_rainfall_mm']
    if not all(c in df.columns for c in env_cols):
        print("  ⚠  Environmental columns not present — skipping refinement")
        print(f"     (Run enrich_with_indian_sources.py + add_rainfall_to_dataset.py first)")
        return df, 0, 0

    upgraded_low_to_med = 0
    upgraded_med_to_high = 0

    # Low → Medium
    mask_low_risky = (
        (df['risk'] == 'Low') &
        (df['elevation'] < 10) &
        ((df['ksdma_zone'] >= 4) | (df['annual_rainfall_mm'] > 3500))
    )
    df.loc[mask_low_risky, 'risk'] = 'Medium'
    upgraded_low_to_med = int(mask_low_risky.sum())

    # Medium → High
    mask_med_high = (
        (df['risk'] == 'Medium') &
        (df['flood_history_count'] >= 1) &
        (df['elevation'] < 15) &
        (df['ksdma_zone'] >= 4) &
        (df['annual_rainfall_mm'] > 3500)
    )
    df.loc[mask_med_high, 'risk'] = 'High'
    upgraded_med_to_high = int(mask_med_high.sum())

    return df, upgraded_low_to_med, upgraded_med_to_high


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    print('=' * 65)
    print('REBUILDING TRAINING DATA (Option A + B)')
    print('=' * 65)

    # ── A: Multi-year raster intersection ───────────────────────────
    print('\n[A] Multi-year raster intersection...')
    frequency, transform, crs = load_and_align_rasters()
    if frequency is None:
        return

    # Load raw arrays for per-year flooded flags
    year_arrays = []
    for fname in RASTERS.values():
        with rasterio.open(DATA_DIR / fname) as src:
            arr = src.read(1).astype(np.uint8)
            year_arrays.append((arr > 0).astype(np.uint8))

    np.random.seed(42)
    records = []

    # High risk: flooded in 2 or 3 years
    high_2 = sample_points(frequency, transform, 2, TARGET['High']//2, 'High', year_arrays)
    high_3 = sample_points(frequency, transform, 3, TARGET['High']//2, 'High', year_arrays)
    records.extend(high_2 + high_3)
    print(f'  High  : {len(high_2)} (2-yr) + {len(high_3)} (3-yr) = {len(high_2)+len(high_3)} samples')

    # Medium risk: flooded in exactly 1 year
    med = sample_points(frequency, transform, 1, TARGET['Medium'], 'Medium', year_arrays)
    records.extend(med)
    print(f'  Medium: {len(med)} samples')

    # Low risk: never flooded
    low = sample_points(frequency, transform, 0, TARGET['Low'], 'Low', year_arrays)
    records.extend(low)
    print(f'  Low   : {len(low)} samples')

    df = pd.DataFrame(records)
    # Ensure column order
    cols = ['latitude', 'longitude', 'flooded_2018', 'flooded_2019', 'flooded_2021',
            'flood_history_count', 'risk']
    df = df[cols]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f'\n  Before refinement:')
    print(f'  {df["risk"].value_counts().to_dict()}')

    # ── B: Environmental label refinement ───────────────────────────
    print('\n[B] Environmental label refinement...')
    enhanced_path = DATA_DIR / 'enhanced_training_data.csv'
    if enhanced_path.exists():
        enhanced = pd.read_csv(enhanced_path)
        env_lookup = {}
        for _, row in enhanced.iterrows():
            key = (round(row['latitude'], 4), round(row['longitude'], 4))
            env_lookup[key] = {
                'elevation':          row.get('elevation', 50),
                'ksdma_zone':         row.get('ksdma_zone', 3),
                'annual_rainfall_mm': row.get('annual_rainfall_mm', 3000),
            }
        # Attach env features to new df (nearest match rounded to 4dp)
        for col in ['elevation', 'ksdma_zone', 'annual_rainfall_mm']:
            df[col] = df.apply(
                lambda r: env_lookup.get((round(r['latitude'],4), round(r['longitude'],4)),
                                         {}).get(col, np.nan), axis=1
            )
        df, u_lm, u_mh = refine_labels(df)
        print(f'  Low → Medium upgrades  : {u_lm}')
        print(f'  Medium → High upgrades : {u_mh}')
        # Drop temp env columns (will be re-added by enrich script)
        df = df.drop(columns=['elevation', 'ksdma_zone', 'annual_rainfall_mm'], errors='ignore')
    else:
        print('  (No enhanced_training_data.csv found — skipping env refinement)')
        print('  Run enrich_with_indian_sources.py after this to add env features.')

    # ── Save ────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)

    print(f'\n{"="*65}')
    print('✅  DONE!')
    print(f'   Total samples : {len(df):,}')
    print(f'   Risk dist     : {df["risk"].value_counts().to_dict()}')
    print(f'   Saved         : {OUTPUT_CSV}')
    print(f'\n   Next steps:')
    print(f'   1. python scripts/enrich_with_indian_sources.py')
    print(f'   2. python scripts/add_rainfall_to_dataset.py')
    print(f'   3. python ml/train_model.py')
    print(f'{"="*65}')


if __name__ == '__main__':
    main()
