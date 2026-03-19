import pandas as pd

df = pd.read_csv('data/enhanced_training_data.csv')
print(f"Total rows        : {len(df):,}")

exact = df[['latitude','longitude']].drop_duplicates()
print(f"Unique exact coords: {len(exact):,}")

# Open-Meteo archive API resolution is ~0.1 deg
grid_01 = df[['latitude','longitude']].round(1).drop_duplicates()
print(f"Unique 0.1-deg grid: {len(grid_01):,}  <- meaningful rainfall resolution")

# 0.25 deg grid (ERA5 native)
df['lat25'] = (df['latitude'] / 0.25).round() * 0.25
df['lon25'] = (df['longitude'] / 0.25).round() * 0.25
grid_25 = df[['lat25','lon25']].drop_duplicates()
print(f"Unique 0.25-deg ERA5 cells: {len(grid_25):,}  <- minimum API calls needed")
