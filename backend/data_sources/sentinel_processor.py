"""
Sentinel-1 SAR flood history processor for SafeLand.

Reads directly from the pre-processed flood raster files (.tif) generated
during training, ensuring inference matches training-time feature values.
"""

import os
import numpy as np
from typing import Dict
from backend.config import Config
from backend.cache import cache_result

# ── Raster paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR  = os.path.join(BASE_DIR, "data")

RASTER_PATHS = {
    2018: os.path.join(DATA_DIR, "kerala_flood_2018_raster.tif"),
    2019: os.path.join(DATA_DIR, "kerala_flood_2019_raster.tif"),
    2021: os.path.join(DATA_DIR, "kerala_flood_2021_raster.tif"),
}

# How many pixels to search around the exact coordinate (catches point-on-boundary misses)
SEARCH_RADIUS_PX = 3


class SentinelProcessor:
    """Process flood history from pre-built Kerala flood rasters."""

    def __init__(self):
        self._rasters: dict = {}   # year → (data ndarray, transform)
        self._load_rasters()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_rasters(self):
        """Load each annual flood raster into memory (they are small ~1 MB)."""
        try:
            import rasterio
            for year, path in RASTER_PATHS.items():
                if os.path.exists(path):
                    with rasterio.open(path) as src:
                        self._rasters[year] = (src.read(1), src.transform)
                    print(f"✓ Sentinel raster {year} loaded ({path})")
                else:
                    print(f"⚠ Flood raster not found: {path}")
        except ImportError:
            print("⚠ rasterio not installed — flood history will use fallback heuristic")
        except Exception as e:
            print(f"Error loading flood rasters: {e}")

    def _is_flooded(self, year: int, lat: float, lon: float) -> bool:
        """
        Return True if the location was flooded in the given year.

        Searches a SEARCH_RADIUS_PX neighbourhood to handle sub-pixel misses.
        """
        if year not in self._rasters:
            return False

        try:
            from rasterio.transform import rowcol
            data, transform = self._rasters[year]

            row, col = rowcol(transform, lon, lat)

            r0 = max(0, row - SEARCH_RADIUS_PX)
            r1 = min(data.shape[0] - 1, row + SEARCH_RADIUS_PX)
            c0 = max(0, col - SEARCH_RADIUS_PX)
            c1 = min(data.shape[1] - 1, col + SEARCH_RADIUS_PX)

            if r0 > r1 or c0 > c1:
                return False   # Out of bounds

            window = data[r0:r1 + 1, c0:c1 + 1]
            return bool(np.any(window == 1))

        except Exception as e:
            print(f"Error reading raster {year} at ({lat},{lon}): {e}")
            return False

    # ── Public API ────────────────────────────────────────────────────────────

    @cache_result(expiry_hours=720)   # elevation-like data; cache 30 days
    def get_flood_events_detail(self, lat: float, lon: float) -> Dict:
        """
        Return per-year flood flags and aggregate count for a location.

        Falls back to elevation-based heuristic when rasters are unavailable.
        """
        if self._rasters:
            f18 = self._is_flooded(2018, lat, lon)
            f19 = self._is_flooded(2019, lat, lon)
            f21 = self._is_flooded(2021, lat, lon)
            flood_count = int(f18) + int(f19) + int(f21)
        else:
            # Fallback heuristic (no rasters loaded)
            from backend.data_sources.bhuvan_api import bhuvan_api
            from backend.data_sources.ksdma_zones import ksdma_zones
            elevation = bhuvan_api.get_elevation(lat, lon)
            zone      = ksdma_zones.get_vulnerability_zone(lat, lon)

            if elevation < 10 and zone >= 4:
                flood_count = 3
            elif elevation < 30 and zone >= 3:
                flood_count = 1
            else:
                flood_count = 0

            f18 = flood_count >= 1
            f19 = flood_count >= 2
            f21 = flood_count >= 3

        return {
            "total_events":   flood_count,
            "flooded_2018":   f18,
            "flooded_2019":   f19,
            "flooded_2021":   f21,
            "risk_category":  "High" if flood_count >= 2 else "Medium" if flood_count == 1 else "Low",
        }

    @cache_result(expiry_hours=720)
    def get_flood_history(self, lat: float, lon: float) -> int:
        """Return 1 if the location has any flood history, 0 otherwise."""
        return 1 if self.get_flood_events_detail(lat, lon)["total_events"] > 0 else 0

    @cache_result(expiry_hours=720)
    def get_flood_frequency(self, lat: float, lon: float) -> int:
        """Return number of flood events (0-3) at the location."""
        return self.get_flood_events_detail(lat, lon)["total_events"]


# Singleton instance
sentinel_processor = SentinelProcessor()
