"""Mock responses for DWD tests."""
import numpy as np
import pandas as pd
import xarray as xr

# Mock HTML responses
HTML_DIRECTORY_LISTING = b"""
<html><body>
<a href="icon-eu_europe_regular-lat-lon_single-level_202301010000_000_T_2M.grib2.bz2">T_2M file</a>
<a href="icon-eu_europe_regular-lat-lon_single-level_202301010000_000_CLCT.grib2.bz2">CLCT file</a>
<a href="icon-eu_europe_regular-lat-lon_single-level_202301010000_000_ASWDIR_S.grib2.bz2">ASWDIR_S file</a>
</body></html>
"""

HTML_EMPTY_DIRECTORY = b"<html><body></body></html>"

# Mock GRIB data
def create_mock_grib_dataset(
    variable_name: str,
    height: int = 561,
    width: int = 1097,
    lat_bounds: tuple = (30.0, 65.0),
    lon_bounds: tuple = (-23.5, 45.0),
) -> xr.Dataset:
    """Create a mock GRIB dataset with the specified dimensions."""
    # Create coordinate arrays
    lats = np.linspace(lat_bounds[0], lat_bounds[1], height)
    lons = np.linspace(lon_bounds[0], lon_bounds[1], width)
    times = pd.date_range("2023-01-01", periods=1, freq="h")

    # Create mock data array
    data = np.random.rand(1, height, width)

    # Create dataset
    ds = xr.Dataset(
        {
            variable_name.lower(): xr.DataArray(
                data,
                coords={
                    "time": times,
                    "latitude": lats,
                    "longitude": lons,
                },
                dims=["time", "latitude", "longitude"],
            )
        }
    )

    return ds

# Mock configurations
MOCK_CONFIG = {
    "input_data": {
        "nwp": {
            "dwd": {
                "local_output_dir": "test_output",
                "nwp_channels": ["T_2M", "CLCT"],
                "nwp_accum_channels": ["ASWDIR_S"],
            }
        }
    },
    "grid": {
        "height": 561,
        "width": 1097,
        "latitude_bounds": [65.0, 30.0],
        "longitude_bounds": [-23.5, 45.0],
    },
}

# Mock file paths
MOCK_GRIB_FILE = "icon-eu_europe_regular-lat-lon_single-level_202301010000_000_T_2M.grib2"
MOCK_BZ2_FILE = f"{MOCK_GRIB_FILE}.bz2"

# Mock error responses
class MockHTTPError(Exception):
    """Mock HTTP error for testing."""
    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")

class MockResponse:
    """Mock response object for requests."""
    def __init__(self, content: bytes = b"", status_code: int = 200, raise_error: bool = False):
        self.content = content
        self.status_code = status_code
        self.text = content.decode("utf-8") if content else ""
        self.raise_error = raise_error

    def raise_for_status(self):
        """Raise an error if status code is not 200."""
        if self.raise_error or self.status_code != 200:
            raise MockHTTPError(self.status_code)

    def iter_content(self, chunk_size: int = None):
        """Mock iter_content method."""
        return [self.content] 