import pytest
from unittest.mock import Mock
from pathlib import Path
import xarray as xr

from open_data_pvnet.nwp.dwd import (
    generate_variable_url,
    fetch_dwd_data,
    process_dwd_data,
)
from tests.data.dwd.mock_responses import (
    HTML_DIRECTORY_LISTING,
    HTML_EMPTY_DIRECTORY,
    MOCK_CONFIG,
    create_mock_grib_dataset,
    MockResponse,
    MockHTTPError,
)


@pytest.fixture
def mock_config():
    return MOCK_CONFIG


def test_generate_variable_url():
    """Test the URL generation for DWD data."""
    url = generate_variable_url("T_2M", 2023, 1, 1, 0)
    assert url == "https://opendata.dwd.de/weather/nwp/icon-eu/grib/00/t_2m/icon-eu_europe_regular-lat-lon_single-level_2023010100_*"

    url = generate_variable_url("CLCT", 2023, 12, 31, 23)
    assert url == "https://opendata.dwd.de/weather/nwp/icon-eu/grib/23/clct/icon-eu_europe_regular-lat-lon_single-level_2023123123_*"


def test_generate_variable_url_invalid_hour():
    """Test URL generation with invalid hour."""
    with pytest.raises(ValueError, match="Hour must be between 0 and 23"):
        generate_variable_url("T_2M", 2023, 1, 1, 24)


def test_fetch_dwd_data_success(mocker, mock_config, tmp_path):
    """Test successful fetching of DWD data."""
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.dwd.PROJECT_BASE", str(tmp_path))
    mocker.patch("open_data_pvnet.nwp.dwd.CONFIG_PATH", "test_config.yaml")
    mocker.patch("open_data_pvnet.nwp.dwd.load_config", return_value=mock_config)

    # Mock requests
    mock_head = mocker.patch("requests.head")
    mock_head.return_value = MockResponse(status_code=200)

    mock_get = mocker.patch("requests.get")
    mock_get.return_value = MockResponse(content=HTML_DIRECTORY_LISTING)

    # Mock file operations
    mocker.patch("pathlib.Path.mkdir")
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("bz2.open")
    mocker.patch("os.remove")

    # Call function
    total_files = fetch_dwd_data(2023, 1, 1, 0)

    # Assertions
    assert total_files == 3
    assert mock_get.call_count == 6  # Three directory listings and three file downloads


def test_fetch_dwd_data_no_files(mocker, mock_config, tmp_path):
    """Test fetching DWD data when no files are available."""
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.dwd.PROJECT_BASE", str(tmp_path))
    mocker.patch("open_data_pvnet.nwp.dwd.CONFIG_PATH", "test_config.yaml")
    mocker.patch("open_data_pvnet.nwp.dwd.load_config", return_value=mock_config)

    # Mock empty HTML response
    mock_head = mocker.patch("requests.head")
    mock_head.return_value = MockResponse(status_code=404)

    # Call function
    total_files = fetch_dwd_data(2023, 1, 1, 0)

    # Assertions
    assert total_files == 0


def test_fetch_dwd_data_connection_error(mocker, mock_config, tmp_path):
    """Test handling of connection errors during data fetch."""
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.dwd.PROJECT_BASE", str(tmp_path))
    mocker.patch("open_data_pvnet.nwp.dwd.CONFIG_PATH", "test_config.yaml")
    mocker.patch("open_data_pvnet.nwp.dwd.load_config", return_value=mock_config)

    # Mock connection error
    mock_head = mocker.patch("requests.head")
    mock_head.side_effect = ConnectionError("Failed to connect to server")

    # Call function and check exception
    with pytest.raises(ConnectionError, match="Failed to connect to server"):
        fetch_dwd_data(2023, 1, 1, 0)


def test_process_dwd_data_success(mocker, mock_config, tmp_path):
    """Test successful processing of DWD data."""
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.dwd.PROJECT_BASE", str(tmp_path))
    mocker.patch("open_data_pvnet.nwp.dwd.CONFIG_PATH", "test_config.yaml")
    mocker.patch("open_data_pvnet.nwp.dwd.load_config", return_value=mock_config)

    # Create mock dataset
    mock_ds = create_mock_grib_dataset("t2m")
    mock_open_dataset = mocker.patch("xarray.open_dataset", return_value=mock_ds)

    # Create test file path and config
    test_file = tmp_path / "test.grib2"
    test_config = mock_config["grid"]

    # Call function
    result = process_dwd_data(str(test_file), test_config)

    # Assertions
    assert isinstance(result, xr.Dataset)
    assert "t2m" in result.data_vars
    mock_open_dataset.assert_called_once()


def test_process_dwd_data_invalid_dimensions(mocker, mock_config, tmp_path):
    """Test processing data with invalid dimensions."""
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.dwd.PROJECT_BASE", str(tmp_path))
    mocker.patch("open_data_pvnet.nwp.dwd.CONFIG_PATH", "test_config.yaml")
    mocker.patch("open_data_pvnet.nwp.dwd.load_config", return_value=mock_config)

    # Create mock dataset with wrong dimensions
    mock_ds = create_mock_grib_dataset("t2m", height=100, width=100)
    mocker.patch("xarray.open_dataset", return_value=mock_ds)

    # Create test file path and config
    test_file = tmp_path / "test.grib2"
    test_config = mock_config["grid"]

    # Call function and check exception
    with pytest.raises(ValueError, match="Dataset dimensions .* do not match expected dimensions"):
        process_dwd_data(str(test_file), test_config)


def test_process_dwd_data_no_files(mocker, mock_config):
    """Test processing when no files are downloaded."""
    with pytest.raises(FileNotFoundError):
        process_dwd_data("nonexistent_file.grib2", mock_config["grid"])


def test_process_dwd_data_corrupt_file(mocker, mock_config, tmp_path):
    """Test handling of corrupt GRIB files."""
    # Setup mocks
    mocker.patch("open_data_pvnet.nwp.dwd.PROJECT_BASE", str(tmp_path))
    mocker.patch("open_data_pvnet.nwp.dwd.CONFIG_PATH", "test_config.yaml")
    mocker.patch("open_data_pvnet.nwp.dwd.load_config", return_value=mock_config)

    # Mock xarray raising an error
    mock_open_dataset = mocker.patch("xarray.open_dataset")
    mock_open_dataset.side_effect = ValueError("Invalid GRIB file")

    # Create test file
    test_file = tmp_path / "test.grib2"
    test_file.touch()

    # Call function and check exception
    with pytest.raises(ValueError, match="Invalid GRIB file"):
        process_dwd_data(str(test_file), mock_config["grid"]) 