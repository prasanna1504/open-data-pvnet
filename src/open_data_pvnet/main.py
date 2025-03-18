import argparse
import logging
import calendar
from datetime import datetime
from open_data_pvnet.utils.env_loader import load_environment_variables
from open_data_pvnet.utils.data_downloader import (
    load_zarr_data,
    load_zarr_data_for_day,
    merge_hours_to_day,
    process_month_by_days,
    merge_days_to_month,
)
from pathlib import Path
import concurrent.futures
from typing import List, Tuple, Dict, Any
from open_data_pvnet.scripts.archive import handle_archive
from open_data_pvnet.nwp.met_office import CONFIG_PATHS
from open_data_pvnet.nwp.dwd import process_dwd_data

logger = logging.getLogger(__name__)

PROVIDERS = ["metoffice", "gfs", "dwd"]
DEFAULT_REGION = "global"  # Default region for Met Office datasets

# Provider-specific configurations
PROVIDER_CONFIGS = {
    "metoffice": {
        "regions": ["global", "uk"],
        "default_region": "global",
        "hours": range(24),
    },
    "dwd": {
        "regions": ["eu"],
        "default_region": "eu",
        "hours": range(24),
    },
    "gfs": {
        "regions": ["global"],
        "default_region": "global",
        "hours": range(0, 24, 3),  # GFS data is available every 3 hours
    },
}

def validate_date(year: int, month: int, day: int = None) -> None:
    """
    Validate date arguments.

    Args:
        year (int): Year value
        month (int): Month value (1-12)
        day (int, optional): Day value (1-31)

    Raises:
        ValueError: If any date component is invalid
    """
    current_year = datetime.now().year
    
    if not (1900 <= year <= current_year + 1):
        raise ValueError(f"Year must be between 1900 and {current_year + 1}")
    
    if not (1 <= month <= 12):
        raise ValueError("Month must be between 1 and 12")
    
    if day is not None:
        max_days = calendar.monthrange(year, month)[1]
        if not (1 <= day <= max_days):
            raise ValueError(f"Day must be between 1 and {max_days} for {calendar.month_name[month]} {year}")

def validate_hour(hour: int, provider: str) -> None:
    """
    Validate hour argument for a specific provider.

    Args:
        hour (int): Hour value
        provider (str): Provider name

    Raises:
        ValueError: If hour is invalid for the provider
    """
    valid_hours = PROVIDER_CONFIGS[provider]["hours"]
    if hour not in valid_hours:
        if len(valid_hours) == 24:
            raise ValueError("Hour must be between 0 and 23")
        else:
            valid_hours_str = ", ".join(map(str, valid_hours))
            raise ValueError(f"Hour must be one of: {valid_hours_str}")

def validate_region(region: str, provider: str) -> None:
    """
    Validate region argument for a specific provider.

    Args:
        region (str): Region value
        provider (str): Provider name

    Raises:
        ValueError: If region is invalid for the provider
    """
    valid_regions = PROVIDER_CONFIGS[provider]["regions"]
    if region not in valid_regions:
        raise ValueError(f"Region must be one of: {', '.join(valid_regions)}")

def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.

    Args:
        args (argparse.Namespace): Parsed command line arguments

    Raises:
        ValueError: If any argument is invalid
    """
    if args.command not in PROVIDERS:
        raise ValueError(f"Provider must be one of: {', '.join(PROVIDERS)}")

    validate_date(args.year, args.month, args.day)
    
    if hasattr(args, "hour") and args.hour is not None:
        validate_hour(args.hour, args.command)
    
    if hasattr(args, "region"):
        validate_region(args.region, args.command)

def load_env_and_setup_logger():
    """Initialize environment variables and configure logging.

    This function performs two main tasks:
    1. Loads environment variables from configuration files
    2. Sets up basic logging configuration with INFO level

    Raises:
        FileNotFoundError: If the environment configuration file cannot be found
    """
    try:
        load_environment_variables()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        logger.info("Environment variables loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Error loading environment variables: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during environment setup: {e}")
        raise

def _add_common_arguments(parser, provider_name):
    """Add arguments common to both archive and load operations."""
    parser.add_argument("--year", type=int, required=True, help="Year of data")
    parser.add_argument("--month", type=int, required=True, help="Month of data")
    parser.add_argument(
        "--day",
        type=int,
        help="Day of data (optional - if not provided, processes entire month)",
        default=None,
    )

    # Add provider-specific arguments
    if provider_name in PROVIDER_CONFIGS:
        config = PROVIDER_CONFIGS[provider_name]
        
        if len(config["hours"]) > 0:
            hour_help = (
                "Hour of data (0-23)"
                if len(config["hours"]) == 24
                else f"Hour of data (valid hours: {', '.join(map(str, config['hours']))})"
            )
            parser.add_argument(
                "--hour",
                type=int,
                help=f"{hour_help}. If not specified, process all valid hours.",
                default=None,
            )
        
        if len(config["regions"]) > 1:
            parser.add_argument(
                "--region",
                choices=config["regions"],
                default=config["default_region"],
                help=f"Specify the dataset region (default: {config['default_region']})",
            )

    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="Overwrite existing files in output directories",
    )

def parse_chunks(chunks_str: str) -> Dict[str, int]:
    """
    Parse chunks string into dictionary.

    Args:
        chunks_str (str): Chunk specification string (e.g., 'time:24,latitude:100')

    Returns:
        Dict[str, int]: Dictionary mapping dimensions to chunk sizes

    Raises:
        ValueError: If chunk string is malformed
    """
    if not chunks_str:
        return None
    
    try:
        chunks = {}
        for chunk in chunks_str.split(","):
            dim, size = chunk.split(":")
            chunks[dim.strip()] = int(size)
        return chunks
    except ValueError as e:
        raise ValueError(
            "Invalid chunk specification. Format should be 'dim1:size1,dim2:size2'"
        ) from e

def handle_load(provider: str, year: int, month: int, day: int, **kwargs) -> Any:
    """
    Handle loading archived data.

    Args:
        provider (str): Data provider name
        year (int): Year of data
        month (int): Month of data
        day (int): Day of data
        **kwargs: Additional arguments including chunks, remote, and hour

    Returns:
        Any: Loaded dataset

    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the arguments are invalid
    """
    try:
        # Validate arguments
        validate_date(year, month, day)
        if "hour" in kwargs and kwargs["hour"] is not None:
            validate_hour(kwargs["hour"], provider)

        chunks = parse_chunks(kwargs.get("chunks"))
        remote = kwargs.get("remote", False)
        hour = kwargs.get("hour")

        # Base path for the data
        base_path = Path("data") / str(year) / f"{month:02d}" / f"{day:02d}"

        if hour is not None:
            # Load specific hour
            archive_path = base_path / f"{year}-{month:02d}-{day:02d}-{hour:02d}.zarr.zip"
            logger.info(f"Loading dataset for {year}-{month:02d}-{day:02d} hour {hour:02d}")
            dataset = load_zarr_data(
                archive_path,
                chunks=chunks,
                remote=remote,
                download=not remote,
            )
            logger.info(f"Successfully loaded dataset for {year}-{month:02d}-{day:02d} hour {hour:02d}")
        else:
            # Load all hours for the day
            logger.info(f"Loading all datasets for {year}-{month:02d}-{day:02d}")
            dataset = load_zarr_data_for_day(
                base_path,
                year,
                month,
                day,
                chunks=chunks,
                remote=remote,
                download=not remote,
            )
            logger.info(f"Successfully loaded all datasets for {year}-{month:02d}-{day:02d}")

        return dataset

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Invalid argument: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def configure_parser():
    """
    Configure the main argument parser for the CLI tool.

    Returns:
        argparse.ArgumentParser: Configured argument parser

    The parser supports the following commands:
    - metoffice: Download and process Met Office data
    - gfs: Download and process GFS data
    - dwd: Download and process DWD data
    """
    parser = argparse.ArgumentParser(
        description="CLI tool for downloading and processing weather data from various providers."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create subparsers for each provider
    for provider in PROVIDERS:
        provider_parser = subparsers.add_parser(
            provider,
            help=f"Download and process {provider.upper()} data",
            description=f"Download and process weather data from {provider.upper()}",
        )
        
        # Add operation subparsers (archive/load)
        operation_subparsers = provider_parser.add_subparsers(
            dest="operation",
            help="Operation to perform",
            required=True
        )

        # Archive operation
        archive_parser = operation_subparsers.add_parser(
            "archive",
            help="Archive data for a specific time period",
            description="Download and archive weather data for a specific time period",
        )
        _add_common_arguments(archive_parser, provider)

        # Load operation
        load_parser = operation_subparsers.add_parser(
            "load",
            help="Load archived data",
            description="Load previously archived weather data",
        )
        _add_common_arguments(load_parser, provider)
        load_parser.add_argument(
            "--chunks",
            type=str,
            help="Chunk specification (e.g., 'time:24,latitude:100')",
            default=None,
        )
        load_parser.add_argument(
            "--remote",
            action="store_true",
            help="Load data from remote storage",
        )

    return parser


def chunk_hours(start: int = 0, end: int = 23, chunk_size: int = 6) -> List[Tuple[int, int]]:
    """Split hours into chunks."""
    chunks = []
    for i in range(start, end + 1, chunk_size):
        chunk_end = min(i + chunk_size - 1, end)
        chunks.append((i, chunk_end))
    return chunks


def archive_hours_chunk(
    provider: str,
    year: int,
    month: int,
    day: int,
    hour_range: Tuple[int, int],
    region: str,
    overwrite: bool,
    archive_type: str,
) -> None:
    """Archive a chunk of hours."""
    start_hour, end_hour = hour_range
    for hour in range(start_hour, end_hour + 1):
        archive_to_hf(
            provider=provider,
            year=year,
            month=month,
            day=day,
            hour=hour,
            region=region,
            overwrite=overwrite,
            archive_type=archive_type,
        )


def parallel_archive(
    provider: str,
    year: int,
    month: int,
    day: int,
    region: str,
    overwrite: bool,
    archive_type: str,
    max_workers: int = 4,
) -> None:
    """Archive data in parallel using multiple workers."""
    hour_chunks = chunk_hours()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                archive_hours_chunk,
                provider,
                year,
                month,
                day,
                chunk,
                region,
                overwrite,
                archive_type,
            )
            for chunk in hour_chunks
        ]
        concurrent.futures.wait(futures)

        # Check for exceptions
        for future in futures:
            if future.exception():
                raise future.exception()


def handle_monthly_consolidation(**kwargs):
    """Handle consolidating data into zarr.zip files."""
    logger.debug(f"Received kwargs: {kwargs}")
    chunks = parse_chunks(kwargs.get("chunks"))
    base_path = Path("data")
    year = kwargs.get("year")
    month = kwargs.get("month")
    day = kwargs.get("day")

    logger.debug(f"Extracted values - year: {year}, month: {month}")

    if year is None or month is None:
        raise ValueError("Year and month must be specified for consolidation")

    try:
        if day is not None:
            # Consolidate a single day
            logger.info(f"Consolidating day {year}-{month:02d}-{day:02d}")
            daily_file = merge_hours_to_day(base_path, year, month, day, chunks)
            logger.info(f"Successfully consolidated day to {daily_file}")
            return

        # First ensure all days are processed
        logger.info(f"Processing all days in month {year}-{month:02d}")
        successful_files = process_month_by_days(base_path, year, month, chunks)

        if successful_files:
            logger.info("\nSuccessfully created daily files:")
            for file in successful_files:
                logger.info(f"- {file}")

            # Now create the monthly file with safe_chunks=False
            logger.info("\nCreating monthly consolidated file")
            monthly_file = merge_days_to_month(base_path, year, month, chunks, safe_chunks=False)
            logger.info(f"Successfully created monthly file: {monthly_file}")
        else:
            logger.warning("No daily files were created, cannot create monthly file")

    except Exception as e:
        logger.error(f"Error in consolidation: {e}")
        raise


def handle_upload(provider: str, year: int, month: int, day: int = None, **kwargs):
    """Handle uploading data to Hugging Face."""
    config_path = Path("config.yaml")
    overwrite = kwargs.get("overwrite", False)
    upload_type = kwargs.get("type", "hourly")  # New parameter to specify upload type

    try:
        if upload_type == "monthly":
            # Upload monthly consolidated file
            logger.info(f"Uploading monthly consolidated file for {year}-{month:02d}")
            upload_monthly_zarr(
                config_path=config_path, year=year, month=month, overwrite=overwrite
            )
        else:
            # Original hourly upload functionality
            logger.info(f"Uploading hourly data for {year}-{month:02d}-{day:02d}")
            upload_to_huggingface(
                config_path=config_path,
                folder_name=f"{year}-{month:02d}-{day:02d}",
                year=year,
                month=month,
                day=day,
                overwrite=overwrite,
            )

    except Exception as e:
        logger.error(f"Error in upload: {e}")
        raise


def archive_to_hf(provider: str, year: int, month: int, day: int = None, **kwargs):
    """Handle archiving data."""
    overwrite = kwargs.get("overwrite", False)

    try:
        # Use provider-specific archive processing for daily data
        handle_archive(
            provider=provider,
            year=year,
            month=month,
            day=day,
            hour=kwargs.get("hour"),
            region=kwargs.get("region", "global"),
            overwrite=overwrite,
        )

    except Exception as e:
        logger.error(f"Error in archive: {e}")
        raise


def main():
    """
    Main entry point for the CLI tool.

    This function:
    1. Sets up logging and environment
    2. Parses command line arguments
    3. Validates arguments
    4. Executes the requested operation
    """
    try:
        # Initialize environment and logging
        load_env_and_setup_logger()
        
        # Parse and validate arguments
        parser = configure_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
            
        # Validate arguments
        validate_arguments(args)
        
        # Extract common arguments
        provider = args.command
        year = args.year
        month = args.month
        day = args.day
        hour = getattr(args, "hour", None)
        region = getattr(args, "region", PROVIDER_CONFIGS[provider]["default_region"])
        overwrite = getattr(args, "overwrite", False)
        
        logger.info(f"Processing {provider.upper()} data for {year}-{month:02d}" + 
                   (f"-{day:02d}" if day else "") +
                   (f" hour {hour:02d}" if hour is not None else "") +
                   f" (region: {region})")

        if args.operation == "archive":
            # Handle archiving operation
            if provider == "metoffice":
                config_path = CONFIG_PATHS[region]
                handle_archive(config_path, year, month, day, hour, overwrite)
            elif provider == "dwd":
                process_dwd_data(year, month, day, hour, overwrite)
            else:
                raise NotImplementedError(f"Archive operation not implemented for {provider}")
                
        elif args.operation == "load":
            # Handle loading operation
            chunks = getattr(args, "chunks", None)
            remote = getattr(args, "remote", False)
            
            dataset = handle_load(
                provider=provider,
                year=year,
                month=month,
                day=day,
                hour=hour,
                chunks=chunks,
                remote=remote,
            )
            
            logger.info("Dataset loaded successfully")
            return dataset
            
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return None
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return None
    except NotImplementedError as e:
        logger.error(f"Not implemented: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


if __name__ == "__main__":
    main()
