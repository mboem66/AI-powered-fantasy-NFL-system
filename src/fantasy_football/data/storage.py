"""Validate downloads and safely replace local Parquet files."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl


def validate_download(frame: pl.DataFrame, season: int | None) -> None:
    """Reject empty tables and accidental downloads of a different season."""
    if frame.is_empty():
        raise ValueError("Source returned an empty table; existing file was preserved")
    if season is not None:
        if "season" not in frame.columns:
            raise ValueError("Season dataset is missing its season column")
        values = frame.get_column("season")
        if values.null_count() or values.cast(pl.Int64).unique().to_list() != [season]:
            raise ValueError(f"Downloaded rows do not all belong to season {season}")


def save_parquet(frame: pl.DataFrame, path: Path) -> None:
    """Write beside the destination first, so failed writes preserve prior data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        frame.write_parquet(temporary, compression="zstd")
        # Confirm the new file is readable before replacing the previous copy.
        if pl.scan_parquet(temporary).select(pl.len()).collect().item() != frame.height:
            raise ValueError("Saved row count does not match the downloaded table")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
