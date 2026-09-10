"""Apply the column configuration to existing local Parquet files without downloading."""

from datetime import datetime, timezone
import json

import polars as pl

from fantasy_football.config import PROJECT_ROOT
from fantasy_football.data.columns import load_column_selection, select_columns
from fantasy_football.data.storage import save_parquet, validate_download


def main() -> None:
    """Preflight every local table, then save narrower copies in the same locations."""
    selections = load_column_selection()
    pending = []
    for name in selections:
        for path in sorted((PROJECT_ROOT / "data" / "raw" / name).glob("*.parquet")):
            original = pl.read_parquet(path)
            season = None if path.stem == "latest" else int(path.stem)
            validate_download(original, season)
            # Validate all selections before modifying any file. To restore a
            # removed column later, download that dataset again with --refresh.
            selected = select_columns(original, name, selections)
            pending.append((path, original.columns, selected))

    results = []
    for path, before, selected in pending:
        if before != selected.columns:
            save_parquet(selected, path)
        results.append({
            "path": str(path.relative_to(PROJECT_ROOT)), "rows": selected.height,
            "columns_before": len(before), "columns_after": selected.width,
            "removed_columns": [c for c in before if c not in selected.columns],
            "kept_columns": selected.columns,
        })
        print(f"{path.parent.name}/{path.name}: {len(before)} -> {selected.width} columns")

    now = datetime.now(timezone.utc)
    folder = PROJECT_ROOT / "data" / "manifests"
    folder.mkdir(parents=True, exist_ok=True)
    report = folder / f"columns_{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    report.write_text(json.dumps({
        "operation": "local_column_selection", "processed_at_utc": now.isoformat(),
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Local selection report: {report}")


if __name__ == "__main__":
    main()
