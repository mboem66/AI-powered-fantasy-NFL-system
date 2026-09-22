#One command to refresh sources, build player games, and generate features.
import argparse
import json

from fantasy_football.config import PROJECT_ROOT, load_settings
from fantasy_football.data import download
from fantasy_football.data.build_player_games import build as build_games
from fantasy_football.features.build_features import build as build_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", help="Rebuild from saved data without downloading")
    args = parser.parse_args(argv)
    download_status = 0
    if not args.local:
        download_status = download.main(["--mode", "all"])
        manifest = max((PROJECT_ROOT / "data/manifests").glob("download_*.json"))
        results = json.loads(manifest.read_text(encoding="utf-8"))["results"]
        required_failures = [r for r in results if r["status"] == "failed" and r["dataset"] != "snap_counts"]
        if required_failures:
            print("Required downloads failed; processed files were not rebuilt. See", manifest)
            return 1
        if download_status:
            print("Warning: some snap downloads failed. Saved snaps may be stale or missing; continuing.")
    else:
        print("Using local source files; this does not check for newer results.")
    settings = load_settings()
    build_games(PROJECT_ROOT, settings)
    build_features(PROJECT_ROOT, settings)
    print("Update finished." if not download_status else "Build finished with download warnings (exit code 1).")
    return download_status


if __name__ == "__main__":
    raise SystemExit(main())
