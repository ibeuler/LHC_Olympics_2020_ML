"""Download official LHC Olympics 2020 datasets from Zenodo."""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Zenodo URLs for official LHCO2020 datasets
DATASETS = {
    "rnd": {
        "filename": "events_LHCO2020_RnD.h5",
        "url": "https://zenodo.org/records/6466204/files/events_anomalydetection_v2.h5?download=1",
        "description": "R&D Dataset (1.1M events: 1M background + 100k signal)",
    },
    "background": {
        "filename": "events_LHCO2020_backgroundMC_Pythia.h5",
        "url": "https://zenodo.org/records/4536624/files/events_LHCO2020_backgroundMC_Pythia.h5?download=1",
        "description": "Background MC Pythia Dataset (1M background events)",
    },
    "blackbox1": {
        "filename": "events_LHCO2020_BlackBox1.h5",
        "url": "https://zenodo.org/records/4536624/files/events_LHCO2020_BlackBox1.h5?download=1",
        "description": "Black Box 1 Dataset (Unlabeled challenge data)",
    },
}


def _download_file(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req) as resp, open(temp_dest, "wb") as f:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB blocks
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = min(100.0, (downloaded / total_size) * 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(
                        f"\rProgress: {percent:5.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)"
                    )
                else:
                    mb_downloaded = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\rDownloaded: {mb_downloaded:.1f} MB")
                sys.stdout.flush()
        print()
        temp_dest.replace(dest)
        print(f"Successfully saved to {dest}")
    except Exception as e:
        if temp_dest.exists():
            temp_dest.unlink()
        print(f"\nFailed to download {url}: {e}")
        raise e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LHC Olympics 2020 dataset files.")
    parser.add_argument(
        "--dataset",
        choices=["rnd", "background", "blackbox1", "all"],
        default="rnd",
        help="Dataset to download: 'rnd' (default), 'background', 'blackbox1', or 'all'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where dataset files will be saved (default: data/raw)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dest_dir = args.output_dir

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    print("==================================================")
    print("LHC Olympics 2020 Data Downloader")
    print("==================================================")

    for key in keys:
        info = DATASETS[key]
        dest_file = dest_dir / info["filename"]

        if dest_file.exists():
            print(f"\nFile already exists: {dest_file} (Skipping)")
            continue

        print(f"\nDataset [{key}]: {info['description']}")
        _download_file(info["url"], dest_file)

    print("\nAll requested downloads complete.")


if __name__ == "__main__":
    main()
