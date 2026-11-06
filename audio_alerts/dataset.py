"""Download a pinned, four-class subset of ESC-50 for local research use."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

ESC_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
BASE = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{ESC_COMMIT}"
CLASSES = ("chainsaw", "hand_saw", "engine", "helicopter")


def selected_rows(csv_text: str) -> list[dict[str, str]]:
    rows = [row for row in csv.DictReader(csv_text.splitlines()) if row["category"] in CLASSES]
    if len(rows) != 160 or {row["category"] for row in rows} != set(CLASSES):
        raise ValueError("Pinned ESC-50 metadata did not contain 40 clips per selected class")
    for row in rows:
        name = row["filename"]
        if Path(name).name != name or not name.endswith(".wav") or not 1 <= int(row["fold"]) <= 5:
            raise ValueError(f"Invalid dataset row: {row}")
    return rows


def download(destination: Path, workers: int = 8) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{BASE}/meta/esc50.csv", timeout=30) as response:
        csv_text = response.read().decode("utf-8")
    rows = selected_rows(csv_text)
    audio_dir = destination / "audio"
    audio_dir.mkdir(exist_ok=True)

    def fetch(row: dict[str, str]) -> None:
        path = audio_dir / row["filename"]
        if path.exists() and path.stat().st_size > 1000:
            return
        with urlopen(f"{BASE}/audio/{row['filename']}", timeout=45) as response:
            content = response.read()
        if not content.startswith(b"RIFF") or len(content) < 1000:
            raise ValueError(f"Invalid WAV download: {row['filename']}")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, row) for row in rows]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 40 == 0 or i == len(rows):
                print(f"Downloaded/verified {i}/{len(rows)} clips")
    with (destination / "selected.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"ESC-50 subset ready at {destination.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data/esc50"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    download(args.destination, args.workers)


if __name__ == "__main__":
    main()
