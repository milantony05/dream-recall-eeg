"""Fetch version-pinned public DREAM metadata and Tononi archives, with MD5 checks."""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

CATALOGUE_ID = 22133105
CATALOGUE_VERSION = 9
TONONI_ID = 23306054
TONONI_VERSION = 3


def article(article_id: int, version: int) -> dict:
    with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{article_id}/versions/{version}", timeout=45) as response:
        result = json.load(response)
    if result["id"] != article_id or result["version"] != version:
        raise ValueError("Figshare article/version mismatch")
    return result


def fetch_file(info: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = info.get("computed_md5") or info["supplied_md5"]
    if destination.exists() and md5(destination) == expected:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(4):
        digest = hashlib.md5()
        try:
            with urllib.request.urlopen(info["download_url"], timeout=120) as src, temporary.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != expected or temporary.stat().st_size != info["size"]:
                raise ValueError(f"Download integrity failure: {destination.name}")
            temporary.replace(destination)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            # Public Figshare asset redirects can return a transient 403; retry only
            # this known public endpoint, never a request-only dataset.
            if attempt == 3 or (isinstance(exc, urllib.error.HTTPError) and exc.code not in (403, 429, 500, 502, 503, 504)):
                raise
            pause = 2 ** (attempt + 1)
            print(f"Transient download problem ({type(exc).__name__}); retrying in {pause}s", flush=True)
            time.sleep(pause)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--tononi-subject", type=int, action="append", default=[], help="1–39; repeat to fetch several")
    parser.add_argument("--all-tononi", action="store_true", help="Fetch all ~7.83 GB, including subject archives")
    args = parser.parse_args()
    root = args.root.resolve()
    for aid, version in [(CATALOGUE_ID, CATALOGUE_VERSION), (TONONI_ID, TONONI_VERSION)]:
        metadata = article(aid, version)
        folder = root / ("catalogue-v9" if aid == CATALOGUE_ID else "tononi-v3")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "source-article.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        for item in metadata["files"]:
            name = item["name"]
            if aid == CATALOGUE_ID or "No_PSGs" in name or args.all_tononi or any(f"-s{n:02d}_PSGs" in name for n in args.tononi_subject):
                print(f"Fetching/checking {name}", flush=True)
                fetch_file(item, folder / name)
    print(f"Pinned public files saved under {root}; raw files remain outside git.")


if __name__ == "__main__":
    main()
