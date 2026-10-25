"""Download the challenge data package from the public Google Drive folder.

Walks the folder tree through Drive's public `embeddedfolderview` page (which,
unlike the gdown folder scraper, is not capped at 50 entries) and downloads
every file by ID, mirroring the folder structure under --out.

Usage:
    python scripts/download_data.py --out data/raw
"""
import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT_FOLDER = "1iVJiEl_iaqGXXJgZZ5InoLjqJdt1Sonq"
ENTRY_RE = re.compile(
    r'<div class="flip-entry" id="entry-([A-Za-z0-9_-]+)".*?'
    r'<a href="([^"]+)".*?<div class="flip-entry-title">([^<]+)</div>',
    re.S,
)
SKIP = {"__MACOSX", ".DS_Store"}
# The Drive root also holds Experiment-2.zip / Experiment-3.zip, which bundle
# the source authors' MATLAB code, results and LSTM notebook. Those are not
# allowed, so only the unpacked folders are downloaded.
SKIP_ROOT_SUFFIXES = (".zip",)

session = requests.Session()


def list_folder(folder_id):
    html = session.get(
        "https://drive.google.com/embeddedfolderview", params={"id": folder_id}, timeout=60
    ).text
    for fid, href, title in ENTRY_RE.findall(html):
        yield fid, "/folders/" in href, title.strip()


def walk(folder_id, rel=Path()):
    for fid, is_dir, title in list_folder(folder_id):
        if title in SKIP or title.startswith("._") or title.endswith(SKIP_ROOT_SUFFIXES):
            continue
        if is_dir:
            yield from walk(fid, rel / title)
        else:
            yield fid, rel / title


def download(fid, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return dest, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = "https://drive.usercontent.google.com/download"
    params = {"id": fid, "export": "download", "confirm": "t"}
    with session.get(url, params=params, stream=True, timeout=300) as r:
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            raise RuntimeError(f"got HTML instead of file for {dest}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(dest)
    return dest, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)

    files = list(walk(ROOT_FOLDER))
    print(f"found {len(files)} files")
    failed = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(download, fid, out / rel): rel for fid, rel in files}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:  # keep going; report at the end
                failed.append((futs[fut], e))
            if i % 100 == 0:
                print(f"  {i}/{len(files)}")
    for rel, e in failed:
        print("FAILED", rel, e)
    print("done" if not failed else f"{len(failed)} failures -- re-run to retry (cached files are skipped)")


if __name__ == "__main__":
    main()
