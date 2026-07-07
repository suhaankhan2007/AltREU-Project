import os
import tarfile
import requests

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
URL = "https://wwwmacho.anu.edu.au/Systems/Coords/lmc57charts.tar.gz"
TIMEOUT = 60
EXTRACT = True   # unpack the tarball after downloading (it's small, ~8 MB)

download_root = "lmc57_finder_charts"
os.makedirs(download_root, exist_ok=True)
archive_path = os.path.join(download_root, "lmc57charts.tar.gz")

session = requests.Session()
session.headers.update({"User-Agent": "altREU-DISCORD-downloader/1.0 (research use)"})


def stream_to_file(url, filepath):
    """Download url to filepath using a .part temp file + atomic rename,
    so an interrupted run never leaves a truncated file that looks complete."""
    r = session.get(url, stream=True, timeout=TIMEOUT)
    if r.status_code != 200:
        return False
    tmp = filepath + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):  # 64 KB chunks
            if chunk:
                f.write(chunk)
    os.replace(tmp, filepath)   # atomic on the same filesystem
    return True


def safe_extract(tar_path, dest_dir):
    """Extract while rejecting any member that would escape dest_dir
    (a classic tar-bomb / path-traversal guard)."""
    with tarfile.open(tar_path, "r:gz") as tar:
        dest_abs = os.path.abspath(dest_dir)
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(dest_dir, member.name))
            if not member_path.startswith(dest_abs + os.sep) and member_path != dest_abs:
                raise ValueError(f"Unsafe path in tarball, aborting extraction: {member.name}")
        tar.extractall(dest_dir)


# ----------------------------------------------------------------------
if os.path.exists(archive_path):
    print(f"Already downloaded: {archive_path}")
else:
    print(f"Downloading finder charts (LMC 5.7 Events, 0.63 sec/px, N up, W right) from {URL} ...")
    if stream_to_file(URL, archive_path):
        print(f"Saved to {archive_path}")
    else:
        print("Download failed (non-200 response).")
        raise SystemExit(1)

if EXTRACT:
    extract_dir = os.path.join(download_root, "charts")
    os.makedirs(extract_dir, exist_ok=True)
    print(f"Extracting to {extract_dir} ...")
    safe_extract(archive_path, extract_dir)
    print("Extraction complete.")
