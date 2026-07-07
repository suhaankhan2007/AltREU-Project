import os
import re
import time
import requests
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
YEARS = ["2022", "2023", "2024", "2025", "2026"]   # <- edit to grab other seasons
MAX_WORKERS = 16        # parallel downloads. Lower to 3-4 if you want to be gentler.
TIMEOUT = 60            # seconds per request
MAX_RETRIES = 3         # attempts per file before giving up
RETRY_BACKOFF = 2       # seconds, multiplied by attempt number

BASE = "https://ftp.astrouw.edu.pl/ogle/ogle4/ews/"
download_root = "ogle_ews_lightcurves"
os.makedirs(download_root, exist_ok=True)

# A single Session reuses the TCP connection across requests -> faster.
session = requests.Session()
session.headers.update({"User-Agent": "altREU-DISCORD-downloader/1.0 (research use)"})


def stream_to_file(url, filepath):
    """Download url to filepath using a .part temp file + atomic rename,
    so an interrupted run never leaves a truncated file that looks complete."""
    r = session.get(url, timeout=TIMEOUT)
    if r.status_code != 200:
        return False
    tmp = filepath + ".part"
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, filepath)   # atomic on the same filesystem
    return True


# ----------------------------------------------------------------------
# 1. Get the event list for a year from the single lenses.par summary file
#    (one request per year, instead of scraping the HTML index or hitting
#    every event directory just to see what's there).
# ----------------------------------------------------------------------
def get_event_list(year):
    url = urljoin(BASE, f"{year}/lenses.par")
    resp = session.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    events = []
    for line in resp.text.splitlines():
        m = re.match(rf"\s*{year}-BLG-(\d+)", line)
        if m:
            events.append(m.group(1))  # zero-padded number, e.g. "0001"
    return events


# ----------------------------------------------------------------------
# 2. Download one event's params.dat + phot.dat
#    (skips the .tar.gz on purpose -- it bundles finder charts and
#    postscript plots that aren't needed for light-curve training and
#    are 5-10x bigger than the two data files combined).
# ----------------------------------------------------------------------
def download_event(year, num):
    event_dir = os.path.join(download_root, year, f"blg-{num}")
    os.makedirs(event_dir, exist_ok=True)
    results = []
    for fname in ("phot.dat", "params.dat"):
        filepath = os.path.join(event_dir, fname)
        if os.path.exists(filepath):
            results.append(f"skip {year}/blg-{num}/{fname}")
            continue
        url = urljoin(BASE, f"{year}/blg-{num}/{fname}")
        last_err = None
        ok = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                ok = stream_to_file(url, filepath)
                break
            except Exception as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF * attempt)
        if ok:
            results.append(f"ok   {year}/blg-{num}/{fname}")
        else:
            results.append(f"MISS {year}/blg-{num}/{fname}" + (f" ({last_err})" if last_err else ""))
    return " | ".join(results)


# ----------------------------------------------------------------------
# 3. Build the full job list across years, then run in parallel
# ----------------------------------------------------------------------
all_jobs = []
for year in YEARS:
    print(f"Fetching {year} event list from lenses.par ...")
    try:
        events = get_event_list(year)
    except Exception as e:
        print(f"  could not fetch {year} event list: {e}")
        continue
    print(f"  found {len(events)} events for {year}")
    all_jobs.extend((year, num) for num in events)

print(f"\nDownloading phot.dat + params.dat for {len(all_jobs)} events "
      f"with {MAX_WORKERS} parallel workers...\n")

done = 0
total = len(all_jobs)
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(download_event, year, num): (year, num) for year, num in all_jobs}
    for fut in as_completed(futures):
        done += 1
        print(f"[{done}/{total}] {fut.result()}")

print("\nAll OGLE EWS downloads complete!")
