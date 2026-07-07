import os
import re
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
YEAR = "2025"            # <- change this to grab a different season
YY = YEAR[2:]           # "25" -- used to build the KB<YY><NNNN> event code
MAX_WORKERS = 16        # parallel downloads. Lower to 3-4 if you want to be gentler.
TIMEOUT = 60            # seconds per request
MAX_RETRIES = 3         # attempts per file before giving up
RETRY_BACKOFF = 2       # seconds, multiplied by attempt number

download_dir = f"kmtnet_{YEAR}_lightcurves"
os.makedirs(download_dir, exist_ok=True)

base_url = f"https://kmtnet.kasi.re.kr/ulens/event/{YEAR}/"

# A single Session reuses the TCP connection across requests -> faster.
session = requests.Session()
session.headers.update({"User-Agent": "altREU-DISCORD-downloader/1.0 (research use)"})

# ----------------------------------------------------------------------
# 1. Get the event list from the year index page
# ----------------------------------------------------------------------
print(f"Fetching the {YEAR} event list from {base_url} ...")
try:
    resp = session.get(base_url, timeout=TIMEOUT)
    resp.raise_for_status()
except Exception as e:
    print(f"\nCould not reach the {YEAR} index page: {e}")
    print("The public release for this year may not exist yet. "
          "Open the base_url in a browser to check.")
    raise SystemExit(1)

soup = BeautifulSoup(resp.text, "html.parser")

# Pull the zero-padded event numbers out of links like
# ".../view.php?event=KMT-2025-BLG-0001"
event_numbers = set()
for a in soup.find_all("a", href=True):
    m = re.search(rf"event=KMT-{YEAR}-BLG-(\d+)", a["href"])
    if m:
        event_numbers.add(m.group(1))   # keep as the padded string, e.g. "0001"

event_numbers = sorted(event_numbers, key=int)
print(f"Found {len(event_numbers)} events.")

if not event_numbers:
    print(f"\nNo {YEAR} events found. The public data release for {YEAR} is "
          "probably not up yet (KMTNet releases each season ~mid the following year).")
    raise SystemExit(0)


# ----------------------------------------------------------------------
# 2. Download one event's DIA tarball
# ----------------------------------------------------------------------
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


def download_event(num):
    event_code = f"KB{YY}{num}"                       # e.g. KB250001
    filename = f"KMT-{YEAR}-BLG-{num}_diapl.tar.gz"
    filepath = os.path.join(download_dir, filename)

    if os.path.exists(filepath):
        return f"skip   {filename}"

    # Primary path: build the tarball URL directly (no page fetch needed).
    direct_url = urljoin(base_url, f"data/{event_code}/diapl/diapl.tar.gz")
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if stream_to_file(direct_url, filepath):
                return f"ok     {filename}" + (f" (attempt {attempt})" if attempt > 1 else "")
            break  # non-200 that isn't an exception -> fall through to page fallback
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    # Fallback: the direct URL 404'd (or kept erroring), so scrape the event page.
    event_url = urljoin(base_url, f"view.php?event=KMT-{YEAR}-BLG-{num}")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ev = session.get(event_url, timeout=TIMEOUT)
            ev.raise_for_status()
            ev_soup = BeautifulSoup(ev.text, "html.parser")
            for a in ev_soup.find_all("a", href=True):
                href = a["href"]
                if "/diapl/" in href and href.endswith(".tar.gz"):
                    if stream_to_file(urljoin(event_url, href), filepath):
                        return f"ok*    {filename} (via page)"
            return f"MISS   {filename} (no diapl tarball found)"
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    return f"err    {filename} (after {MAX_RETRIES} attempts): {last_err}"


# ----------------------------------------------------------------------
# 3. Run downloads in parallel
# ----------------------------------------------------------------------
print(f"Downloading with {MAX_WORKERS} parallel workers...\n")
done = 0
total = len(event_numbers)
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    futures = {ex.submit(download_event, n): n for n in event_numbers}
    for fut in as_completed(futures):
        done += 1
        print(f"[{done}/{total}] {fut.result()}")

print("\nAll DIA downloads complete!")