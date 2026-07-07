import os
import csv
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
# All six of these ANU MACHO pages are simple static HTML tables where
# every event/star has a "[Data]" link to one data file, so one generic
# scraper handles all of them -- each just gets its own output folder.
#
# NOTE: the "binaries" entry (astro-ph/9907369) links out to
# darkstar.astro.washington.edu, a University of Washington host that no
# longer resolves (confirmed dead as of 2026-07-04, and Wayback Machine
# has no earlier working snapshot either). That data file appears to be
# lost; the script will report it as a MISS rather than silently skip it.
CATALOGS = [
    {
        "label": "LMC Eclipsing Cepheids (astro-ph/0201481)",
        "page_url": "https://wwwmacho.anu.edu.au/Data/EclCep/ecl_cep.htm",
        "out_dir": "lmc_eclipsing_cepheids",
        "extensions": (".txt",),
    },
    {
        "label": "Binary Microlensing Events (astro-ph/9907369)",
        "page_url": "https://wwwmacho.anu.edu.au/Data/binaries.html",
        "out_dir": "binary_microlensing_events",
        "extensions": (".tar.gz",),
    },
    {
        "label": "LMC Microlensing Events (ApJ 486 697, 1997)",
        "page_url": "https://wwwmacho.anu.edu.au/Data/fts.html",
        "out_dir": "lmc_microlensing_events",
        "extensions": (".publc",),
    },
    {
        "label": "SMC Microlensing Event (ApJ 491 L11, 1997)",
        "page_url": "https://wwwmacho.anu.edu.au/Data/smcfts.html",
        "out_dir": "smc_microlensing_events",
        "extensions": (".publc",),
    },
    {
        "label": "Bulge Microlensing Events (ApJ 479 119, 1997)",
        "page_url": "https://wwwmacho.anu.edu.au/Data/bulgefts.html",
        "out_dir": "bulge_microlensing_events",
        "extensions": (".publc",),
    },
    {
        "label": "LMC Beat RR Lyrae (ApJ 482 89, 1996)",
        "page_url": "https://wwwmacho.anu.edu.au/Data/beatrrfts.html",
        "out_dir": "lmc_beat_rr_lyrae",
        "extensions": (".publc",),
    },
]

TIMEOUT = 60
MAX_RETRIES = 3
RETRY_BACKOFF = 2

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


def download_with_retries(url, filepath):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return stream_to_file(url, filepath), last_err
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    return False, last_err


# ----------------------------------------------------------------------
# Scrape one catalog page: find every <a href> that ends in one of the
# configured extensions, record the other cell values in its table row
# (if any) as metadata, and download the linked file.
# ----------------------------------------------------------------------
def process_catalog(catalog):
    label = catalog["label"]
    page_url = catalog["page_url"]
    out_dir = catalog["out_dir"]
    extensions = catalog["extensions"]

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {label} ===")
    print(f"Fetching {page_url} ...")
    try:
        resp = session.get(page_url, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"  could not fetch page: {e}")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    links = [a for a in soup.find_all("a", href=True) if a["href"].lower().endswith(extensions)]
    print(f"  found {len(links)} data link(s)")

    rows_meta = []
    for a in links:
        href = a["href"]
        url = urljoin(page_url, href)
        filename = os.path.basename(href)
        filepath = os.path.join(out_dir, filename)

        # Any other <td> text in this link's table row is useful metadata
        # (MACHO ID, coordinates, event name, ...) -- capture it alongside
        # the file since this HTML table is the only place it's recorded.
        tr = a.find_parent("tr")
        row_values = []
        if tr:
            for td in tr.find_all("td"):
                if td.find("a"):
                    continue
                text = td.get_text(strip=True)
                if text:
                    row_values.append(text)
        rows_meta.append([filename, url] + row_values)

        if os.path.exists(filepath):
            print(f"  skip {filename}")
            continue
        ok, err = download_with_retries(url, filepath)
        if ok:
            print(f"  ok   {filename}")
        else:
            print(f"  MISS {filename}" + (f" ({err})" if err else " (non-200 response)"))

    if rows_meta:
        csv_path = os.path.join(out_dir, "events.csv")
        max_cols = max(len(r) for r in rows_meta)
        header = ["filename", "url"] + [f"col{i}" for i in range(1, max_cols - 1)]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows_meta)
        print(f"  wrote metadata for {len(rows_meta)} row(s) to {csv_path}")


# ----------------------------------------------------------------------
for catalog in CATALOGS:
    process_catalog(catalog)

print("\nAll catalog page downloads complete!")
