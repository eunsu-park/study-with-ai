#!/usr/bin/env python3
"""Download academic paper PDFs via Unpaywall, ADS, Semantic Scholar, and fallback strategies.

Usage:
    python scripts/pdf_download.py <doi_or_url> <output_path> [--email EMAIL]

Examples:
    python scripts/pdf_download.py "10.1038/323533a0" path/to/paper.pdf
    python scripts/pdf_download.py "https://arxiv.org/pdf/1301.3781" path/to/paper.pdf
"""

import argparse
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_EMAIL = "test@test.com"
TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _make_request(url: str, timeout: int = TIMEOUT) -> bytes | None:
    """Fetch URL content with browser-like headers and SSL fallback.

    Args:
        url: URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Response bytes, or None on failure.
    """
    headers = {"User-Agent": USER_AGENT}
    req = urllib.request.Request(url, headers=headers)

    # Try with default SSL first, then unverified as fallback
    for ctx in (None, ssl.create_default_context()):
        try:
            if ctx is None:
                resp = urllib.request.urlopen(req, timeout=timeout)
            else:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
            return resp.read()
        except (urllib.error.URLError, urllib.error.HTTPError, ssl.SSLError,
                TimeoutError, ConnectionError):
            continue
    return None


def _is_pdf(data: bytes) -> bool:
    """Check if data starts with PDF magic bytes."""
    return data[:5] == b"%PDF-"


def download_via_unpaywall(doi: str, email: str = DEFAULT_EMAIL) -> bytes | None:
    """Try to download PDF via Unpaywall API.

    Args:
        doi: DOI string (e.g., "10.1038/323533a0").
        email: Email for Unpaywall API.

    Returns:
        PDF bytes if successful, None otherwise.
    """
    api_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    # Try best OA location first
    pdf_urls = []
    best_oa = data.get("best_oa_location")
    if best_oa:
        if best_oa.get("url_for_pdf"):
            pdf_urls.append(best_oa["url_for_pdf"])
        if best_oa.get("url"):
            pdf_urls.append(best_oa["url"])

    # Try all OA locations
    for loc in data.get("oa_locations", []):
        if loc.get("url_for_pdf"):
            pdf_urls.append(loc["url_for_pdf"])

    for url in pdf_urls:
        result = _make_request(url)
        if result and _is_pdf(result):
            return result

    return None


def download_via_direct_url(url: str) -> bytes | None:
    """Download PDF from a direct URL.

    Args:
        url: Direct URL to a PDF file.

    Returns:
        PDF bytes if successful and valid PDF, None otherwise.
    """
    result = _make_request(url)
    if result and _is_pdf(result):
        return result
    return None


def download_via_arxiv(identifier: str) -> bytes | None:
    """Try to download from arXiv using various ID formats.

    Args:
        identifier: arXiv ID, DOI containing arXiv, or title keywords.

    Returns:
        PDF bytes if found, None otherwise.
    """
    # Extract arXiv ID if present
    arxiv_match = re.search(r"(\d{4}\.\d{4,5})", identifier)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        result = _make_request(url)
        if result and _is_pdf(result):
            return result
    return None


def download_via_ads(doi: str) -> bytes | None:
    """Try to download scanned PDF from NASA ADS via DOI → bibcode resolution.

    ADS hosts scanned PDFs of many older journal articles. This strategy
    resolves a DOI to an ADS bibcode, then fetches the scanned PDF.

    Args:
        doi: DOI string (e.g., "10.1007/BF00145821").

    Returns:
        PDF bytes if successful, None otherwise.
    """
    headers = {"User-Agent": USER_AGENT}

    # Step 1: Resolve DOI to ADS bibcode via redirect
    resolve_url = f"https://ui.adsabs.harvard.edu/abs/doi:{doi}"
    req = urllib.request.Request(resolve_url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        match = re.search(r"/abs/([^/]+)", resp.url)
        if not match:
            return None
        bibcode = match.group(1)
    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, ConnectionError):
        return None

    # Step 2: Fetch scanned PDF from ADS articles server
    pdf_url = f"https://articles.adsabs.harvard.edu/pdf/{bibcode}"
    result = _make_request(pdf_url)
    if result and _is_pdf(result):
        return result
    return None


def download_via_semantic_scholar(doi: str) -> bytes | None:
    """Try to download PDF via Semantic Scholar API.

    Args:
        doi: DOI string.

    Returns:
        PDF bytes if an open-access PDF is available, None otherwise.
    """
    api_url = (
        f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        f"?fields=openAccessPdf"
    )
    headers = {"User-Agent": USER_AGENT}
    req = urllib.request.Request(api_url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT)
        data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    oa_pdf = data.get("openAccessPdf")
    if not oa_pdf:
        return None
    pdf_url = oa_pdf.get("url", "")
    if not pdf_url or oa_pdf.get("status") == "CLOSED":
        return None

    result = _make_request(pdf_url)
    if result and _is_pdf(result):
        return result
    return None


def download_pdf(doi_or_url: str, output_path: Path,
                 email: str = DEFAULT_EMAIL) -> dict:
    """Attempt to download a PDF using multiple strategies.

    Args:
        doi_or_url: DOI string or direct URL.
        output_path: Where to save the PDF.
        email: Email for Unpaywall API.

    Returns:
        Result dict with keys: ok, method, path (or error).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine if input is a URL or DOI
    is_url = doi_or_url.startswith("http://") or doi_or_url.startswith("https://")
    is_doi = re.match(r"^10\.\d{4,}/", doi_or_url) is not None

    strategies: list[tuple[str, callable]] = []

    if is_url:
        strategies.append(("direct_url", lambda: download_via_direct_url(doi_or_url)))
        # Also try to extract arXiv ID from URL
        strategies.append(("arxiv", lambda: download_via_arxiv(doi_or_url)))
    elif is_doi:
        strategies.append(("unpaywall", lambda: download_via_unpaywall(doi_or_url, email)))
        strategies.append(("arxiv", lambda: download_via_arxiv(doi_or_url)))
        strategies.append(("semantic_scholar", lambda: download_via_semantic_scholar(doi_or_url)))
        strategies.append(("ads", lambda: download_via_ads(doi_or_url)))
    else:
        # Might be an arXiv ID
        strategies.append(("arxiv", lambda: download_via_arxiv(doi_or_url)))

    for method_name, method_fn in strategies:
        try:
            result = method_fn()
            if result:
                output_path.write_bytes(result)
                return {
                    "ok": True,
                    "method": method_name,
                    "path": str(output_path),
                    "size_bytes": len(result),
                }
        except Exception as e:
            continue

    return {
        "ok": False,
        "error": f"All download strategies failed for: {doi_or_url}",
        "tried": [s[0] for s in strategies],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download academic paper PDFs."
    )
    parser.add_argument("doi_or_url", help="DOI (e.g., 10.1038/323533a0) or direct URL")
    parser.add_argument("output_path", help="Where to save the PDF")
    parser.add_argument("--email", default=DEFAULT_EMAIL,
                        help=f"Email for Unpaywall API (default: {DEFAULT_EMAIL})")
    args = parser.parse_args()

    result = download_pdf(args.doi_or_url, Path(args.output_path), args.email)
    print(json.dumps(result, ensure_ascii=False))

    if not result["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
