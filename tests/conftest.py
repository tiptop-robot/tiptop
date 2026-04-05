"""Shared test fixtures and configuration."""

import hashlib
import zipfile
from pathlib import Path

import gdown
import pytest

ASSETS_DIR = Path(__file__).parent / "assets"


def _download_and_extract_assets():
    """Download test assets from Google Drive, caching the zip for subsequent runs."""
    gdrive_url = "https://drive.google.com/uc?id=1IKKMzaEOsz7ydPhQKMHW5Wf3tU551iLH"
    expected_md5 = "9a0502d4c49daa1a06683bc66c4476cf"

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ASSETS_DIR / "tiptop_h5_assets.zip"

    # Use cached zip if hash matches, otherwise re-download
    cached_md5 = hashlib.md5(zip_path.read_bytes()).hexdigest() if zip_path.exists() else None
    need_download = cached_md5 != expected_md5
    if need_download:
        gdown.download(gdrive_url, str(zip_path), quiet=False)
        actual_md5 = hashlib.md5(zip_path.read_bytes()).hexdigest()
        assert actual_md5 == expected_md5, f"Downloaded zip MD5 mismatch: expected {expected_md5}, got {actual_md5}"

    # Only extract if we just downloaded or h5 files are missing
    if need_download or not any(ASSETS_DIR.glob("*.h5")):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(ASSETS_DIR)


@pytest.fixture(scope="session")
def h5_assets():
    """Ensure test H5 assets are downloaded and available. Returns the assets directory."""
    _download_and_extract_assets()
    h5_files = sorted(ASSETS_DIR.glob("*.h5"))
    assert len(h5_files) > 0, f"No H5 files found in {ASSETS_DIR} after download"
    return ASSETS_DIR
