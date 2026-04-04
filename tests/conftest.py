"""Shared test fixtures and configuration."""

import hashlib
import zipfile
from pathlib import Path

import pytest

ASSETS_DIR = Path(__file__).parent / "assets"
GDRIVE_URL = "https://drive.google.com/uc?id=1IKKMzaEOsz7ydPhQKMHW5Wf3tU551iLH"
EXPECTED_MD5 = "9a0502d4c49daa1a06683bc66c4476cf"
ZIP_FILENAME = "tiptop_h5_assets.zip"

# Marker so unmarked tests don't accidentally require services
def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires external perception services")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _download_and_extract_assets():
    """Download test assets from Google Drive, caching the zip for subsequent runs."""
    import gdown

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ASSETS_DIR / ZIP_FILENAME

    # Use cached zip if hash matches, otherwise re-download
    need_download = not zip_path.exists() or _md5(zip_path) != EXPECTED_MD5
    if need_download:
        gdown.download(GDRIVE_URL, str(zip_path), quiet=False)
        actual_md5 = _md5(zip_path)
        assert actual_md5 == EXPECTED_MD5, (
            f"Downloaded zip MD5 mismatch: expected {EXPECTED_MD5}, got {actual_md5}"
        )

    # Only extract if we just downloaded or h5 files are missing
    if need_download or not any(ASSETS_DIR.glob("*.h5")):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(ASSETS_DIR)


@pytest.fixture(scope="session", autouse=False)
def h5_assets():
    """Ensure test H5 assets are downloaded and available. Returns the assets directory."""
    _download_and_extract_assets()
    h5_files = sorted(ASSETS_DIR.glob("*.h5"))
    assert len(h5_files) > 0, f"No H5 files found in {ASSETS_DIR} after download"
    return ASSETS_DIR
