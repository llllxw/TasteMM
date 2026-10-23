"""Fetch a model archive in CI, validate checksum and extract only safe paths."""
import hashlib
import os
from pathlib import Path
import tarfile
import urllib.request

root = Path(__file__).resolve().parent
url, expected = os.environ['TASTEMM_MODEL_URL'], os.environ['TASTEMM_MODEL_SHA256']
if not url.startswith('https://'):
    raise SystemExit('Model URL must use HTTPS.')
archive = root / 'model_download.tar.gz'
urllib.request.urlretrieve(url, archive)
if hashlib.sha256(archive.read_bytes()).hexdigest().lower() != expected.lower():
    raise SystemExit('Model bundle checksum mismatch.')
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if root not in target.parents or member.issym() or member.islnk():
            raise SystemExit('Model archive contains an unsupported path.')
    tar.extractall(root)
if not (root / 'model_bundle/manifest.json').exists():
    raise SystemExit('Extracted model bundle is missing its manifest.')
archive.unlink()
