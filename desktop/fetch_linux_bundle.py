"""Recover the unchanged model resources from the published Linux preview."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import urllib.request

root = Path(__file__).resolve().parent
url = 'https://github.com/llllxw/TasteMM/releases/download/desktop-v0.1.0-linux/TasteMM-Linux-x86_64.tar.gz'
expected = 'f2bc89592191200a3ce64084c40acb645eb125e7452fb0f1be3f7bab0161935e'
archive = root / 'model_download.tar.gz'
urllib.request.urlretrieve(url, archive)
digest = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
if digest.hexdigest() != expected:
    raise SystemExit('Published Linux archive checksum mismatch.')
with tarfile.open(archive) as tar:
    for member in tar:
        parts = PurePosixPath(member.name).parts
        if parts[:2] != ('TasteMM', 'model_bundle'):
            continue
        target = root.joinpath(*parts[1:]).resolve()
        if root not in target.parents or member.issym() or member.islnk():
            raise SystemExit('Unsupported model archive member.')
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output)
        else:
            raise SystemExit('Unsupported model archive member type.')
manifest = json.loads((root / 'model_bundle/manifest.json').read_text())
if manifest['checkpoint_sha256'] != '767c919782879488a258a2d942ada03766d25d0f13494ac49cce3ddb1f8d45da':
    raise SystemExit('Unexpected checkpoint identity.')
archive.unlink()
print(json.dumps({'linux_archive_sha256': expected, 'manifest': manifest}, indent=2))
