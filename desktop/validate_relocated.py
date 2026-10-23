"""Extract the actual release and verify it with no build Python on PATH."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('--archive', required=True, type=Path)
parser.add_argument('--destination', required=True, type=Path)
parser.add_argument('--report', required=True, type=Path)
args = parser.parse_args()
archive = args.archive.resolve()
destination = args.destination.resolve()
if destination.exists():
    raise SystemExit('Choose a new extraction directory.')
destination.mkdir(parents=True)
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        target = (destination / name).resolve()
        if destination not in target.parents:
            raise SystemExit('Invalid archive path.')
    z.extractall(destination)
environment = os.environ.copy()
for key in ['PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'TCL_LIBRARY', 'TK_LIBRARY']:
    environment.pop(key, None)
environment.update(PATH=str(Path(os.environ['SystemRoot']) / 'System32'),
                   HF_HOME=str(destination / 'empty-huggingface-cache'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='',
                   HTTP_PROXY='http://127.0.0.1:9', HTTPS_PROXY='http://127.0.0.1:9',
                   ALL_PROXY='http://127.0.0.1:9')
binary = destination / 'TasteMM/TasteMM.exe'
reports = {}
for flag, name in [('--self-test', 'inference'), ('--self-test-ui', 'drawing'),
                   ('--self-test-workflow', 'workflow')]:
    report = destination / (name + '.json')
    subprocess.run([str(binary), flag, '--report', str(report)], cwd=destination,
                   env=environment, check=True, timeout=240)
    reports[name] = json.loads(report.read_text(encoding='utf-8'))
    assert reports[name]['passed']
with archive.open('rb') as stream:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
args.report.write_text(json.dumps({'passed': True, 'archive_sha256': digest.hexdigest(),
    'relocated_path': str(destination), 'python_and_conda_removed_from_path': True,
    'empty_huggingface_cache': True, 'hub_offline_flags': True, 'external_http_proxy_unreachable': True,
    'note': 'Offline flags and unreachable proxy; this is not an OS firewall audit.',
    'checks': reports}, indent=2), encoding='utf-8')
