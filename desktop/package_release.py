"""Validate a native application and archive its executable and offline model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--validate', action='store_true')
args = parser.parse_args()
out = root / 'releases'
out.mkdir(exist_ok=True)
system = {'linux': 'Linux', 'win32': 'Windows', 'darwin': 'macOS'}[sys.platform]
architecture = platform.machine().lower()
distribution = root / 'dist' / ('TasteMM.app' if sys.platform == 'darwin' else 'TasteMM')
binary = distribution / ('Contents/MacOS/TasteMM' if sys.platform == 'darwin' else 'TasteMM.exe' if sys.platform == 'win32' else 'TasteMM')
if args.validate:
    report = out / f'validation-{system}-{architecture}.json'
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    subprocess.run([str(binary), '--self-test', '--report', str(report)], check=True, env=env, timeout=180)
    if not json.loads(report.read_text())['passed']:
        raise SystemExit('Frozen inference validation failed.')
    if sys.platform != 'linux' or env.get('DISPLAY'):
        ui_report = out / f'validation-ui-{system}-{architecture}.json'
        subprocess.run([str(binary), '--self-test-ui', '--report', str(ui_report)], check=True, env=env, timeout=180)
        if not json.loads(ui_report.read_text())['passed']:
            raise SystemExit('Frozen structure-rendering validation failed.')
        workflow_report = out / f'validation-workflow-{system}-{architecture}.json'
        subprocess.run([str(binary), '--self-test-workflow', '--report', str(workflow_report)], check=True, env=env, timeout=240)
        if not json.loads(workflow_report.read_text())['passed']:
            raise SystemExit('Frozen desktop workflow validation failed.')
name = f'TasteMM-{system}-{architecture}'
if sys.platform == 'linux':
    archive = out / (name + '.tar.gz')
    with tarfile.open(archive, 'w:gz', compresslevel=5) as tar:
        tar.add(distribution, arcname='TasteMM')
elif sys.platform == 'darwin':
    archive = out / (name + '.zip')
    subprocess.run(['ditto', '-c', '-k', '--sequesterRsrc', '--keepParent', str(distribution), str(archive)], check=True)
else:
    archive = out / (name + '.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for file in distribution.rglob('*'):
            if file.is_file():
                z.write(file, Path('TasteMM') / file.relative_to(distribution))
checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
archive.with_suffix(archive.suffix + '.sha256').write_text(checksum + '  ' + archive.name + '\n')
print(str(archive))
