"""Validate the packaged arm64 app after relocation on native macOS."""
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile

if sys.platform != 'darwin' or platform.machine() != 'arm64':
    raise SystemExit('Run this validation on native Apple Silicon macOS.')
root = Path(__file__).resolve().parent
out = root / 'releases'
archive = out / 'TasteMM-macOS-arm64.zip'
digest = hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
checksum = digest.hexdigest()
assert checksum == archive.with_suffix('.zip.sha256').read_text().split()[0]
reports = {}
with tempfile.TemporaryDirectory(prefix='TasteMM-macOS-') as temp:
    destination = Path(temp) / '中文路径 relocation'
    destination.mkdir()
    subprocess.run(['ditto', '-x', '-k', str(archive), str(destination)], check=True, timeout=120)
    bundle = destination / 'TasteMM.app'
    binary = bundle / 'Contents/MacOS/TasteMM'
    architecture = subprocess.check_output(['lipo', '-archs', str(binary)], text=True).strip()
    assert architecture == 'arm64', architecture
    subprocess.run(['codesign', '--verify', '--deep', '--strict', str(bundle)], check=True, timeout=120)
    with (bundle / 'Contents/Info.plist').open('rb') as stream:
        info = plistlib.load(stream)
    assert info['CFBundleShortVersionString'] == '0.1.0', info
    assert info['CFBundleVersion'] == '0.1.0', info
    environment = os.environ.copy()
    for key in ['PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'TCL_LIBRARY', 'TK_LIBRARY', 'DYLD_LIBRARY_PATH', 'DYLD_FRAMEWORK_PATH']:
        environment.pop(key, None)
    environment.update(PATH='/usr/bin:/bin:/usr/sbin:/sbin',
                       HF_HOME=str(destination / 'empty-huggingface-cache'),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       CUDA_VISIBLE_DEVICES='', HTTP_PROXY='http://127.0.0.1:9',
                       HTTPS_PROXY='http://127.0.0.1:9', ALL_PROXY='http://127.0.0.1:9')
    for flag, name in [('--self-test', 'relocated-inference'),
                       ('--self-test-ui', 'relocated-drawing'),
                       ('--self-test-workflow', 'relocated-workflow')]:
        report = out / f'validation-{name}-macOS-arm64.json'
        subprocess.run([str(binary), flag, '--report', str(report)],
                       cwd=destination, env=environment, check=True, timeout=240)
        reports[name] = json.loads(report.read_text())
        assert reports[name]['passed']
    sandbox = Path('/usr/bin/sandbox-exec')
    if not sandbox.exists():
        raise SystemExit('sandbox-exec is required for the network-denial check.')
    profile = '(version 1) (allow default) (deny network*)'
    network_report = out / 'validation-network-denied-macOS-arm64.json'
    subprocess.run([str(sandbox), '-p', profile, str(binary), '--self-test', '--report', str(network_report)],
                   cwd=destination, env=environment, check=True, timeout=180)
    reports['network-denied-inference'] = json.loads(network_report.read_text())
    assert reports['network-denied-inference']['passed']
    # LaunchServices is how a user opens the .app, rather than its inner binary.
    opened_report = out / 'validation-launchservices-macOS-arm64.json'
    subprocess.run(['/usr/bin/open', '-W', '-n', str(bundle), '--args', '--self-test', '--report', str(opened_report)],
                   cwd=destination, env=environment, check=True, timeout=180)
    reports['launchservices-inference'] = json.loads(opened_report.read_text())
    assert reports['launchservices-inference']['passed']
    summary = {'passed': True, 'platform': platform.platform(), 'architecture': architecture,
               'archive_sha256': checksum, 'archive_bytes': archive.stat().st_size,
               'relocated_unicode_path': True, 'no_external_python_on_path': True,
               'empty_model_cache': True, 'network_denied_by_macos_sandbox': True,
               'code_signature_deep_strict_verified': True,
               'bundle_identifier': info.get('CFBundleIdentifier'),
               'app_version': info['CFBundleShortVersionString'],
               'minimum_system_version_metadata': info.get('LSMinimumSystemVersion'),
               'signing': 'ad-hoc', 'notarized': False,
               'gatekeeper_downloaded_app_tested': False,
               'automated_widget_tests': True, 'manual_visual_acceptance': False,
               'checks': reports}
(out / 'validation-archive-macOS-arm64.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
