"""Stage auditable provenance, documentation, examples and installed notices."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import subprocess
from datetime import datetime, timezone
root = Path(__file__).resolve().parent
mac = sys.platform == 'darwin'
distribution = root / 'dist' / ('TasteMM.app' if mac else 'TasteMM')
target = distribution / 'Contents/Resources' if mac else distribution
binary = distribution / 'Contents/MacOS/TasteMM' if mac else distribution / 'TasteMM.exe'
if not binary.exists():
    raise SystemExit('Build the native executable first.')

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()
files = {str(p.relative_to(root)).replace('\\', '/'): sha256(p) for p in sorted((root / 'model_bundle').rglob('*')) if p.is_file()}
assert files['model_bundle/model.pth'] == '767c919782879488a258a2d942ada03766d25d0f13494ac49cce3ddb1f8d45da'
provenance = {'release': ('macOS Apple Silicon' if mac else 'Windows x64') + ' preview; port of desktop-v0.1.0-linux', 'built_at_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(), 'machine': platform.machine(), 'python': sys.version, 'desktop_source_base_sha256': 'fc14643a9e06a98b8abcd82985a7a2f6fd0b144b38568932202bccdb5801b2f8', 'linux_archive_sha256': 'f2bc89592191200a3ce64084c40acb645eb125e7452fb0f1be3f7bab0161935e', 'comparison_repository_commit': '41d333bc0ff6ac332a85e680395c641a7d13c9f7', 'model_manifest': json.loads((root / 'model_bundle/manifest.json').read_text()), 'model_file_sha256': files, 'inference_source_sha256': {p: sha256(root / p) for p in ['engine.py', 'features.py', 'tastemm_engine/model.py']}, 'installed_packages': dict(sorted(((d.metadata['Name'], d.version) for d in metadata.distributions()))), 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(), 'software_license': 'MIT', 'code_signing': 'ad-hoc; not notarized' if mac else 'unsigned'}
(root / 'BUILD_PROVENANCE.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
names = ['LICENSE', 'LICENSE_SCOPE.txt', 'BUILD_PROVENANCE.json']
names += ['RELEASE_MACOS.txt', 'BUILD_MACOS.txt', 'environment_build_macos.txt'] if mac else ['RELEASE_WINDOWS.txt', 'MACOS_ASSESSMENT.txt', 'environment_build_windows.txt']
for name in names:
    shutil.copy2(root / name, target / name)
(target / 'examples').mkdir(exist_ok=True)
(target / 'examples/molecules.csv').write_text('Name,SMILES\nEthanol,CCO\nSodium chloride,[Na+].[Cl-]\nInvalid example,not a smiles\nEmpty example,\n', encoding='utf-8-sig')
notices = target / 'THIRD_PARTY_LICENSES'
notices.mkdir(exist_ok=True)
for dist in metadata.distributions():
    for file in dist.files or []:
        if any((word in file.name.lower() for word in ['license', 'copying', 'notice'])):
            source = Path(dist.locate_file(file))
            if source.is_file() and source.suffix.lower() not in ['.py', '.pyc', '.pyd']:
                output = notices / dist.metadata['Name'] / str(file).replace('..', '_')
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, output)
for name in ['LICENSE.txt', 'LICENSE']:
    source = Path(sys.base_prefix) / name
    if source.is_file():
        shutil.copy2(source, notices / ('Python-' + name))
(notices / 'README.txt').write_text('These are installed dependency notices collected from the build environment.\nProject source code is MIT licensed; see LICENSE and LICENSE_SCOPE.txt.\nDatasets, trained weights and pretrained resources have separate terms.\nBERT resources were preserved byte-for-byte from the original Linux release;\nthat release did not include a model-card revision or separate weight license.\nConfirm that provenance before publishing an open-source redistribution claim.\n', encoding='utf-8')
if mac:
    subprocess.run(['codesign', '--force', '--deep', '--sign', '-', str(distribution)], check=True)
    subprocess.run(['codesign', '--verify', '--deep', '--strict', str(distribution)], check=True)
print(json.dumps({'distribution': str(target), 'model_files': len(files), 'notices': len(list(notices.rglob('*')))}, indent=2))
