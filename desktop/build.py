"""Build and stage the desktop application on its target OS."""
import argparse
import json
import importlib.util
from pathlib import Path
import platform
import shutil
import re
import subprocess
import sys
import plistlib
root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--bundle', type=Path, default=root / 'model_bundle')
parser.add_argument('--without-model', action='store_true')
parser.add_argument('--stage-only', action='store_true', help='Stage resources into an existing native build')
args = parser.parse_args()
command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--windowed', '--name', 'TasteMM', '--paths', str(root), '--collect-all', 'torch_geometric', '--collect-data', 'transformers', '--collect-all', 'rdkit', '--copy-metadata', 'torch', '--copy-metadata', 'transformers', '--copy-metadata', 'tokenizers', '--copy-metadata', 'safetensors', '--hidden-import', 'engine', '--hidden-import', 'features', '--hidden-import', 'app', '--hidden-import', 'validate_workflow', '--hidden-import', 'PIL.ImageTk', '--hidden-import', 'PIL._tkinter_finder', '--hidden-import', 'model', '--hidden-import', 'transformers.models.bert.modeling_bert', '--hidden-import', 'transformers.models.bert.tokenization_bert', '--paths', str(root / 'tastemm_engine')]
for module in ['matplotlib', 'sklearn', 'scipy', 'IPython', 'pytest', 'torchvision', 'torchaudio', 'tensorboard', 'tensorflow', 'jax', 'flax']:
    command += ['--exclude-module', module]
if sys.platform == 'darwin':
    if platform.machine() != 'arm64':
        raise SystemExit('This macOS preview targets native Apple Silicon (arm64).')
    command += ['--target-arch', 'arm64', '--osx-bundle-identifier', 'org.tastemm.desktop']
command += [str(root / 'launch.py')]
if not args.stage_only:
    subprocess.run(command, cwd=root, check=True)
distribution = root / 'dist' / ('TasteMM.app' if sys.platform == 'darwin' else 'TasteMM')
if sys.platform != 'darwin':
    helper = importlib.util.find_spec('PIL._tkinter_finder')
    if helper and helper.origin:
        helper_dir = distribution / '_internal/PIL'
        helper_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(helper.origin, helper_dir / '_tkinter_finder.py')
if sys.platform == 'linux' and Path('/usr/share/tcltk/tk8.6').exists():
    internal = distribution / '_internal'
    libraries = [Path('/lib/x86_64-linux-gnu/libtk8.6.so'), Path('/lib/x86_64-linux-gnu/libtcl8.6.so')]
    if all((p.exists() for p in libraries)):
        for library in libraries:
            shutil.copy2(library, internal / library.name)
            dependencies = subprocess.check_output(['ldd', str(library)], text=True)
            for path in re.findall('=> (/\\S+)', dependencies):
                dependency = Path(path)
                if dependency.name.startswith(('libc.', 'libm.', 'libdl.', 'libpthread.', 'librt.')):
                    continue
                if not (internal / dependency.name).exists():
                    shutil.copy2(dependency, internal / dependency.name)
        for folder, source in [('_tcl_data', '/usr/share/tcltk/tcl8.6'), ('_tk_data', '/usr/share/tcltk/tk8.6')]:
            if (internal / folder).exists():
                shutil.rmtree(internal / folder)
            shutil.copytree(source, internal / folder)
destination = distribution / 'Contents/Resources/model_bundle' if sys.platform == 'darwin' else distribution / 'model_bundle'
if not args.without_model:
    if not (args.bundle / 'manifest.json').exists():
        raise SystemExit('Model bundle is missing. Supply --bundle or build with --without-model.')
    shutil.copytree(args.bundle, destination, dirs_exist_ok=True)
help_text = 'TasteMM Desktop\n\nExtract the complete archive before launching. Keep model_bundle next to the application.\nWindows: open TasteMM.exe. Linux: run ./TasteMM. macOS: open TasteMM.app.\nNo Python installation, GPU or internet connection is required for inference.\n\nSingle molecule: enter SMILES, then click Predict taste.\nBatch CSV: choose a UTF-8 CSV with a SMILES column.\nSelect a result to view its structure and six probabilities. Export CSV saves all results.\n\nModel configuration: fold 0 checkpoint and validation-derived temperature.\nIt is a single-label research model. Salty results remain exploratory because the development dataset has 28 salty records.\n'
help_destination = distribution / 'Contents/Resources/HELP.txt' if sys.platform == 'darwin' else distribution / 'HELP.txt'
help_destination.write_text(help_text, encoding='utf-8')
if sys.platform == 'darwin':
    info_path = distribution / 'Contents/Info.plist'
    with info_path.open('rb') as stream:
        info = plistlib.load(stream)
    info.update(CFBundleShortVersionString='0.1.0', CFBundleVersion='0.1.0', NSHumanReadableCopyright='Copyright (c) 2026 Xiaowei Liu', LSApplicationCategoryType='public.app-category.education')
    with info_path.open('wb') as stream:
        plistlib.dump(info, stream)
    subprocess.run(['codesign', '--force', '--deep', '--sign', '-', str(distribution)], check=True)
status = {'platform': platform.platform(), 'architecture': platform.machine(), 'python': sys.version, 'app_version': '0.1.0', 'distribution': str(distribution), 'model_included': not args.without_model, 'validation': 'Run the frozen executable with --self-test before distribution.'}
(root / 'dist/build_manifest.json').write_text(json.dumps(status, indent=2))
print(json.dumps(status, indent=2))
