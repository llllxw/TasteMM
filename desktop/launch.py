"""Native executable entry point, with an offline inference verification command."""
import argparse
import json
from pathlib import Path
import sys
import os
import tempfile

# All resources ship with the application. Never consult the model hub.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('HF_HOME', str(Path(tempfile.gettempdir()) / 'TasteMM' / 'huggingface'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--self-test-ui', action='store_true')
    parser.add_argument('--self-test-workflow', action='store_true')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.self_test_workflow:
        from validate_workflow import validate
        report = validate(args.bundle)
        if args.report:
            args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
        return
    if args.self_test_ui:
        import tkinter as tk
        from PIL import ImageTk
        from rdkit import Chem
        from rdkit.Chem import Draw
        root = tk.Tk()
        root.withdraw()
        drawing = Draw.MolToImage(Chem.MolFromSmiles('CCO'), size=(150, 160))
        photo = ImageTk.PhotoImage(drawing)
        label = tk.Label(root, image=photo)
        label.pack()
        root.update()
        root.destroy()
        if args.report:
            args.report.write_text(json.dumps({'passed': True, 'structure_drawing': True, 'tk_photo_image': True}))
        return
    if args.self_test:
        import numpy as np
        import pandas as pd
        import torch
        from engine import Predictor, TASTES
        from paths import default_model_bundle
        torch.set_num_threads(2)
        bundle = args.bundle or default_model_bundle()
        reference = pd.read_csv(bundle / 'reference_predictions.csv')
        result = Predictor(bundle).predict(reference[['SMILES']])
        assert result.status.tolist() == reference.status.tolist(), 'Input handling mismatch'
        assert result.predicted_taste.fillna('').tolist() == reference.predicted_taste.fillna('').tolist(), 'Class mismatch'
        columns = ['probability_' + t for t in TASTES]
        valid = reference.status == 'ok'
        error = float(np.max(np.abs(result.loc[valid, columns].values - reference.loc[valid, columns].values)))
        assert error < 1e-4, f'Prediction mismatch: {error}'
        report = {'passed': True, 'platform': sys.platform, 'records': len(reference), 'max_probability_error': error, 'offline': True}
        if args.report:
            args.report.write_text(json.dumps(report, indent=2))
        if sys.stdout:
            print(json.dumps(report))
        return
    import tkinter as tk
    from app import App
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        import tempfile
        log = Path(tempfile.gettempdir()) / 'TasteMM-error.log'
        log.write_text(traceback.format_exc(), encoding='utf-8')
        if not any(arg.startswith('--self-test') for arg in sys.argv[1:]):
            from tkinter import messagebox
            messagebox.showerror('TasteMM could not start', f'Diagnostic details were saved to:\n{log}')
        raise
