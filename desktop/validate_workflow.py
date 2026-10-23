"""Exercise real desktop input, worker, selection, drawing and CSV export."""
from pathlib import Path
import tempfile
import time


def validate(bundle=None):
    import numpy as np
    import pandas as pd
    import torch
    import tkinter as tk
    import app as gui

    torch.set_num_threads(2)
    errors = []
    root = tk.Tk()
    root.withdraw()
    root.report_callback_exception = lambda *error: errors.append(str(error))
    desktop = gui.App(root)
    if bundle:
        desktop.bundle.set(str(bundle))
    original_error = gui.messagebox.showerror
    original_save = gui.filedialog.asksaveasfilename
    gui.messagebox.showerror = lambda *args, **kwargs: errors.append(str(args))

    def run():
        desktop.start()
        deadline = time.monotonic() + 180
        while desktop.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(0.02)
        root.update()
        assert not desktop.busy, 'Desktop prediction timed out'
        assert not errors, errors

    try:
        with tempfile.TemporaryDirectory(prefix='TasteMM-validation-') as temp:
            temp = Path(temp)
            desktop.smiles.set('CCO')
            run()
            assert desktop.result.status.tolist() == ['ok']
            assert desktop.molecule_photo is not None, 'No molecule image'
            csv = temp / 'molecules.csv'
            frame = pd.DataFrame({'Name': ['ethanol', 'invalid', 'empty', 'salt'],
                                  'SMILES': ['CCO', 'not a smiles', '', '[Na+].[Cl-]']})
            frame.to_csv(csv, index=False)
            desktop.set_mode('batch')
            desktop.csv.set(str(csv))
            run()
            assert desktop.result.status.tolist() == ['ok', 'invalid_smiles', 'invalid_smiles', 'ok']
            assert len(desktop.table.get_children()) == 4
            desktop.table.selection_set('1')
            desktop.select_result()
            assert desktop.current_values is None
            desktop.table.selection_set('3')
            desktop.select_result()
            assert desktop.current_values is not None
            columns = ['probability_' + taste for taste in gui.TASTES]
            assert np.allclose(desktop.result.loc[[0, 3], columns].sum(axis=1), 1)
            exported = temp / 'exported.csv'
            gui.filedialog.asksaveasfilename = lambda **kw: str(exported)
            desktop.export()
            result = pd.read_csv(exported)
            assert result.Name.tolist() == frame.Name.tolist()
            assert result.status.tolist() == desktop.result.status.tolist()
            assert np.allclose(result.loc[[0, 3], columns], desktop.result.loc[[0, 3], columns])
            assert not errors, errors
        return {'passed': True, 'single_input': True, 'batch_csv': True,
                'invalid_and_empty_rows_preserved': True, 'result_selection': True,
                'molecule_drawing': True, 'probability_sum': True, 'csv_export': True,
                'tk_callback_errors': errors, 'automated_widget_test': True}
    finally:
        gui.messagebox.showerror = original_error
        gui.filedialog.asksaveasfilename = original_save
        root.destroy()
