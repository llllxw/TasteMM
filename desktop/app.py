"""TasteMM desktop preview. Model loading and inference run off the UI thread."""
import math
from pathlib import Path
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from paths import default_model_bundle

BG, WHITE, INK, MUTED, TEAL, LINE = '#F3F6F8', '#FFFFFF', '#142B35', '#687F88', '#137F74', '#DFE8EA'
TASTES = ['bitter', 'sweet', 'umami', 'salty', 'sour', 'tasteless']
COLORS = ['#638479', '#D7A665', '#9B88B6', '#6C9DB8', '#A5AD68', '#A4B5B9']


class App:
    def label(self, parent, text, size=10, color=INK, bold=False, bg=WHITE, **kw):
        return tk.Label(parent, text=text, font=(self.font, size, 'bold' if bold else 'normal'), fg=color, bg=bg, **kw)

    def __init__(self, root):
        self.root = root
        root.title('TasteMM | Molecular taste prediction')
        root.geometry('1280x860'); root.minsize(1160, 800); root.configure(bg=BG)
        self.font = 'Arial' if sys.platform != 'darwin' else 'Helvetica Neue'
        root.option_add('*Font', (self.font, 10))
        self.messages = queue.Queue()
        self.predictor = self.predictor_bundle = self.result = self.molecule_photo = self.current_values = None
        self.busy, self.mode = False, 'single'
        self.bundle = tk.StringVar(value=str(default_model_bundle()))
        self.csv, self.smiles = tk.StringVar(), tk.StringVar()
        self.status = tk.StringVar(value='Ready. Enter a molecule to begin.')
        style = ttk.Style(root); style.theme_use('clam')
        style.configure('TButton', font=(self.font, 10), padding=(12, 9), background=WHITE, foreground=INK, borderwidth=0)
        style.map('TButton', background=[('active', '#E9F1F0')])
        style.configure('Primary.TButton', background=TEAL, foreground=WHITE, font=(self.font, 11, 'bold'), padding=(18, 11))
        style.map('Primary.TButton', background=[('disabled', '#B5CBC6'), ('active', '#0C695F')], foreground=[('disabled', WHITE)])
        style.configure('Tab.TButton', background='#EEF3F4', foreground=MUTED, padding=(12, 8))
        style.configure('Active.Tab.TButton', background='#DFF1EC', foreground='#10665B', padding=(12, 8))
        style.configure('TEntry', fieldbackground=WHITE, foreground=INK, padding=9, bordercolor=LINE, lightcolor=LINE, darkcolor=LINE)
        style.configure('Treeview', font=(self.font, 10), rowheight=36, background=WHITE, fieldbackground=WHITE, foreground=INK, borderwidth=0)
        style.configure('Treeview.Heading', font=(self.font, 9, 'bold'), background='#EEF3F5', foreground=MUTED, padding=(8, 10), relief='flat')
        style.map('Treeview', background=[('selected', '#E1F2ED')], foreground=[('selected', INK)])
        style.configure('TProgressbar', background=TEAL, troughcolor='#EAF0F1', borderwidth=0)
        self.sidebar(); self.content()
        root.after(100, self.poll)

    def sidebar(self):
        side = tk.Frame(self.root, bg=INK, width=202)
        side.pack(side='left', fill='y'); side.pack_propagate(False)
        brand = tk.Frame(side, bg=INK)
        brand.pack(fill='x', padx=20, pady=(29, 40))
        logo = tk.Canvas(brand, width=40, height=44, bg=INK, highlightthickness=0)
        logo.pack(side='left')
        logo.create_polygon(20, 2, 37, 12, 37, 32, 20, 42, 3, 32, 3, 12, outline='#75C6AC', fill=INK, width=2)
        logo.create_text(20, 22, text='M', font=(self.font, 15, 'bold'), fill='#B8E3D4')
        self.label(brand, 'TasteMM', 18, WHITE, True, INK).pack(side='left', padx=(8, 0))
        nav = tk.Frame(side, bg='#25454D'); nav.pack(fill='x', padx=14)
        self.label(nav, '  Predict molecules', 11, '#D7F0E6', True, '#25454D', pady=13).pack(anchor='w', padx=9)
        self.label(side, 'LOCAL WORKSPACE', 8, '#86A1AA', bg=INK).pack(anchor='w', padx=25, pady=(38, 14))
        for text in ['6 taste classes', 'Single & batch inputs', 'Works offline']:
            self.label(side, text, 10, '#A4BAC0', bg=INK).pack(anchor='w', padx=25, pady=(0, 10))
        bottom = tk.Frame(side, bg=INK); bottom.pack(side='bottom', fill='x', padx=24, pady=25)
        tk.Button(bottom, text='Model settings', command=self.choose_bundle, bg=INK, fg='#B4CBD0', activebackground='#25454D', activeforeground=WHITE, relief='flat', anchor='w', cursor='hand2').pack(fill='x')
        self.label(bottom, 'TasteMM Desktop · Preview', 8, '#78959F', bg=INK).pack(anchor='w', pady=(12, 0))

    def card(self, parent):
        return tk.Frame(parent, bg=WHITE, highlightbackground=LINE, highlightthickness=1)

    def content(self):
        main = tk.Frame(self.root, bg=BG); main.pack(fill='both', expand=True, padx=28, pady=24)
        self.label(main, 'MOLECULAR INSIGHTS', 9, TEAL, True, BG).pack(anchor='w')
        self.label(main, 'Explore molecular taste.', 27, INK, True, BG).pack(anchor='w', pady=(6, 22))
        work = tk.Frame(main, bg=BG); work.pack(fill='x')
        work.grid_columnconfigure(0, weight=4, uniform='cards'); work.grid_columnconfigure(1, weight=6, uniform='cards')
        left = self.card(work); left.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        left.configure(height=356); left.grid_propagate(False); left.grid_columnconfigure(0, weight=1)
        self.label(left, '01  /  INPUT', 9, MUTED, True).grid(row=0, column=0, sticky='w', padx=21, pady=(20, 13))
        tabs = tk.Frame(left, bg=WHITE); tabs.grid(row=1, column=0, sticky='w', padx=21)
        self.single_tab = ttk.Button(tabs, text='Single molecule', command=lambda: self.set_mode('single'), style='Active.Tab.TButton'); self.single_tab.pack(side='left')
        self.batch_tab = ttk.Button(tabs, text='Batch CSV', command=lambda: self.set_mode('batch'), style='Tab.TButton'); self.batch_tab.pack(side='left', padx=5)
        self.input_body = tk.Frame(left, bg=WHITE); self.input_body.grid(row=2, column=0, sticky='ew', padx=21, pady=(18, 0))
        self.label(self.input_body, 'SMILES string', 10, INK, True).pack(anchor='w')
        self.entry = ttk.Entry(self.input_body, textvariable=self.smiles, font=(self.font, 11)); self.entry.pack(fill='x', pady=(8, 10))
        self.entry.bind('<Return>', lambda event: self.start())
        examples = tk.Frame(self.input_body, bg=WHITE); examples.pack(fill='x')
        self.label(examples, 'Try', 9, MUTED).pack(side='left', padx=(0, 5))
        for name, smiles in [('Ethanol', 'CCO'), ('Sodium chloride', '[Na+].[Cl-]')]:
            tk.Button(examples, text=name, fg=TEAL, bg=WHITE, activebackground='#EDF7F4', relief='flat', font=(self.font, 9), cursor='hand2', command=lambda s=smiles: self.smiles.set(s)).pack(side='left')
        self.batch_body = tk.Frame(left, bg=WHITE)
        self.label(self.batch_body, 'CSV file', 10, INK, True).pack(anchor='w')
        self.csv_label = self.label(self.batch_body, 'No file selected', 9, MUTED, anchor='w', wraplength=290); self.csv_label.pack(fill='x', pady=(10, 5))
        ttk.Button(self.batch_body, text='Choose CSV file', command=self.choose_csv).pack(anchor='w')
        self.label(self.batch_body, 'Include a column named SMILES.', 9, MUTED).pack(anchor='w', pady=(9, 0))
        self.run = ttk.Button(left, text='Predict taste', style='Primary.TButton', command=self.start); self.run.grid(row=3, column=0, sticky='ew', padx=21, pady=(24, 0))
        self.progress = ttk.Progressbar(left, mode='indeterminate'); self.progress.grid(row=4, column=0, sticky='ew', padx=21, pady=(12, 4))
        self.progress.grid_remove()
        right = self.card(work); right.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
        self.label(right, '02  /  TASTE PROFILE', 9, MUTED, True).pack(anchor='w', padx=21, pady=(20, 7))
        headline = tk.Frame(right, bg=WHITE); headline.pack(fill='x', padx=21)
        self.prediction = self.label(headline, 'Awaiting prediction', 20, INK, True); self.prediction.pack(side='left')
        self.probability = self.label(headline, '', 19, TEAL, True); self.probability.pack(side='right')
        self.profile_note = self.label(right, 'Select a result to explore its probability profile.', 9, MUTED, anchor='w'); self.profile_note.pack(fill='x', padx=21, pady=(5, 12))
        profile = tk.Frame(right, bg=WHITE); profile.pack(fill='both', expand=True, padx=15, pady=(0, 8))
        self.molecule = self.label(profile, 'Molecular\nstructure', 11, '#9AADB3', width=16); self.molecule.pack(side='left', fill='both', padx=(0, 8))
        self.bars = tk.Canvas(profile, bg=WHITE, highlightthickness=0, height=196); self.bars.pack(side='left', fill='both', expand=True)
        self.bars.bind('<Configure>', lambda event: self.render_bars(self.current_values))
        self.render_bars(None)
        heading = tk.Frame(main, bg=BG); heading.pack(fill='x', pady=(20, 10))
        self.label(heading, 'Prediction results', 15, INK, True, BG).pack(side='left')
        self.count = self.label(heading, 'No molecules yet', 9, MUTED, bg=BG); self.count.pack(side='left', padx=14)
        self.export_button = ttk.Button(heading, text='Export CSV  ↗', command=self.export, state='disabled'); self.export_button.pack(side='right')
        holder = self.card(main); holder.pack(fill='both', expand=True)
        self.table = ttk.Treeview(holder, columns=['SMILES', 'taste', 'probability', 'status'], show='headings', selectmode='browse', height=4)
        for col, title, width in [('SMILES', 'MOLECULAR SMILES', 390), ('taste', 'PREDICTED TASTE', 145), ('probability', 'MODEL PROBABILITY', 165), ('status', 'STATUS', 135)]:
            self.table.heading(col, text=title, anchor='w'); self.table.column(col, width=width, minwidth=100, anchor='w')
        self.table.tag_configure('even', background='#F8FAFB'); self.table.tag_configure('invalid', foreground='#A06C45')
        vertical = ttk.Scrollbar(holder, orient='vertical', command=self.table.yview); horizontal = ttk.Scrollbar(holder, orient='horizontal', command=self.table.xview)
        self.table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        vertical.pack(side='right', fill='y'); horizontal.pack(side='bottom', fill='x'); self.table.pack(fill='both', expand=True)
        self.table.bind('<<TreeviewSelect>>', self.select_result)

    def set_mode(self, mode):
        if self.busy: return
        self.mode = mode
        self.single_tab.configure(style='Active.Tab.TButton' if mode == 'single' else 'Tab.TButton')
        self.batch_tab.configure(style='Active.Tab.TButton' if mode == 'batch' else 'Tab.TButton')
        self.input_body.grid_remove(); self.batch_body.grid_remove()
        (self.input_body if mode == 'single' else self.batch_body).grid(row=2, column=0, sticky='ew', padx=21, pady=(18, 0))

    def choose_bundle(self):
        if self.busy: return
        path = filedialog.askdirectory(title='Choose TasteMM model bundle')
        if path: self.bundle.set(path); self.status.set('Model location updated for the next prediction.')

    def choose_csv(self):
        path = filedialog.askopenfilename(filetypes=[('CSV', '*.csv')])
        if path: self.csv.set(path); self.csv_label.configure(text=Path(path).name)

    def start(self):
        if self.busy: return
        bundle, csv, smiles, mode = self.bundle.get(), self.csv.get(), self.smiles.get(), self.mode
        if (mode == 'batch' and not csv) or (mode == 'single' and not smiles.strip()):
            messagebox.showerror('Input required', 'Enter SMILES or choose a CSV in the Batch CSV tab.'); return
        self.busy = True; self.run.configure(state='disabled', text='Predicting…'); self.export_button.configure(state='disabled'); self.progress.grid(); self.progress.start(12)
        self.status.set('Loading model and predicting on this device…')
        def work():
            try:
                import pandas as pd
                from engine import Predictor
                if self.predictor is None or self.predictor_bundle != bundle:
                    self.predictor = Predictor(bundle); self.predictor_bundle = bundle
                frame = pd.read_csv(csv) if mode == 'batch' else pd.DataFrame({'SMILES': [smiles]})
                self.messages.put(('result', self.predictor.predict(frame)))
            except Exception as error: self.messages.put(('error', str(error)))
        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def clean(value):
        return '' if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)

    def show_result(self, value):
        self.result = value; self.table.delete(*self.table.get_children())
        for idx, row in value.head(1000).iterrows():
            valid = row.status == 'ok'; probabilities = [float(row['probability_' + t]) for t in TASTES]
            probability = f'{max(probabilities):.1%}' if valid else '—'
            tags = ('even' if idx % 2 == 0 else '',) + (() if valid else ('invalid',))
            self.table.insert('', 'end', iid=str(idx), values=[self.clean(row.SMILES) or '(empty input)', self.clean(row.predicted_taste).title() if valid else '—', probability, 'Predicted' if valid else 'Invalid SMILES'], tags=tags)
        self.count.configure(text=f'{len(value):,} molecules')
        ok = int((value.status == 'ok').sum())
        self.status.set(f'{ok:,} predicted · {len(value) - ok:,} invalid. Select a row to view its structure and probabilities.' + (' Showing 1,000 rows; export contains all rows.' if len(value) > 1000 else ''))
        self.export_button.configure(state='normal')
        if len(value): self.table.selection_set('0'); self.select_result()
        else:
            self.current_values = None; self.render_bars(None); self.prediction.configure(text='No molecules found'); self.probability.configure(text=''); self.profile_note.configure(text='The input CSV contains no rows.'); self.molecule.configure(image='', text='Molecular\nstructure')

    def select_result(self, event=None):
        selected = self.table.selection()
        if not selected or self.result is None: return
        row = self.result.iloc[int(selected[0])]; valid = row.status == 'ok'
        values = [float(row['probability_' + t]) for t in TASTES] if valid else None
        self.current_values = values
        self.prediction.configure(text=self.clean(row.predicted_taste).title() if valid else 'Check molecular input')
        self.probability.configure(text=f'{max(values):.1%}' if valid else '')
        self.profile_note.configure(text='Model probability · six mutually exclusive classes' if valid else 'Invalid SMILES. No prediction was made.')
        self.render_bars(values)
        try:
            if not valid: raise ValueError('No structure')
            from rdkit import Chem
            from rdkit.Chem import Draw
            from PIL import ImageTk
            drawing = Draw.MolToImage(Chem.MolFromSmiles(row.SMILES), size=(330, 330)); drawing.thumbnail((150, 160))
            self.molecule_photo = ImageTk.PhotoImage(drawing); self.molecule.configure(image=self.molecule_photo, text='', width=155)
        except Exception: self.molecule.configure(image='', text='Structure\nunavailable', width=16)

    def render_bars(self, values):
        self.bars.delete('all'); width = max(self.bars.winfo_width(), 230); left, end = 65, width - 50
        for i, (taste, color) in enumerate(zip(TASTES, COLORS)):
            y = 12 + i * 30
            self.bars.create_text(0, y + 6, text=taste.title(), anchor='w', fill=MUTED, font=(self.font, 9))
            self.bars.create_rectangle(left, y, end, y + 12, fill='#EEF3F4', outline='')
            if values is not None: self.bars.create_rectangle(left, y, left + max(0, end - left) * values[i], y + 12, fill=TEAL if values[i] == max(values) else color, outline='')
            self.bars.create_text(width - 3, y + 6, text=f'{values[i]:.1%}' if values is not None else '—', anchor='e', fill=INK, font=(self.font, 9))

    def poll(self):
        try:
            kind, value = self.messages.get_nowait(); self.busy = False; self.run.configure(state='normal', text='Predict taste'); self.progress.stop(); self.progress.grid_remove()
            if kind == 'error':
                self.status.set('Prediction failed. Check your input and model files.'); self.export_button.configure(state='normal' if self.result is not None else 'disabled'); messagebox.showerror('Prediction failed', value)
            else: self.show_result(value)
        except queue.Empty: pass
        self.root.after(100, self.poll)

    def export(self):
        if self.result is None: return
        path = filedialog.asksaveasfilename(defaultextension='.csv', initialfile='TasteMM_predictions.csv', filetypes=[('CSV', '*.csv')])
        if path:
            try: self.result.to_csv(path, index=False, encoding='utf-8-sig'); self.status.set(f'Results exported to {Path(path).name}.')
            except OSError as error: messagebox.showerror('Export failed', str(error))


if __name__ == '__main__':
    root = tk.Tk(); App(root); root.mainloop()
