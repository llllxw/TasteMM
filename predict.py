from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import BertModel, BertTokenizer

from confidence_methods import conf_margin_from_logits, softmax_np, temperature_scale_logits
from data_process import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    get_mix_fingerprint,
    mean_pool_last_hidden_state,
    mol_from_smiles,
    smiles_to_graph,
)
from model import TasteBaselineModel
from tools import RESULTS_DIR, ensure_dirs, setup_seed


setup_seed(42)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
LABEL_MAP = {
    0: "bitter",
    1: "sweet",
    2: "umami",
    3: "salty",
    4: "sour",
    5: "tasteless",
}


def get_args():
    parser = argparse.ArgumentParser(description="Predict with the final baseline model.")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    return parser.parse_args()


def load_run_payload(run_dir: str) -> dict:
    result_json = os.path.join(run_dir, "result.json")
    with open(result_json, "r", encoding="utf-8") as f:
        return json.load(f)


def get_bert_embeds_single(smiles: str):
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            smiles,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)
        output = model(**inputs)
        embed = mean_pool_last_hidden_state(output.last_hidden_state, inputs["attention_mask"])
        embed = embed.squeeze(0).cpu().numpy()
    return embed


def preprocess_new_data(input_csv: str, encoding: str):
    df = pd.read_csv(input_csv, encoding=encoding)
    required_cols = ["ID", "Name", "SMILES"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    invalid_rows = []
    for idx, row in df.iterrows():
        try:
            mol_from_smiles(row["SMILES"])
        except ValueError as exc:
            invalid_rows.append(f"row={idx}, id={row.get('ID', idx)}, {exc}")
    if invalid_rows:
        preview = "\n".join(invalid_rows[:20])
        suffix = "" if len(invalid_rows) <= 20 else f"\n... {len(invalid_rows) - 20} more invalid rows"
        raise ValueError(f"Invalid SMILES found in prediction input:\n{preview}{suffix}")

    df = df.copy().reset_index(drop=True)
    df["SMILES"] = df["SMILES"].astype(str).str.strip()
    bert_embeds = [get_bert_embeds_single(smiles) for smiles in df["SMILES"]]

    data_list = []
    valid_indices = []
    for idx, (_, row) in enumerate(df.iterrows()):
        smiles = row["SMILES"]
        x, edge_index, edge_attr = smiles_to_graph(smiles)
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mixfp=torch.tensor(get_mix_fingerprint(smiles), dtype=torch.float32),
            bert=torch.tensor(bert_embeds[idx], dtype=torch.float32),
            id=row["ID"],
            name=row["Name"],
        )
        data_list.append(data)
        valid_indices.append(idx)

    return data_list, df.iloc[valid_indices].copy()


def load_model(run_dir: str, payload: dict):
    config = payload["config"]
    checkpoints = payload["checkpoints"]
    model_path = checkpoints["best_classify_model"]
    model = TasteBaselineModel(
        num_graph_features=NODE_FEATURE_DIM,
        edge_attr_dim=EDGE_FEATURE_DIM,
        graph_aux_hidden_dim=128,
        num_classes=config["num_classes"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def load_temperature(payload: dict) -> float:
    return float(payload.get("temperature_T", 1.0))


def predict(run_dir: str, input_csv: str, encoding: str, output_csv: str, confidence_threshold: float):
    ensure_dirs()
    payload = load_run_payload(run_dir)
    model = load_model(run_dir, payload)
    temperature = load_temperature(payload)
    dataset, df_valid = preprocess_new_data(input_csv, encoding)

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    logits_list = []
    with torch.no_grad():
        for data in loader:
            data = data.to(DEVICE)
            logits_list.append(model(data, mode="classify").cpu().numpy())
    logits = np.concatenate(logits_list, axis=0)
    logits_ts = temperature_scale_logits(logits, temperature)
    probs = softmax_np(logits_ts)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    margin = conf_margin_from_logits(logits_ts)

    df_valid["pred_label_id"] = pred
    df_valid["pred_label"] = [LABEL_MAP[idx] for idx in pred]
    df_valid["pred_confidence"] = conf
    df_valid["pred_margin"] = margin
    df_valid["high_confidence"] = conf >= confidence_threshold

    if not output_csv:
        output_csv = os.path.join(RESULTS_DIR, "predictions", "baseline_predictions.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_valid.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] Predictions saved to: {output_csv}")
    return df_valid


if __name__ == "__main__":
    args = get_args()
    predict(
        run_dir=args.run_dir,
        input_csv=args.input_csv,
        encoding=args.encoding,
        output_csv=args.output_csv,
        confidence_threshold=args.confidence_threshold,
    )
