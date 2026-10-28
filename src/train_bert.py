"""Etapa 5: fine-tuning de ModernBERT como regresor de años de experiencia.

Modelo: answerdotai/ModernBERT-base (149 M parámetros) con una cabeza de regresión
(AutoModelForSequenceClassification, num_labels=1). A diferencia de los embeddings
congelados, aquí se ajustan TODOS los pesos, así que el modelo puede aprender a
fijarse en las fechas.

Decisiones:
  - Objetivo escalado: y / 20 para entrenar (las redes optimizan mejor con salidas
    en ~[0, 1]); las predicciones se devuelven a años (× 20) y se recortan a [0, 20].
  - Pérdida (decisión L): MSE por defecto (la exige el enunciado; las etiquetas son
    exactas y los errores grandes cuestan más). --loss huber es el experimento de Q3,
    con delta = 1 año: errores < 1 año se penalizan cuadrático, mayores, lineal.
  - Textos largos (decisión K): max_len = 256. El CV limpio más largo tiene 202
    tokens de ModernBERT, así que ningún CV se trunca (se verifica y se reporta).
  - Early stopping por MAE de validación (paciencia 2 épocas); se conserva la mejor
    época. El test no interviene en ninguna decisión.

Salidas:
  models/bert/<run>/                         mejores pesos + tokenizer (no se versionan)
  output/predictions/<run>.csv               val, test y test_ref
  output/metrics/<run>_run.json              hiperparámetros, historial, tiempos, hardware

Uso (en Colab con GPU):
    python -m src.train_bert                 # MSE
    python -m src.train_bert --loss huber    # Huber (Q3)
Prueba rápida local (CPU; ModernBERT de 2 capas con pesos aleatorios, no guarda nada):
    python -m src.train_bert --smoke
"""

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    EarlyStoppingCallback, Trainer, TrainingArguments,
)

from src.evaluation import fmt, regression_metrics, save_predictions, save_run_info
from src.utils import MODELS_DIR, SEED, TARGET, TARGET_CAP, ensure_dirs, load_splits, set_seed

MODEL_NAME = "answerdotai/ModernBERT-base"
HPARAMS = dict(max_len=256, lr=5e-5, batch_size=16, max_epochs=10, patience=2,
               warmup_ratio=0.1, weight_decay=0.01)
HUBER_DELTA_YEARS = 1.0


class RegressionTrainer(Trainer):
    """Trainer con la pérdida elegida (MSE o Huber) sobre el objetivo escalado."""

    def __init__(self, *args, loss: str = "mse", **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        preds = outputs.logits.squeeze(-1).float()
        if self.loss_name == "huber":
            loss = torch.nn.functional.huber_loss(preds, labels, delta=HUBER_DELTA_YEARS / TARGET_CAP)
        else:
            loss = torch.nn.functional.mse_loss(preds, labels)
        return (loss, outputs) if return_outputs else loss


def to_years(scaled) -> np.ndarray:
    return np.clip(np.asarray(scaled, dtype=float).reshape(-1) * TARGET_CAP, 0, TARGET_CAP)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    return {"mae": float(np.mean(np.abs(to_years(preds) - to_years(labels))))}


def make_dataset(tok, df, col: str, max_len: int) -> tuple[Dataset, int]:
    enc = tok(df[col].tolist(), truncation=True, max_length=max_len)
    full = tok(df[col].tolist(), truncation=False)["input_ids"]
    n_truncated = int(sum(len(x) > max_len for x in full))
    ds = Dataset.from_dict({**enc, "labels": (df[TARGET].values / TARGET_CAP).astype("float32").tolist()})
    return ds, n_truncated


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def main(loss: str, model_name: str, smoke: bool, ckpt_dir: str | None) -> None:
    ensure_dirs()
    set_seed()
    run = f"modernbert_{loss}"
    s = load_splits()
    if smoke:  # mecánica del pipeline en CPU: pocos datos, 1 época
        s = {k: v.sample(min(len(v), 64 if k == "train" else 32), random_state=SEED) for k, v in s.items()}

    hp = dict(HPARAMS, max_epochs=1 if smoke else HPARAMS["max_epochs"])
    tok = AutoTokenizer.from_pretrained(model_name)
    ds = {}
    truncated = {}
    for name, df, col in [("train", s["train"], "text_clean"), ("val", s["val"], "text_clean"),
                          ("test", s["test"], "text_clean"), ("test_ref", s["test"], "text_ref_clean")]:
        ds[name], truncated[name] = make_dataset(tok, df, col, hp["max_len"])

    if smoke:  # ModernBERT en miniatura con pesos aleatorios: prueba el código, no el modelo
        cfg = AutoConfig.from_pretrained(model_name, num_labels=1, problem_type="regression",
                                         hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                         num_attention_heads=2)
        if getattr(cfg, "layer_types", None):  # ModernBERT: un tipo de atención por capa
            cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
        model = AutoModelForSequenceClassification.from_config(cfg)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1, problem_type="regression")

    ckpt = Path(ckpt_dir or tempfile.gettempdir()) / "cv_bert_ckpt" / run
    args = TrainingArguments(
        output_dir=str(ckpt),
        learning_rate=hp["lr"], per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=64, num_train_epochs=hp["max_epochs"],
        # transformers 5: warmup_steps en [0, 1) se interpreta como proporción de pasos
        warmup_steps=hp["warmup_ratio"], weight_decay=hp["weight_decay"],
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="mae", greater_is_better=False,
        logging_strategy="epoch", report_to="none",
        fp16=torch.cuda.is_available(), seed=SEED, data_seed=SEED,
    )
    trainer = RegressionTrainer(
        model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["val"],
        data_collator=DataCollatorWithPadding(tok), processing_class=tok,
        compute_metrics=compute_metrics, loss=loss,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=hp["patience"])],
    )

    t0 = time.perf_counter()
    trainer.train()
    train_secs = time.perf_counter() - t0

    pred = {k: to_years(trainer.predict(ds[k]).predictions) for k in ("val", "test_ref")}
    t1 = time.perf_counter()
    pred["test"] = to_years(trainer.predict(ds["test"]).predictions)
    infer_per_cv = (time.perf_counter() - t1) / len(s["test"])

    val = regression_metrics(s["val"][TARGET], pred["val"])
    history = [{"epoch": h["epoch"], "val_mae": h["eval_mae"], "val_loss": h["eval_loss"]}
               for h in trainer.state.log_history if "eval_mae" in h]
    print(f"[val] {fmt(val)}")
    print("Historial:", json.dumps(history))
    if smoke:
        print("Prueba de humo OK (no se guardó nada).")
        return

    out_dir = MODELS_DIR / "bert" / run
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    save_predictions(run, {"val": (s["val"], pred["val"]), "test": (s["test"], pred["test"]),
                           "test_ref": (s["test"], pred["test_ref"])})
    save_run_info(run, {
        "model": run, "base_model": model_name, "loss": loss,
        "huber_delta_years": HUBER_DELTA_YEARS if loss == "huber" else None,
        "target_scaling": f"y / {TARGET_CAP}", "hparams": hp, "seed": SEED,
        "truncated_cvs": truncated,
        "best_epoch": min(history, key=lambda h: h["val_mae"])["epoch"] if history else None,
        "epochs_run": history[-1]["epoch"] if history else None, "history": history,
        "val_metrics": val, "train_secs": train_secs, "infer_secs_per_cv": infer_per_cv,
        "hardware": hardware(), "fp16": torch.cuda.is_available(),
    })
    print(f"Modelo guardado en {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--smoke", action="store_true", help="prueba rápida con pocos datos; no guarda nada")
    parser.add_argument("--ckpt-dir", default=None, help="carpeta para checkpoints intermedios")
    args = parser.parse_args()
    main(args.loss, args.model_name, args.smoke, args.ckpt_dir)
