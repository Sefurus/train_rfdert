"""
train.py

Entrena un modelo RF-DETR a partir de un dataset ya preparado (carpeta
con train/valid/(test) en formato COCO, generada por prepare_dataset.py).

Toda la configuracion de la corrida vive en un YAML, para poder lanzar
varios modelos/experimentos sin tocar el codigo:

    python train.py --config configs/modelo_small_v1.yaml

Ejemplo de configs/modelo_small_v1.yaml:

    model_size: small          # nano | small | medium | base | large
    dataset_dir: /data/processed/dataset_2
    output_dir: /outputs/modelo_small_v1
    epochs: 10
    lr: 0.0001
    batch_size: 4
    grad_accum_steps: 4
    early_stopping: true
    early_stopping_use_ema: true
    early_stopping_patience: 3
    pretrain_weights: null     # ruta a un checkpoint .pth para continuar/afinar, o null
"""

import argparse
import time
from pathlib import Path

import yaml

MODEL_REGISTRY = {}


def _load_model_registry():
    """Import perezoso de rfdetr para que --help no requiera CUDA/torch listos."""
    from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
    MODEL_REGISTRY.update({
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "base": RFDETRBase,
        "large": RFDETRLarge,
    })


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    required = ["model_size", "dataset_dir", "output_dir", "epochs", "lr", "batch_size"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Faltan campos obligatorios en el config: {missing}")

    cfg.setdefault("grad_accum_steps", 1)
    cfg.setdefault("early_stopping", True)
    cfg.setdefault("early_stopping_use_ema", True)
    cfg.setdefault("early_stopping_patience", 3)
    cfg.setdefault("pretrain_weights", None)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo RF-DETR desde un config YAML")
    parser.add_argument("--config", required=True, help="Ruta al archivo YAML de configuracion")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg["model_size"] not in ("nano", "small", "medium", "base", "large"):
        raise ValueError(f"model_size invalido: {cfg['model_size']}")

    dataset_dir = Path(cfg["dataset_dir"])
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe dataset_dir: {dataset_dir}")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Config:        {args.config}")
    print(f"Modelo:        RFDETR{cfg['model_size'].capitalize()}")
    print(f"Dataset:       {dataset_dir}")
    print(f"Output:        {output_dir}")
    print(f"Epochs:        {cfg['epochs']}")
    print(f"LR:            {cfg['lr']}")
    print(f"Batch size:    {cfg['batch_size']} (grad_accum={cfg['grad_accum_steps']})")
    print("=" * 60)

    _load_model_registry()
    model_cls = MODEL_REGISTRY[cfg["model_size"]]

    model_kwargs = {}
    if cfg.get("pretrain_weights"):
        model_kwargs["pretrain_weights"] = cfg["pretrain_weights"]

    model = model_cls(**model_kwargs)

    start = time.time()
    model.train(
        dataset_dir=str(dataset_dir),
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        batch_size=cfg["batch_size"],
        grad_accum_steps=cfg["grad_accum_steps"],
        early_stopping=cfg["early_stopping"],
        early_stopping_use_ema=cfg["early_stopping_use_ema"],
        early_stopping_patience=cfg["early_stopping_patience"],
        output_dir=str(output_dir),
    )
    elapsed = time.time() - start

    print("=" * 60)
    print(f"Entrenamiento terminado en {elapsed/60:.1f} min")
    print(f"Checkpoints en: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
