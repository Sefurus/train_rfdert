"""
prepare_dataset.py

Toma un dataset en formato COCO (imagenes + un unico json de anotaciones),
lo divide en train/valid/(test) y guarda cada split como una carpeta COCO
independiente, lista para pasarle a rfdetr.

Reemplaza el flujo manual del notebook (celdas de "Preparando Base de datos"
+ "Funcion de division de la base de datos" + "Data Augmentation").

Uso:
    python prepare_dataset.py \
        --base-path /data/raw/ID1_Fusion \
        --annotations /data/raw/ID1_Fusion/coco_detection.json \
        --output /data/processed/dataset_2 \
        --split-ratio 0.7 \
        --include-test \
        --test-ratio 0.95 \
        --aumento-factor 1

Con --aumento-factor 1 no se generan copias aumentadas extra (solo se
guarda el dataset ya dividido). Con --aumento-factor N > 1 se generan
N-1 copias aumentadas adicionales por imagen de train.
"""

import argparse
import json
import os
import shutil
from dataclasses import replace
from glob import glob

import albumentations as A
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm


def build_train_transform() -> A.Compose:
    """Pipeline de augmentation para train. Editar aqui para ajustar
    la receta de aumento entre experimentos."""
    return A.Compose(
        [
            A.Resize(
                height=640, width=640,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                area_for_downscale="image", p=1.0,
            ),
            A.HorizontalFlip(p=0.8),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0),
            A.Perspective(p=0.6),
            A.PixelDropout(dropout_prob=0.1, per_channel=True, p=0.5),
            A.GridDropout(ratio=0.3, random_offset=True, p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 5),
                hole_height_range=(0.1, 0.12),
                hole_width_range=(0.1, 0.12),
                fill="random_uniform", p=0.9,
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["category"],
            clip=True, min_area=25, min_width=1,
        ),
    )


def build_valid_transform() -> A.Compose:
    return A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["category"],
            clip=True, min_area=1,
        ),
    )


def save_split(ds_split, split_name, output_path, classes, aumento_factor=1, transform=None):
    """Guarda un split de sv.DetectionDataset como carpeta COCO.

    Nota: a diferencia de la version del notebook, aqui cada imagen
    aumentada recibe su propio image_id consistente entre "images" y
    "annotations" (en el notebook original las anotaciones de las
    copias aumentadas quedaban apuntando al image_id de la imagen
    original, lo cual generaba anotaciones duplicadas/mal referenciadas
    en el json final).
    """
    split_dir = os.path.join(output_path, split_name)
    os.makedirs(split_dir, exist_ok=True)

    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i, "name": name, "supercategory": "2"}
            for i, name in enumerate(classes)
        ],
        "info": {
            "year": "2025",
            "version": "1",
            "description": "Generado por prepare_dataset.py",
            "contributor": "",
            "url": "",
            "date_created": "",
        },
    }

    ann_id = 1
    next_image_id = 0

    for local_idx, (img_path, _dets) in enumerate(
        tqdm(ds_split.annotations.items(), desc=f"Procesando {split_name}")
    ):
        _, image, annotations = ds_split[local_idx]
        filename = os.path.basename(img_path)
        dst_path = os.path.join(split_dir, filename)
        if not os.path.exists(dst_path):
            shutil.copy(img_path, dst_path)

        image_id = next_image_id
        next_image_id += 1

        coco_dict["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": image.shape[1],
            "height": image.shape[0],
        })

        for box, cls_id in zip(annotations.xyxy, annotations.class_id):
            x_min, y_min, x_max, y_max = [int(v) for v in box.tolist()]
            coco_dict["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(cls_id),
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "iscrowd": 0,
            })
            ann_id += 1

        # Copias aumentadas (solo tiene sentido para train)
        if aumento_factor > 1 and transform is not None:
            for aug_idx in range(aumento_factor - 1):
                image_rgb = image[:, :, ::-1]
                transformed = transform(
                    image=image_rgb,
                    bboxes=annotations.xyxy,
                    category=annotations.class_id,
                )
                aug_image = transformed["image"]
                aug_annotations = replace(
                    annotations,
                    xyxy=np.array(transformed["bboxes"]),
                    class_id=np.array(transformed["category"]),
                )

                aug_filename = f"{filename}_{aug_idx}.jpg"
                aug_dst_path = os.path.join(split_dir, aug_filename)
                cv2.imwrite(aug_dst_path, cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB))

                aug_image_id = next_image_id
                next_image_id += 1

                coco_dict["images"].append({
                    "id": aug_image_id,
                    "file_name": aug_filename,
                    "width": aug_image.shape[1],
                    "height": aug_image.shape[0],
                })

                for box, cls_id in zip(aug_annotations.xyxy, aug_annotations.class_id):
                    x_min, y_min, x_max, y_max = [int(v) for v in box.tolist()]
                    coco_dict["annotations"].append({
                        "id": ann_id,
                        "image_id": aug_image_id,
                        "category_id": int(cls_id),
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "area": (x_max - x_min) * (y_max - y_min),
                        "iscrowd": 0,
                    })
                    ann_id += 1

    with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
        json.dump(coco_dict, f, indent=4)

    print(f"[{split_name}] {len(coco_dict['images'])} imagenes, "
          f"{len(coco_dict['annotations'])} anotaciones -> {split_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepara un dataset COCO para entrenar RF-DETR")
    parser.add_argument("--base-path", required=True, help="Carpeta con las imagenes originales")
    parser.add_argument("--annotations", required=True, help="Ruta al json COCO con las anotaciones")
    parser.add_argument("--output", required=True, help="Carpeta destino (train/valid/test se crean adentro)")
    parser.add_argument("--split-ratio", type=float, default=0.7, help="Proporcion para train (resto va a valid/test)")
    parser.add_argument("--include-test", action="store_true", help="Si se pasa, separa un split de test desde valid")
    parser.add_argument("--test-ratio", type=float, default=0.95, help="Proporcion de valid que se queda como valid (el resto pasa a test)")
    parser.add_argument("--aumento-factor", type=int, default=1, help="1 = sin copias extra. N>1 = N-1 copias aumentadas por imagen de train")
    parser.add_argument("--seed", type=int, default=None, help="Semilla opcional para el shuffle del split (reproducibilidad)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    print(f"Cargando dataset COCO desde {args.base_path} / {args.annotations}")
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=args.base_path,
        annotations_path=args.annotations,
    )
    print(f"Total de imagenes: {len(ds)}")

    ds_train, ds_valid = ds.split(split_ratio=args.split_ratio, shuffle=True)
    ds_test = None
    if args.include_test:
        ds_valid, ds_test = ds_valid.split(split_ratio=args.test_ratio, shuffle=True)

    print(f"train={len(ds_train)}  valid={len(ds_valid)}"
          + (f"  test={len(ds_test)}" if ds_test else ""))

    os.makedirs(args.output, exist_ok=True)

    train_transform = build_train_transform()
    valid_transform = build_valid_transform()

    save_split(ds_train, "train", args.output, ds_train.classes,
               aumento_factor=args.aumento_factor, transform=train_transform)
    save_split(ds_valid, "valid", args.output, ds_train.classes,
               aumento_factor=1, transform=valid_transform)
    if ds_test:
        save_split(ds_test, "test", args.output, ds_train.classes,
                   aumento_factor=1, transform=valid_transform)

    print(f"\nDataset listo en: {args.output}")


if __name__ == "__main__":
    main()
