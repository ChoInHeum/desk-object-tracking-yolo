import shutil
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.random as four

from fiftyone import ViewField as F

from utility import log
from config import EXPERIMENT, CLASSES, RANDOM_SEED, RAW_PATH, YOLO_PATH

# constant
N_PER_CLASS = 250
VAL_RATIO = 0.2

# functions
def download() -> fo.Dataset:
    fo.config.dataset_zoo_dir = str(RAW_PATH)

    dataset = fo.Dataset(name=f"desk-objects-{EXPERIMENT}", overwrite=True)

    for class_ in CLASSES:
        part = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=[class_],
            max_samples=N_PER_CLASS,
            shuffle=True,
            seed=RANDOM_SEED,
            dataset_name=f"open-images-{class_}",
            drop_existing_dataset=True,
        )
        already = dataset.values("filepath")
        added = dataset.add_samples(part.match(~F("filepath").is_in(already)))
        log(f"{class_}: download {part.count()}, added {len(added)}")

    log(f"total images: {dataset.count()}")
    return dataset

def keep_target_boxes(dataset: fo.Dataset) -> fo.DatasetView:
    return dataset.filter_labels("ground_truth", F("label").is_in(CLASSES))

def report(view: fo.DatasetView) -> None:
    log("images / boxes per class")
    for class_ in CLASSES:
        n_image = view.match(F("ground_truth.detections.label").contains(class_)).count()
        n_box = view.filter_labels("ground_truth", F("label") == class_).count("ground_truth.detections")
        log(f" {class_:13s} images={n_image:4d} boxes={n_box:5d}")
    log(f"total images: {view.count()}")

def export_yolo(view: fo.DatasetView) -> None:
    shutil.rmtree(YOLO_PATH, ignore_errors=True)

    four.random_split(view, {"train": 1 - VAL_RATIO, "val": VAL_RATIO}, seed=RANDOM_SEED)

    for split in ["train", "val"]:
        split_view = view.match_tags(split)
        log(f"{split}: {split_view.count()} images")
        split_view.export(
            export_dir=str(YOLO_PATH),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            split=split,
            classes=CLASSES,
        )

def main():
    log(f"experiment: {EXPERIMENT} {CLASSES}")
    dataset = download()
    view = keep_target_boxes(dataset)
    report(view)
    export_yolo(view)
    log(f"exported to {YOLO_PATH}")

if __name__ == "__main__":
    main()