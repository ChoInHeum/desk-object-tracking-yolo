import shutil

from ultralytics import YOLO

from utility import log
from config import (EXPERIMENT, CLASSES, IMG_SIZE, RUN_NAME, DATA_YAML,
                    RUNS_PATH, RESULTS_PATH, TRAIN_RUN_PATH, BEST_WEIGHTS)

# constant
BATCH = 16
CONF_DISPLAY = 0.25
EVAL_NAME = f"{RUN_NAME}-eval"
EVAL_CONF_NAME = f"{RUN_NAME}-eval-conf{int(CONF_DISPLAY * 100)}"

def evaluate(conf: float | None, name: str):
    model = YOLO(BEST_WEIGHTS)
    return model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=IMG_SIZE,
        batch=BATCH,
        conf=conf,
        device=0,
        project=str(RUNS_PATH),
        name=name,
        exist_ok=True,
    )

def report(metrics) -> None:
    box = metrics.box
    log(f"mAP50    : {box.map50:.3f}")
    log(f"mAP50-95 : {box.map:.3f}")
    log(f"{'class':13s} {'P':>6s} {'R':>6s} {'AP50':>6s} {'AP50-95':>8s}")
    for i, class_ in enumerate(CLASSES):
        p, r, ap50, ap = box.class_result(i)
        log(f"{class_:13s} {p:6.3f} {r:6.3f} {ap50:6.3f} {ap:8.3f}")

def copy_results() -> None:
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    files = [
        (TRAIN_RUN_PATH / "results.png", "results.png"),
        (RUNS_PATH / EVAL_NAME / "BoxPR_curve.png", "PR_curve.png"),
        (RUNS_PATH / EVAL_CONF_NAME / "confusion_matrix.png", "confusion_matrix.png"),
    ]
    for src, dst_name in files:
        shutil.copy2(src, RESULTS_PATH / dst_name)
        log(f"copied {src.name} -> results/{EXPERIMENT}/{dst_name}")

def main():
    log(f"experiment: {EXPERIMENT}")
    metrics = evaluate(conf=None, name=EVAL_NAME)
    report(metrics)

    evaluate(conf=CONF_DISPLAY, name=EVAL_CONF_NAME)
    copy_results()

if __name__ == "__main__":
    main()