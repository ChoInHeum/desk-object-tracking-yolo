import time
import shutil

from pathlib import Path
from ultralytics import YOLO

from utility import log
from config import EXPERIMENT, RANDOM_SEED, IMG_SIZE, RUN_NAME, DATA_YAML, RUNS_PATH, WEIGHTS_PATH, BEST_WEIGHTS

# constant
MODEL = "yolo11n.pt"
EPOCHS = 50
BATCH = 16
TEST_RUN = False

# function
def train(epochs: int, name: str) -> Path:
    model = YOLO(MODEL)

    start = time.perf_counter()
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=0,
        seed=RANDOM_SEED,
        project=str(RUNS_PATH),
        name=name,
        exist_ok=True,
    )
    minutes = (time.perf_counter() - start) / 60

    log(f"training time: {minutes:.1f} min")
    log(f"results: {model.trainer.save_dir}")

    return Path(model.trainer.best)

def save_weights(best: Path) -> None:
    WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, BEST_WEIGHTS)
    log(f"copied {best} -> {BEST_WEIGHTS}")

def main():
    log(f"experiment: {EXPERIMENT}")
    if TEST_RUN:
        train(epochs=1, name=f"{RUN_NAME}-test")
        return

    best = train(epochs=EPOCHS, name=RUN_NAME)
    save_weights(best)

if __name__ == "__main__":
    main()