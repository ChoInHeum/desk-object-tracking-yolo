import os

from pathlib import Path

# experiment
# 실험마다 부류가 다르고, 결과는 실험 이름 폴더에 따로 저장된다 (시행착오 비교용)
EXPERIMENTS = {
    "bottle": ["Laptop", "Mobile phone", "Bottle"],         # v1: 투명 생수병 감지 실패
    "mouse": ["Laptop", "Mobile phone", "Computer mouse"],  # v2: 물병 -> 마우스로 교체
}
SOURCE_VIDEOS = {   # demo/raw/ 안의 촬영 원본
    "bottle": "IMG_0877.mp4",
    "mouse": "IMG_0880.mp4",    # 마우스 영상을 새로 찍으면 파일 이름 변경
}
EXPERIMENT = os.environ.get("EXPERIMENT", "mouse")  # 예: EXPERIMENT=bottle python src/evaluate.py

# constant
CLASSES = EXPERIMENTS[EXPERIMENT]
RANDOM_SEED = 42
IMG_SIZE = 640          # 학습·평가 입력 크기
TRACK_IMG_SIZE = 1280   # 추적 입력 크기 (멀리 있는 작은 물체도 잡히도록)
RUN_NAME = f"desk-objects-yolo11n-{EXPERIMENT}"

# directory
ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / "data"
RAW_PATH = DATA_PATH / "raw"                    # Open Images 캐시 (실험 공용)
YOLO_PATH = DATA_PATH / "yolo" / EXPERIMENT
RUNS_PATH = ROOT_PATH / "runs"
TRAIN_RUN_PATH = RUNS_PATH / RUN_NAME
WEIGHTS_PATH = ROOT_PATH / "weights" / EXPERIMENT
RESULTS_PATH = ROOT_PATH / "results" / EXPERIMENT
DEMO_PATH = ROOT_PATH / "demo" / EXPERIMENT
VIDEO_RAW_PATH = ROOT_PATH / "demo" / "raw"     # 촬영 원본 (용량이 커서 커밋 안 함)

# file
DATA_YAML = YOLO_PATH / "dataset.yaml"
BEST_WEIGHTS = WEIGHTS_PATH / "best.pt"
SOURCE_VIDEO = VIDEO_RAW_PATH / SOURCE_VIDEOS[EXPERIMENT]
DEMO_VIDEO = DEMO_PATH / "demo.mp4"
TRACKING_VIDEO = DEMO_PATH / "tracking.mp4"
