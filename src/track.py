import cv2
import imageio_ffmpeg

from ultralytics import YOLO

from utility import log
from config import (EXPERIMENT, CLASSES, TRACK_IMG_SIZE, BEST_WEIGHTS,
                    DEMO_PATH, SOURCE_VIDEO, DEMO_VIDEO, TRACKING_VIDEO)

# constant
CONF = 0.25
TRACKER = "bytetrack.yaml"
OUT_LONG_SIDE = 1280    # 1920x1080 -> 1280x720, 1080x1920 -> 720x1280
BLOCK = 16              # H.264 macroblock: 가로·세로를 16의 배수로 맞춤

# function
def video_info(path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

def output_size(width: int, height: int) -> tuple[int, int]:
    scale = OUT_LONG_SIDE / max(width, height)
    return (round(width * scale / BLOCK) * BLOCK, round(height * scale / BLOCK) * BLOCK)

def open_writer(path, size: tuple[int, int], fps: float):
    writer = imageio_ffmpeg.write_frames(
        str(path), size, fps=fps, codec="libx264", pix_fmt_in="bgr24"
    )
    writer.send(None)
    return writer

def track() -> dict[str, set[int]]:
    width, height, fps = video_info(SOURCE_VIDEO)
    size = output_size(width, height)

    DEMO_PATH.mkdir(parents=True, exist_ok=True)
    demo_writer = open_writer(DEMO_VIDEO, size, fps)
    track_writer = open_writer(TRACKING_VIDEO, size, fps)

    model = YOLO(BEST_WEIGHTS)
    results = model.track(
        source=str(SOURCE_VIDEO),
        conf=CONF,
        imgsz=TRACK_IMG_SIZE,
        tracker=TRACKER,
        device=0,
        stream=True,
        verbose=False,
    )

    track_ids = {class_: set() for class_ in CLASSES}
    for r in results:
        demo_writer.send(cv2.resize(r.orig_img, size))
        track_writer.send(cv2.resize(r.plot(color_mode="instance"), size))

        if r.boxes.id is not None:
            for cls, id_ in zip(r.boxes.cls.int().tolist(), r.boxes.id.int().tolist()):
                track_ids[CLASSES[cls]].add(id_)

    demo_writer.close()
    track_writer.close()
    return track_ids

def main():
    log(f"experiment: {EXPERIMENT}, source: {SOURCE_VIDEO.name}")
    track_ids = track()
    for class_, ids in track_ids.items():
        log(f"{class_:14s} ids={sorted(ids)}")
    log(f"saved demo/{EXPERIMENT}/{DEMO_VIDEO.name}, demo/{EXPERIMENT}/{TRACKING_VIDEO.name}")

if __name__ == "__main__":
    main()
