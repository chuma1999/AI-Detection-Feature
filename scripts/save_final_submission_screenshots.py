from pathlib import Path
import cv2
from ultralytics import YOLO

model = YOLO('models/fine_tuned/best.pt')
out_dir = Path('output/final_submission_images')
out_dir.mkdir(parents=True, exist_ok=True)


def save_frame(video_path, frame_idx, filename):
    cap = cv2.VideoCapture(str(video_path))
    frame_num = 0
    target = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_num == frame_idx:
            target = frame
            break
        frame_num += 1
    cap.release()
    if target is None:
        raise RuntimeError(f'Could not read frame {frame_idx} from {video_path}')

    res = model(target, conf=0.12, imgsz=1280, verbose=False)
    if not res or len(res) == 0:
        raise RuntimeError(f'No detections for {filename}')

    annotated = res[0].plot()
    path = out_dir / filename
    ok = cv2.imwrite(str(path), annotated)
    if not ok:
        raise RuntimeError(f'Failed to write {path}')
    print(f'Saved {filename} -> {path}')


def save_image(image_path, filename):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f'Could not read image {image_path}')
    res = model(img, conf=0.12, imgsz=1280, verbose=False)
    if not res or len(res) == 0:
        raise RuntimeError(f'No detections for {filename}')
    annotated = res[0].plot()
    path = out_dir / filename
    ok = cv2.imwrite(str(path), annotated)
    if not ok:
        raise RuntimeError(f'Failed to write {path}')
    print(f'Saved {filename} -> {path}')


save_frame('input/person.mp4', 160, 'single_person_detection.png')
save_frame('input/car.mp4', 120, 'single_car_detection.png')
save_frame('input/people_vehicles.mp4', 120, 'multi_object_scene.png')
save_image('input/test_frames/test-image-1.jpg', 'distant_object_detection.png')
save_frame('input/real_demo.mp4', 4, 'partial_occlusion_detection.png')
save_frame('input/people_vehicles.mp4', 260, 'full_class_set_detection.png')

print(f'All screenshots were saved to {out_dir}')
