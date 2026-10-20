# AI Detection Prototype

This project is the initial AI detection prototype intended for later integration into a larger Security & Intrusion module.

## Project purpose

The goal of this prototype is to create a reusable detection component that:

- reads a video source
- passes frames to a pretrained YOLO model
- detects supported object classes
- draws boxes and labels on the video
- records detections as structured JSON
- keeps the detector separate from the video input source so a future CCTV or RTSP stream can be used without rewriting the AI logic

## Technologies used

### Python

Python is the main programming language for the application. It allows the project to stay simple, readable, and easy to extend for a beginner-friendly prototype.

### OpenCV

OpenCV is used to work with video frames, read videos, draw boxes, show the processed output, and manage playback.

### YOLO

YOLO is a fast object-detection model family used for identifying objects like people and vehicles. The project is designed to use a lightweight pretrained model such as YOLO11n in the later implementation stage.

### JSON

JSON is used to store detection records in a structured format so the results can be reviewed later, logged, or consumed by another system.

## Folder structure

```text
AI Detection/
├── input/
│   └── .gitkeep
├── output/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── video/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── file_source.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── models.py
│   ├── display/
│   │   ├── __init__.py
│   │   └── renderer.py
│   ├── logging/
│   │   ├── __init__.py
│   │   └── json_logger.py
│   └── application.py
├── tests/
│   ├── __init__.py
│   ├── test_detection.py
│   ├── test_video_source.py
│   └── test_json_logger.py
├── .gitignore
├── DEVELOPMENT_NOTES.md
├── README.md
├── requirements.txt
└── main.py
```

The structure is intentionally simple and modular so the AI detection logic can later be extended without coupling it to a specific file or camera input.

## Step-by-step: how to run the application

Follow these steps in PowerShell from the project folder.

### 1. Open PowerShell in the project folder

```powershell
cd "C:\Users\info\OneDrive\Desktop\AI Detection"
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run this first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

### 4. Install the project dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs OpenCV, Ultralytics, NumPy, pytest, and the other packages required by the app.

### 5. Add a video file to the input folder

Put your video in the `input` folder. For example:

```text
input\objects.mp4
```

The application expects a real video file such as `.mp4`.

### 6. Run the app

Use the source argument to point to your video:

```powershell
python main.py --source input/objects.mp4
```

You can also use the default YOLO model and confidence threshold without extra arguments:

```powershell
python main.py --source input/objects.mp4 --model yolo11n.pt --confidence 0.25
```

### 7. What happens while the app is running

- the video opens in a window
- the YOLO model analyzes each frame
- detected objects are boxed and labeled
- the app displays the annotated video
- the program continues until you press `q` to quit

### 8. Output files

When the video ends or you press `q`, the app saves the detection log to:

```text
output\detections.json
```

This JSON file includes:

- timestamp
- object class
- confidence score
- bounding box coordinates
- source path

### 9. Optional custom output path

You can write the results to a different file if needed:

```powershell
python main.py --source input/objects.mp4 --output output/my_detections.json
```

### 10. First run note

On the first run, Ultralytics may download the YOLO model automatically if it is not available locally. This is normal and expected.

## Running the application

The application is designed so the source video can be changed without rewriting the detector code. In other words, you only need to change the `--source` path to analyze a different file.

Example:

```powershell
python main.py --source input/your_video.mp4
```

## Future detection pipeline

The intended later architecture is:

```text
Video source
  ↓
Frames
  ↓
YOLO detector
  ↓
Detection results
  ↓
OpenCV display
  ↓
JSON logging
```

The architecture will keep these concerns separate:

- video input layer
- detection logic
- display/rendering
- JSON logging

This separation is important because the final system must be able to work with either a local video file or a live CCTV/RTSP stream without changing the core AI detection logic.

## Output and logging

In the completed prototype, the output will include:

- video window showing bounding boxes
- object labels near each detected object
- confidence values displayed for each detection
- a JSON file saved under the output folder

The JSON output is intended to store:

- timestamp
- object class name
- confidence score
- bounding box coordinates
- source information

## Future integration

The current video abstraction will be designed so that a future CCTV or RTSP source can replace the local video file source without changing the detection logic.

The key idea is:

- the detector receives frames
- the detector does not care whether the frame came from a file, a stream, or a camera
- a future `RTSPVideoSource` or similar source can be plugged in cleanly

This is the correct architectural approach for the eventual Security & Intrusion module.

## Notes for this stage

This repository is intentionally set up in a clean, maintainable structure before the AI detection logic is added. This reduces complexity, keeps the code easy to learn, and makes it easier to add the actual detection system in a controlled next step.
