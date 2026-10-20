# Development Notes

## 1. How OpenCV reads a video

OpenCV is used to open a video file and read it one frame at a time. A video is not a single image; it is a sequence of images called frames. Each frame is like one still image in a film.

In the later implementation, OpenCV will:

- open the video file
- read the next frame
- pass that frame to the model
- repeat until the end of the video is reached

This is important because object detection works on frames, not on the entire video as one object.

## 2. How a frame is passed to YOLO

After OpenCV reads a frame, the frame is passed into the detector. The detector then sends that frame to a pretrained YOLO model. The model looks at the image and tries to find objects.

A simple way to think about it is:

```text
frame -> YOLO model -> list of detected objects
```

The model returns multiple detection results, each one describing an object it believes it has found.

## 3. How YOLO returns detections

YOLO does not return a single answer; it returns a list of results. Each result usually includes:

- object class name
- confidence score
- bounding box coordinates

That means the detector can decide which objects are relevant and filter out low-confidence results.

## 4. How bounding boxes are obtained

A bounding box is a rectangle drawn around a detected object. The model returns a box in terms of coordinates such as:

- x1
- y1
- x2
- y2

These values define the rectangle corners. OpenCV then draws the box over the frame so the user can see where the object is located.

## 5. How confidence scores work

Confidence is a number between 0 and 1 that tells how certain the model is that it found the right object.

Example:

- 0.91 means 91% confidence
- 0.72 means 72% confidence

In the real system, a confidence threshold is used so only detections above a chosen value are accepted and displayed.

## 6. How object classes are identified

The YOLO model knows many common classes such as:

- person
- car
- motorcycle
- bus
- truck

The project will keep a configurable set of supported classes. This means the detector can ignore objects that are not needed for the current detection task.

This is useful because the final system may only care about people and vehicles, not every object the model happens to know.

## 7. How timestamps are obtained

This is a very important requirement for the project: the timestamp must reflect the actual position of the object in the video, not the computer's current clock time.

OpenCV gives the current video position in time via the video frame timestamp. This is not the wall-clock time from the computer. It is the time within the playback sequence. For example:

```text
00:00:02.450 -> person
00:00:04.120 -> car
```

This matters because later, when real CCTV or RTSP streams are supported, the system can adapt to stream timing while still keeping a consistent detection record.

## 8. How detections are converted into JSON

Each detection result is converted into a record with fields such as:

- timestamp
- object class
- confidence
- bounding box coordinates

The JSON logger writes these records into a file in the output folder. This makes detections easy to review, debug, and reuse in later systems.

## 9. How the video source is separated from the detector

The code is designed so that the detector receives frames instead of directly depending on a local file. In other words, the AI logic should not know whether the input is:

- a local MP4 file
- an RTSP stream
- a CCTV feed

This separation is important for future integration. The same detector can work with different video sources by changing only the source layer.

## 10. How this component can later connect to Security & Intrusion

This prototype is intentionally focused only on producing clean detection objects. It does not include restricted areas, alerts, security rules, incidents, or database logic.

Later, the Security & Intrusion system can consume the detection results and apply logic such as:

- zone checks
- rule matching
- alarms
- event generation
- incident handling

The key purpose of this stage is to produce high-quality detection records that can be reused by a larger system without reworking the core AI pipeline.

## Project status

This repository is currently in Stage 1 of development. The project foundation, dependency setup, and beginner-friendly documentation have been created. The actual AI detection implementation will be added in a later stage after review and confirmation.
