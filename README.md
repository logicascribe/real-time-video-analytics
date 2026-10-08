# Real-Time Video Analytics – Object Detection and Tracking

Personal project exploring practical real-time video analytics with Python,
OpenCV and a pretrained object detector.

The current implementation detects road users in video, associates detections
over time with a lightweight IoU-based multi-object tracker, displays persistent
track IDs, and collects tracking-quality and performance statistics.

The project is intentionally developed incrementally: the current tracker is
kept simple enough to study its behavior and limitations before adding motion
prediction and more advanced event analytics.

## Current Features

- Video file processing with OpenCV
- Pretrained YOLO object detection
- Detection of selected COCO classes:
  - person
  - bicycle
  - car
  - motorcycle
  - bus
  - truck
- Bounding-box visualization
- IoU-based multi-object tracking
- Optimal one-to-one detection/track association
- Configurable temporal track confirmation
- Track lifecycle:
  - `Tentative`
  - `Tracked`
  - `ToDelete`
- Per-track trajectory recording using a ground-contact point derived from the
  bottom center of the bounding box
- Tracking statistics:
  - total tracks created
  - confirmed tracks
  - track duration
  - median track duration
  - number of detection hits
  - hit distribution
  - very short tracks
  - maximum number of simultaneous tracks
- Basic processing/inference latency measurements
- Parameter sweeps for tracking-window and IoU-threshold evaluation

## Processing Pipeline

```text
   Video frame
        |
        v
  YOLO inference
        |
        v
Detection filtering
(person / vehicle classes)
        |
        v
    IoU matrix
        |
        v
Optimal one-to-one
   assignment
        |
        v
Track lifecycle update
        |
        +---- unmatched detection ---> new tentative track
        |
        +---- unmatched track -------> miss / possible deletion
        |
        v
Trajectory + statistics
        |
        v
OpenCV visualization

```

## Tracking Approach

Each detection is represented by a bounding box:

```text
(x1, y1, x2, y2)
```

Association between existing tracks and current detections is currently based
on Intersection over Union (IoU).

For two bounding boxes A and B:

```text
IoU(A, B) = area(A intersection B) / area(A union B)
```

A score matrix is built between existing tracks and detections. A global
one-to-one assignment is then computed instead of greedily selecting the
largest IoU first.

Only associations above a configurable IoU threshold are accepted.

### Track lifecycle

A track maintains a sliding history of successful and missed associations.

Example configuration:

```text
tracking window : 10 frames
minimum hits    : 7
maximum misses  : 10
IoU threshold   : 0.25
```

A new track starts as `Tentative`.

It becomes `Tracked` after accumulating enough successful associations inside
the temporal window. Tracks that accumulate too many misses are marked
`ToDelete` and archived for statistics.

This confirmation stage prevents isolated detections from immediately becoming
persistent tracks.

## Trajectories

For each matched bounding box, a ground point is derived from the center of its
bottom edge:

```text
           bounding box
        +---------------+
        |               |
        |    object     |
        |               |
        +-------o-------+
                ^
           ground point
```

This point is more useful than the box center for future scene-level analytics,
including:

- trajectory visualization
- line crossing
- zone entry/exit
- direction estimation
- ground-plane projection

## Preliminary Tracking Experiments

Tracking parameters were evaluated on the same 447-frame traffic sequence at
two input resolutions.

Reference configuration:

```text
IoU threshold       = 0.25
tracking window     = 10
minimum hits        = 7
maximum misses      = 10
```

| Metric | 480x270 | 640x360 |
|---|---:|---:|
| Frames | 447 | 447 |
| Tracks created | 46 | 60 |
| Tracks confirmed | 28 | 26 |
| Very short tracks (<= 2 hits) | 15 | 15 |
| Mean track duration (frames) | 83.17 | 69.55 |
| Median track duration (frames) | 31.5 | 22.5 |
| Maximum simultaneous tracks | 14 | 12 |

For the 640x360 sequence, the median number of detection hits per track was
7 frames.

These measurements are preliminary and are not presented as tracking accuracy
metrics. They are primarily used to study tracker fragmentation and parameter
sensitivity.

An interesting observation is that increasing the input resolution did not
automatically produce more stable tracks: more tracks were created at 640x360,
while their median duration decreased.

Further visual analysis is required to distinguish between:

- additional useful detections of small/distant objects;
- intermittent detections;
- fragmentation of existing tracks;
- incorrect identity associations.

## Detection and Tracking Limitations

The current tracker deliberately uses a simple geometric association model.

### No motion model

A track is associated using its last observed bounding box.

After several missed detections, a moving object may have moved far enough that:

```text
IoU(previous_box, new_detection) ~= 0
```

even though both boxes correspond to the same physical object.

Increasing the track lifetime can keep the track alive, but cannot solve this
geometric association problem.

Motion prediction is therefore the main planned tracking improvement.

### Classification instability

Detection class is currently treated as an observed property rather than a hard
association constraint.

For example, a small or distant cyclist may be detected intermittently as:

```text
person -> motorcycle -> person
```

A strict class-equality constraint would unnecessarily fragment such a track.

### Identity switches

Global statistics such as track count and duration cannot detect all identity
switches. Visual evaluation of selected long-lived and fragmented tracks is
therefore part of the ongoing validation work.

## Performance

The pipeline includes timing instrumentation for:

- inference latency
- processing latency
- end-to-end frame processing
- processing FPS

On the current development machine equipped with an NVIDIA GeForce RTX 2070 (8192 MB), an earlier 480x270 benchmark with YOLO26n measured approximately:

```text
Mean inference latency   : 6.7 ms/frame
Mean processing latency  : 7.2 ms/frame
Processing throughput    : ~139 FPS
Video source rate        : ~59.5 FPS
```

These measurements are specific to this hardware, model and input resolution
and should not be interpreted as general YOLO performance figures.

## Running the Project

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages.

The main dependencies currently include:

```text
OpenCV
Ultralytics
NumPy
SciPy
more-itertools
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

Run the analytics pipeline:

```bash
python3 run_video_analytics.py \
    --video_source path/to/video.mp4
```

Run without visualization, for example for benchmarking:

```bash
python3 run_video_analytics.py \
    --video_source path/to/video.mp4 \
    --no_display
```

The pretrained YOLO model weights are loaded by Ultralytics.

## Project Structure

A simplified view of the project is:

```text
AxisVideoAnalytics/
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── run_video_analytics.py
├── tracker.py
├── IoUTools/
│   ├── __init__.py
│   └── compute_iou.py
```

## Validation

The core algorithms have been exercised with deterministic synthetic test cases,
including:

- IoU computation
- one-to-one association
- optimal assignment
- track creation and confirmation
- missed detections
- track deletion
- simultaneous tracks

Synthetic sequences are used to make tracker behavior easy to inspect before
testing on real video.

A structured automated test suite is planned as part of the next cleanup step.

## Sample data

The test videos used for the experiments are not distributed with the repository.

The current benchmarks were performed on derived 480x270 and 640x360 versions
of the same 447-frame road-traffic sequence.

Any compatible video file can be provided with:

```bash
python3 run_video_analytics.py --video_source path/to/video.mp4

## Architecture Evolution Under Consideration

The current implementation intentionally keeps the processing pipeline simple
while the detection and tracking algorithms are being evaluated.

As the project evolves, the architecture is expected to separate the main
responsibilities more clearly:

```text
        Video Source
             |
             v
        Frame Pipeline
             |
             v
          Detector
             |
             v
     Detection Filtering
             |
             v
          Tracker
       /           \
      v             v
 Trajectories    Track States
      \             /
       v           v
        Event Engine
             |
             v
      Events / Metrics
             |
             v
       Visualization
```
Several architectural evolutions are currently being considered:

### Detector abstraction
  Isolate the object-detection backend from the rest of the pipeline. The
  current Ultralytics implementation could later be replaced by an ONNX,
  TensorFlow Lite or other edge-oriented inference backend without changing the
  tracking and event-processing logic.
### Association strategy abstraction
  Separate track lifecycle management from the detection-to-track association
  algorithm. This would allow the current IoU association to evolve toward
  center-distance gating, motion prediction or Kalman-based association.
### Event-processing layer
  Make line crossing, zone entry/exit and similar analytics consume confirmed
  tracks and trajectories rather than embedding event logic directly into the
  tracker.
### Runtime / evaluation separation
  Separate real-time video processing from benchmarking and tracking-quality
  analysis. The same detector and tracker implementations could then be used
  both by the runtime application and by reproducible evaluation tools.
### Explicit data models
  Progressively replace loosely structured detection data with explicit
  Detection, Track and Event data models, making interfaces easier to
  validate and test.
```text
Detection -> Tracking -> Events
```
### Configuration separation
  Move detector, tracker and event parameters out of the main application code
  into explicit runtime configuration.

These changes are architectural directions rather than features already
implemented in the current prototype.


## Roadmap

Short-term improvements:

- Add center-distance information to track association
- Add constant-velocity motion prediction
- Evaluate Kalman filtering
- Perform targeted visual evaluation of track fragmentation and ID switches
- Add line-crossing event detection
- Add ROI entry/exit events
- Improve automated tracking metrics

Possible later extensions:

- Ground-plane homography / bird's-eye-view projection
- Speed and direction estimation
- C++17 / OpenCV implementation
- ONNX or TensorFlow Lite inference
- Edge-oriented performance experiments

## Design Goals

This project is primarily an engineering exercise in implementing and evaluating
an end-to-end video-analytics pipeline rather than training a neural network
from scratch.

The focus is on:

- integration of pretrained detection models;
- temporal association and tracking;
- quantitative evaluation;
- performance measurement;
- explicit analysis of failure modes;
- progressive transition from prototype code toward production-oriented
  software design.

## Status

**Work in progress.**

The current version provides a functional detection and IoU-tracking pipeline.
Tracking robustness, motion prediction and event detection are active next
steps.

