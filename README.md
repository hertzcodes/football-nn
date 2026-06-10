### TEAM
```
Mohammadreza RasouliSadr [40217693]
Parham Moafi [40221603]
Radmehr Soleimanian [40218473]
```


# Football Analytics & Tactical Minimap Generator

An end-to-end computer vision pipeline that processes football broadcast videos to track players, referees, and the ball, assign teams automatically based on jersey colors, and generate a dynamic top-down tactical minimap.

## Features

- **Object Tracking**: Uses YOLO for player, goalkeeper, referee, and ball detection, integrated with ByteTrack to maintain stable IDs across occlusion and frames.
- **Dynamic Team Assignment**: Automatically clusters players into two teams using KMeans clustering on extracted jersey HSV color features. Team colors are dynamically calculated back to BGR from the cluster centers, matching the actual color of their jerseys.
- **Pitch Homography**: Maps broadcast camera pixel coordinates to a top-down standard pitch (1050x680) using a YOLO pose model to detect keypoints and RANSAC to solve the homography matrix.
- **Keyframe Interpolation & Flow Compensation**: Recomputes homographies at regular keyframe intervals and interpolates between them to prevent minimap position jumps. Compensates camera pan, tilt, and zoom frame-by-frame via sparse Lucas-Kanade optical flow on grass-field features.
- **Modular Architecture**: Clean separation between tracking (detection/linking), mathematical transformers (homography), and drawing/visualization (renderer).

---

## Directory Structure

- **`cmd/`**: Contains execution entries for the CLI (`analyze.py` for pipeline execution, `train.py` for training).
- **`homography/`**: Handles pitch-to-camera matrix transformations, RANSAC solver, and camera tracking updates.
- **`renderer/`**: Responsible for creating the minimap canvas and drawing all graphical overlays (ellipse indicators, ball triangles, bird's eye view).
- **`assigner/`**: Implements HSV jersey feature extraction and KMeans clustering.
- **`tracking/`**: Contains YOLO detection code for player tracking and pitch keypoint tracking.
- **`utils/`**: Helper methods for video reading and writing.
- **`tests/`**: Unit test suite verifying geometry, assigner clustering, and homography logic.

---

## Usage

### Configuration
All pipeline variables, models, input video paths, and fallback colors are configured in `config.py`.

### Requirements
```bash
python -m venv .venv

windows: .venv\Scripts\activate
unix: source .venv/bin/activate

pip install -r requirements.txt
```

### Running Analysis
To process a video and generate the annotated tactical video:
```bash
python main.py analyze
```
This runs the full tracking, jersey clustering, homography estimation, and renderer pipeline. The resulting video is saved to the path specified in `config.py` (default: `output_video.avi`).

### Training Models
To train the player detection model:
```bash
python main.py train
```

---

## Running Tests

Unit tests are written from scratch using `pytest` to verify the mathematical and logical structures of the project. Run them using:
```bash
.venv/bin/pytest tests/
```