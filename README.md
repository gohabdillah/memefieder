# Memefieder

Memefieder is a real-time pose-to-meme matcher for IoT coursework. It uses a laptop webcam and MediaPipe Holistic to extract body, hand, and selected face keypoints, classifies poses locally, and overlays a matching meme image on the live video feed.

When local confidence is low, the client sends keypoints to a cloud Flask service for a second opinion.

## Features

- Real-time holistic keypoint extraction (pose + hands + face)
- Default 333-dimensional feature vector (pose 99 + hands 126 + selected face 108)
- Local RandomForest pose classification
- Cloud fallback inference when local confidence is below threshold
- On-device data collection for labeled pose samples
- Cloud correction endpoint with automatic retraining every 20 corrections

## Project Structure

```text
memefieder/
├── README.md
├── requirements.txt
├── data/
├── memes/
├── models/
├── src/
│   ├── collect_data.py
│   ├── feature_config.py
│   ├── train.py
│   ├── local_inference.py
│   └── utils.py
├── cloud/
│   ├── app.py
│   ├── cloud_train.py
│   └── requirements.txt
└── docs/
    ├── architecture.md
    └── poses.md
```

## Requirements

- Python 3.10+
- Webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

For cloud-only deployment:

```bash
pip install -r cloud/requirements.txt
```

## 1. Collect Pose Data

Collect labeled keypoint rows into `data/{label}.csv`.

Note: old 99-dimensional pose CSVs are still accepted, but recollecting with the holistic collector gives better facial/hand expression accuracy.

Supported labels:

- `girl_look_fire`
- `monkey-pointing`
- `monkey-thinking-monkey`
- `side-eye`
- `think-better`
- `hmm-eye`
- `neutral`

Example:

```bash
python src/collect_data.py --label girl_look_fire
```

List available cameras first:

```bash
python src/collect_data.py --label girl_look_fire --list-cameras
```

Pick a specific camera ID (for example, built-in webcam):

```bash
python src/collect_data.py --label girl_look_fire --camera-id 1
```

Quick collection shortcuts (run each command separately):

```bash
python src/collect_data.py --label girl_look_fire
python src/collect_data.py --label monkey-pointing
python src/collect_data.py --label monkey-thinking-monkey
python src/collect_data.py --label side-eye
python src/collect_data.py --label think-better
python src/collect_data.py --label hmm-eye
python src/collect_data.py --label neutral
```

Press `w` in the webcam window to start collecting samples and `q` to stop.

## 2. Train Local Model

Train local RandomForest classifier and save model to `models/local_model.pkl`.

```bash
python src/train.py
```

The script prints a classification report and a confusion matrix.

Example hyperparameter tuning run:

```bash
python src/train.py --n-estimators 300 --max-depth 20 --min-samples-leaf 2 --class-weight balanced_subsample
```

Quick grid-search mode:

```bash
python src/train.py --quick-grid-search --cv-folds 3 --scoring f1_macro
```

Optional report exports:

```bash
python src/train.py --report-path reports/local_report.txt --confusion-matrix-csv reports/local_confusion.csv
```

## 3. Train Cloud Model

Train cloud classifier and save to `models/cloud_model.pkl`.

```bash
python cloud/cloud_train.py
```

Cloud training supports the same tuning/report flags as local training.

## 4. Run Cloud Server

Start Flask cloud API (`/infer` and `/correct`) on port `5000`:

```bash
python cloud/app.py
```

Or with Flask CLI:

```bash
cd cloud
flask --app app run --host 0.0.0.0 --port 5000
```

If port `5000` is unavailable on your machine, use another port:

```bash
FLASK_APP=cloud/app.py flask run --host 0.0.0.0 --port 5050
```

## 5. Run Local Real-Time Inference

Start local webcam inference:

```bash
python src/local_inference.py
```

List available cameras:

```bash
python src/local_inference.py --list-cameras
```

Use a specific camera ID:

```bash
python src/local_inference.py --camera-id 1
```

Optional cloud URL override:

```bash
CLOUD_URL=http://localhost:5000 python src/local_inference.py
```

If cloud runs on port `5050`:

```bash
CLOUD_URL=http://localhost:5050 python src/local_inference.py
```

If local confidence is below `0.75`, the client sends keypoints to cloud `/infer`.

During local inference, press `t` to toggle layout between webcam-main and meme-main (large meme with small webcam inset).

During local inference, press `c` to cycle between detected camera IDs.

If the predicted label is `neutral`, the app intentionally shows no meme image.

## Cloud API

### `POST /infer`

Request:

```json
{
  "keypoints": [0.1, 0.2, 0.3, "... 333 floats total (default holistic) ..."],
  "device_id": "laptop-001"
}
```

Response:

```json
{
  "meme": "think-better",
  "confidence": 0.95
}
```

### `POST /correct`

Request:

```json
{
  "keypoints": [0.1, 0.2, 0.3, "... 333 floats total (default holistic) ..."],
  "correct_label": "think-better",
  "device_id": "laptop-001"
}
```

Corrections are buffered and trigger cloud retraining every 20 samples.

## Meme-to-Pose Mapping

| Pose Label       | Meme Image       |
|------------------|------------------|
| `girl_look_fire` | `girl_look_fire.png` |
| `monkey-pointing` | `monkey-pointing.png` |
| `monkey-thinking-monkey` | `monkey-thinking-monkey.png` |
| `side-eye` | `side-eye.png` |
| `think-better` | `think-better.png` |
| `hmm-eye` | `hmm-eye.png` |
| `neutral` | *(no meme shown)* |

## Notes

- Meme images in `memes/` are starter placeholders. Replace with your preferred PNG assets.
- Train models after collecting data before running inference.
