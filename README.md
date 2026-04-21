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

## Architecture Overview

Memefieder uses an edge-cloud split architecture:

1. Edge device (MacBook): captures webcam frames, extracts 333-dim holistic keypoints, runs local model inference, and renders meme overlay.
2. Cloud server (EC2): hosts Flask/Gunicorn API for fallback inference and correction-driven retraining.
3. Decision path: if local confidence is >= 0.75, use local prediction; otherwise call cloud `/infer` and use cloud result.

High-level flow:

```text
MacBook webcam -> MediaPipe Holistic -> Local model prediction
            | confidence < 0.75
            v
          EC2 cloud /infer
            |
            v
        final label + meme overlay
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

### Run Cloud Server with Gunicorn (recommended for demo/prod)

Gunicorn is a production WSGI server that runs multiple worker processes, which is better for concurrent requests than Flask's built-in development server.

Install cloud dependencies (includes Gunicorn):

```bash
pip install -r cloud/requirements.txt
```

Run Gunicorn from repository root:

```bash
gunicorn --chdir cloud app:app --bind 0.0.0.0:5000 --workers 2 --threads 4 --timeout 60
```

Quick health check:

```bash
curl http://localhost:5000/health
```

On EC2, open inbound TCP 5000 in the Security Group, then test from your laptop:

```bash
curl http://<EC2_PUBLIC_IP>:5000/health
```

### EC2 Setup Commands (Ubuntu)

Use these commands on EC2 to run cloud inference reliably:

```bash
cd ~/memefieder
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r cloud/requirements.txt
python3 cloud/cloud_train.py
gunicorn --chdir cloud app:app --bind 0.0.0.0:5000 --workers 2 --threads 4 --timeout 60
```

Optional systemd service (auto-start on reboot):

```bash
sudo tee /etc/systemd/system/memefieder-cloud.service >/dev/null <<'EOF'
[Unit]
Description=Memefieder Cloud API (Gunicorn)
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/memefieder
Environment=PATH=/home/ubuntu/memefieder/.venv/bin
ExecStart=/home/ubuntu/memefieder/.venv/bin/gunicorn --chdir cloud app:app --bind 0.0.0.0:5000 --workers 2 --threads 4 --timeout 60
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now memefieder-cloud
sudo systemctl status memefieder-cloud --no-pager
curl http://localhost:5000/health
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

For EC2 cloud fallback from your MacBook:

```bash
CLOUD_URL=http://<EC2_PUBLIC_IP>:5000 python src/local_inference.py
```

Fast profile (recommended when webcam feels slow):

```bash
CLOUD_URL=http://<EC2_PUBLIC_IP>:5000 \
CLOUD_TIMEOUT_SEC=0.25 CLOUD_RETRIES=1 \
CLOUD_MIN_INTERVAL_SEC=0.40 CLOUD_RESULT_TTL_SEC=1.5 \
LOCAL_CONFIDENCE_THRESHOLD=0.70 \
python src/local_inference.py --camera-id 1 --max-cameras 2 --frame-width 640 --frame-height 480 --model-complexity 0 --disable-face-refine --show-fps
```

This profile reduces cloud blocking and Holistic compute load. Use a higher threshold for better cloud usage (`0.75`) or a lower threshold for better FPS (`0.65` to `0.70`).

If cloud runs on port `5050`:

```bash
CLOUD_URL=http://localhost:5050 python src/local_inference.py
```

If local confidence is below `0.75`, the client sends keypoints to cloud `/infer`.

During local inference, press `t` to toggle layout between webcam-main and meme-main (large meme with small webcam inset).

During local inference, press `c` to cycle between detected camera IDs.

During local inference, press `f` to toggle face landmark markers on/off in the live view.

If the predicted label is `neutral`, the app intentionally shows no meme image.

Important: do not run `src/local_inference.py` on a headless EC2 instance. It requires a local webcam/display. Run local inference on your MacBook and keep EC2 for cloud API.

## Correct Demo Workflow (MacBook + EC2)

1. Start cloud API on EC2 (Gunicorn or systemd service).
2. Verify EC2 health endpoint from EC2 and from your MacBook:

```bash
curl http://localhost:5000/health
curl http://<EC2_PUBLIC_IP>:5000/health
```

3. On MacBook, run local inference against EC2:

```bash
CLOUD_URL=http://<EC2_PUBLIC_IP>:5000 python src/local_inference.py
```

4. Stand in front of webcam:
- high-confidence frames use local prediction directly
- low-confidence frames (< 0.75) call EC2 `/infer` for second opinion

5. Meme overlay updates from the final label, demonstrating both:
- local real-time inference
- live cloud communication fallback

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

### `GET /metrics`

Returns cloud fallback telemetry counters and latency summary:

```json
{
  "infer_request_count": 42,
  "infer_device_counts": {"abdillahs-macbook-pro": 42},
  "infer_label_counts": {"side-eye": 10, "neutral": 8},
  "infer_avg_latency_ms": 18.7,
  "last_infer": {
    "device_id": "abdillahs-macbook-pro",
    "label": "side-eye",
    "confidence": 0.91,
    "latency_ms": 16.2,
    "timestamp": 1776781234.56
  }
}
```

Query from EC2 or your MacBook:

```bash
curl http://localhost:5000/metrics
curl http://<EC2_PUBLIC_IP>:5000/metrics
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

## Model Performance

Loaded samples: 2256
Feature dimension: 333
Class distribution:
- girl_look_fire: 322
- hmm-eye: 344
- monkey-pointing: 347
- monkey-thinking-monkey: 323
- neutral: 293
- side-eye: 306
- think-better: 321

Evaluation scope: holdout test split

Classification report:
                        precision    recall  f1-score   support

        girl_look_fire       1.00      1.00      1.00        64
       monkey-pointing       1.00      1.00      1.00        70
monkey-thinking-monkey       1.00      1.00      1.00        65
              side-eye       1.00      1.00      1.00        61
          think-better       1.00      1.00      1.00        64
               hmm-eye       1.00      1.00      1.00        69
               neutral       1.00      1.00      1.00        59

              accuracy                           1.00       452
             macro avg       1.00      1.00      1.00       452
          weighted avg       1.00      1.00      1.00       452


Confusion matrix:
true\pred               girl_look_fire          monkey-pointing         monkey-thinking-monkey  side-eye                think-better            hmm-eye                 neutral                 
girl_look_fire          64                      0                       0                       0                       0                       0                       0                       
monkey-pointing         0                       70                      0                       0                       0                       0                       0                       
monkey-thinking-monkey  0                       0                       65                      0                       0                       0                       0                       
side-eye                0                       0                       0                       61                      0                       0                       0                       
think-better            0                       0                       0                       0                       64                      0                       0                       
hmm-eye                 0                       0                       0                       0                       0                       69                      0                       
neutral                 0                       0                       0                       0                       0                       0                       59                      
