# Memefieder 5-Minute Rubric Demo Script

## Goal
Deliver a tight 5-minute demo that covers:
- Section 1: local real-time input + local inference + result display
- Section 2: EC2 VM inference + IoT to cloud communication
- Section 3: own model training evidence + concurrent users + online model update path

## Pre-Demo Setup (do this before recording)
1. EC2 cloud API is running on port 5000 with Gunicorn.
2. Security Group allows inbound TCP 5000 from your presenter machine.
3. Mac local environment is activated and camera ID is known.
4. Have two terminals visible:
- Terminal A (EC2 SSH): cloud logs
- Terminal B (Mac): local inference + curl
5. Keep these tabs ready:
- README with architecture section
- reports/local_report.txt and reports/cloud_report.txt

## Fast Command Pack (copy and run)

Terminal A (EC2):

source .venv/bin/activate
sudo lsof -t -iTCP:5000 -sTCP:LISTEN | xargs -r sudo kill -9
gunicorn --chdir cloud app:app --bind 0.0.0.0:5000 --workers 2 --threads 4 --timeout 60 --access-logfile - --error-logfile - --capture-output

Terminal B (Mac):

source /Users/abdi/Desktop/IotCourseProject/.venv/bin/activate
curl http://75.101.189.100:5000/health
curl http://75.101.189.100:5000/metrics

CLOUD_URL=http://75.101.189.100:5000 \
CLOUD_TIMEOUT_SEC=0.25 CLOUD_RETRIES=1 \
CLOUD_MIN_INTERVAL_SEC=0.40 CLOUD_RESULT_TTL_SEC=1.5 \
LOCAL_CONFIDENCE_THRESHOLD=0.75 CLOUD_LOG_EVENTS=1 \
python src/local_inference.py --camera-id 1 --max-cameras 2 --frame-width 640 --frame-height 480 --model-complexity 0 --disable-face-refine --show-fps

After 20 to 30 seconds of posing:

curl http://75.101.189.100:5000/metrics

Concurrency proof (Terminal B):

python scripts/load_test.py --url http://75.101.189.100:5000 --requests 150 --workers 25 --timeout 2.0 --report-path reports/load_test_summary.json

Online update endpoint proof (Terminal B):

curl -X POST http://75.101.189.100:5000/reload-model

## 5-Minute Talk Track (timestamped)

### 0:00 to 0:30 - Architecture and objective
Say:
This project is edge first. The MacBook webcam app performs local inference in real time. If confidence drops below threshold, it sends keypoints to the EC2 cloud model and displays the returned result.

Show:
- README architecture block
- EC2 health endpoint output

Rubric hit:
- Section 2 VM setup context
- Section 2 communication design clarity

### 0:30 to 1:40 - Local real-time input and local inference
Say:
I am now collecting live camera input and running inference locally on-device. The UI overlays the predicted meme and confidence.

Do:
- Run local_inference.py command above
- Stand in frame and trigger 2 to 3 known labels
- Press t once to show alternate display layout
- Press f once to hide/show face markers

Show on screen:
- Pose label and confidence
- Source line showing LOCAL for high-confidence frames

Rubric hit:
- Section 1 Collect User Input: real-time camera
- Section 1 Infer Locally and Display Result
- Section 1 Physical device demonstration (MacBook edge runtime)

### 1:40 to 2:40 - IoT to cloud fallback live proof
Say:
Now I reduce confidence by changing pose quality and angle, so fallback requests fire to EC2.

Do:
- Continue posing with ambiguous movements
- Point to Terminal B logs showing [CLOUD FALLBACK]
- Point to EC2 Terminal A logs showing [CLOUD INFER]
- Run curl metrics and show infer_request_count increased

Rubric hit:
- Section 2 Communicate Between IoT and Cloud (highest weight)
- Section 2 Cloud VM receives and processes requests

### 2:40 to 3:25 - Cloud VM inference evidence
Say:
This EC2 endpoint is not localhost. The public IP endpoint receives requests and returns inference JSON.

Do:
- Show curl health and curl metrics against EC2 public IP
- Highlight low latency and device counter fields

Rubric hit:
- Section 2 Run Inference in Cloud VM

### 3:25 to 4:10 - Advanced task: concurrent users
Say:
The cloud backend supports concurrent users through Gunicorn workers and thread pool handling. I now run a burst load test.

Do:
- Run scripts/load_test.py command
- Show success rate, throughput, p95 latency
- Mention report saved to reports/load_test_summary.json

Rubric hit:
- Section 3 Support Multiple Concurrent Users

### 4:10 to 4:45 - Advanced task: training and online update
Say:
I trained my own model and documented evaluation reports in this repository. The cloud also supports model reload and correction-driven online updates.

Do:
- Show reports/local_report.txt and reports/cloud_report.txt
- Run curl POST /reload-model and show model metadata returned
- Mention /correct endpoint triggers retraining every 20 corrections

Rubric hit:
- Section 3 Train Your Own Model
- Section 3 Support Online Model Updating

### 4:45 to 5:00 - Close and rubric checklist recap
Say:
This demo showed real-time local inference on a physical edge device, live IoT to EC2 fallback communication, concurrent request handling, and model lifecycle update mechanisms.

Show:
- One-page checklist slide or README section mapping each rubric item to evidence

## Rubric Coverage Matrix

- Section 1.1 Real-time input: local webcam capture in local_inference.py
- Section 1.2 Local inference and display: on-screen pose label, confidence, meme overlay
- Section 1.3 Device runtime: local app running on MacBook hardware
- Section 2.1 Cloud VM inference: EC2 Gunicorn plus /health and /infer responses
- Section 2.2 IoT-cloud communication: fallback logs plus /metrics counters
- Section 3.1 Train own model: train.py, cloud_train.py, reports outputs
- Section 3.2 Concurrent users: scripts/load_test.py summary and latency stats
- Section 3.3 Online updates: /correct buffered retrain, /reload-model endpoint

## Backup Plan If Something Fails

If live webcam is unstable:
- Show pre-recorded 20 second clip of local inference
- Continue live with curl /metrics and EC2 logs

If EC2 network drops:
- Switch CLOUD_URL to localhost and continue local demo
- Show prior screenshot of EC2 metrics and health from same day

If camera index changes:
- Run python src/local_inference.py --list-cameras
- Relaunch with working --camera-id

## Submission Artifacts to Attach
- Screenshot 1: local inference window with Source and FPS
- Screenshot 2: Mac terminal [CLOUD FALLBACK] lines
- Screenshot 3: EC2 terminal [CLOUD INFER] lines
- Screenshot 4: curl public EC2 /metrics JSON
- Screenshot 5: load_test.py summary
- Screenshot 6: local_report and cloud_report snippet
