# Memefieder Architecture

Memefieder uses a hybrid local-plus-cloud inference architecture:

1. Webcam frame is captured on the client.
2. MediaPipe Holistic extracts pose, hand, and face landmarks.
3. Landmarks are flattened into a fixed feature vector (default 333 dimensions).
4. Local model (`models/local_model.pkl`) predicts a pose label and confidence.
5. If confidence is >= 0.75, local prediction is used directly.
6. If confidence is < 0.75, keypoints are sent to cloud `POST /infer` for a second opinion.
7. Final label selects the matching meme image and overlays it on the live frame.

Backward compatibility:

1. Legacy 99-dimensional pose CSV files are padded automatically during training.
2. If a model expects fewer features than the live extractor emits, features are truncated safely at inference.

Correction loop:

1. Client can send corrected labels to cloud `POST /correct`.
2. Cloud stores corrections in a pending buffer.
3. After 20 corrections, cloud appends them into `data/*.csv` and retrains `models/cloud_model.pkl`.
