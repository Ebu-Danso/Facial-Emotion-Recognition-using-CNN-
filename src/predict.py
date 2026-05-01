"""
predict.py — Single image or webcam inference for FER
Usage:
  python src/predict.py --image path/to/face.jpg
  python src/predict.py --webcam
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model

from src.utils import load_config


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲",
}

CLASS_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 0  ),
    "fear":     (128, 0,   128),
    "happy":    (0,   200, 200),
    "neutral":  (180, 180, 180),
    "sad":      (200, 100, 0  ),
    "surprise": (0,   180, 255),
}


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray, img_size: int = 48) -> np.ndarray:
    """Convert BGR image to model-ready input (1, img_size, img_size, 1)."""
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (img_size, img_size))
    norm    = resized.astype("float32") / 255.0
    return norm.reshape(1, img_size, img_size, 1)


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────

def predict(model, img_bgr: np.ndarray, class_names: list, img_size: int = 48):
    """
    Run inference on a single BGR image.
    Returns: (label, confidence, probs)
    """
    tensor = preprocess(img_bgr, img_size)
    probs  = model.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs


# ─────────────────────────────────────────────────────────────
# FACE DETECTOR
# ─────────────────────────────────────────────────────────────

def get_face_detector():
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(path):
        print("⚠  Haar cascade not found — using full frame.")
        return None
    return cv2.CascadeClassifier(path)


def detect_faces(gray_frame: np.ndarray, detector) -> list:
    if detector is None:
        h, w = gray_frame.shape
        return [(0, 0, w, h)]
    faces = detector.detectMultiScale(gray_frame, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30))
    return list(faces) if len(faces) else []


# ─────────────────────────────────────────────────────────────
# DRAW OVERLAY
# ─────────────────────────────────────────────────────────────

def draw_overlay(frame, x, y, w, h, label, confidence, probs, class_names):
    color = CLASS_COLORS.get(label, (255, 255, 255))

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Label badge
    text = f"{EMOJI.get(label, '')} {label}  {confidence*100:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x, y - th - 12), (x + tw + 8, y), color, -1)
    cv2.putText(frame, text, (x + 4, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Probability bars (top-right)
    bx = frame.shape[1] - 160
    for i, (cls, p) in enumerate(zip(class_names, probs)):
        by  = 10 + i * 22
        bw  = int(p * 140)
        c   = CLASS_COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(frame, (bx, by), (bx + bw, by + 14), c, -1)
        cv2.putText(frame, f"{cls[:4]} {p*100:4.1f}%",
                    (bx - 110, by + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    return frame


# ─────────────────────────────────────────────────────────────
# MODE 1 — SINGLE IMAGE
# ─────────────────────────────────────────────────────────────

def predict_image(args, model, class_names, img_size):
    if not os.path.isfile(args.image):
        sys.exit(f"❌  File not found: {args.image}")

    frame = cv2.imread(args.image)
    label, confidence, probs = predict(model, frame, class_names, img_size)

    print(f"\n{'─'*40}")
    print(f"  Image      : {args.image}")
    print(f"  Prediction : {EMOJI.get(label, '')}  {label.upper()}")
    print(f"  Confidence : {confidence*100:.2f}%")
    print(f"{'─'*40}")
    print("  All probabilities:")
    for cls, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
        bar = "█" * int(p * 30)
        print(f"    {cls:<10} {p*100:5.2f}%  {bar}")

    if args.show or args.save_output:
        detector = get_face_detector()
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces    = detect_faces(gray, detector)
        x, y, w, h = faces[0] if faces else (0, 0, frame.shape[1], frame.shape[0])
        frame = draw_overlay(frame, x, y, w, h, label, confidence, probs, class_names)

        if args.save_output:
            cv2.imwrite(args.save_output, frame)
            print(f"\n✓ Saved to: {args.save_output}")
        if args.show:
            cv2.imshow("FER Prediction", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
# MODE 2 — WEBCAM
# ─────────────────────────────────────────────────────────────

def predict_webcam(args, model, class_names, img_size):
    cap      = cv2.VideoCapture(args.camera_id)
    detector = get_face_detector()

    if not cap.isOpened():
        sys.exit(f"❌  Cannot open camera {args.camera_id}")

    print(f"✓ Webcam started — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)

        for (x, y, w, h) in faces:
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            label, confidence, probs = predict(model, crop, class_names, img_size)
            frame = draw_overlay(frame, x, y, w, h, label, confidence, probs, class_names)

        cv2.imshow("FER Real-Time (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✓ Webcam closed.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="FER Inference")
    parser.add_argument("--config", type=str, default="configs/base.yaml")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--image",  type=str, help="Path to image file")
    mode.add_argument("--webcam", action="store_true")

    parser.add_argument("--camera-id",   type=int, default=0)
    parser.add_argument("--show",        action="store_true")
    parser.add_argument("--save-output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args        = parse_args()
    config      = load_config(args.config)
    class_names = config["classes"]
    img_size    = config["image"]["img_size"]
    model_path  = config["output"]["model_path"]

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("✓ Model loaded.")

    if args.webcam:
        predict_webcam(args, model, class_names, img_size)
    else:
        predict_image(args, model, class_names, img_size)
