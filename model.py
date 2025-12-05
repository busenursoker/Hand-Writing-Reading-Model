#!/usr/bin/env python3
"""
model.py — Train from Kaggle EMNIST CSVs and read SINGLE WORD images (non-cursive).

Folder structure:
  project/
    model.py
    data/
      emnist-letters-train.csv
      emnist-letters-test.csv
      emnist-letters-mapping.txt

Install:
  pip install tensorflow pandas numpy opencv-python pillow scikit-learn

Train:
  python model.py train --set letters --epochs 10

Predict one word (recommended):
  python model.py predict-word --image test.jpg --infer-orientation auto --dump-chars

Key idea:
- EMNIST CSV needs TRAIN orientation fix (usually transpose).
- Real photos are already upright, so INFER orientation should be "none" (default).
- Auto-orientation selection uses confidence + aspect ratio heuristics.
"""

import os, json, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

IMG_H = 28
IMG_W = 28
PIXELS = IMG_H * IMG_W

DEFAULT_DATA_DIR = "data"
DEFAULT_ARTIFACT_DIR = "artifacts"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_ARTIFACT_DIR, "model.keras")
DEFAULT_LABELS_PATH = os.path.join(DEFAULT_ARTIFACT_DIR, "labels.json")


def debug(msg: str): print(f"[DEBUG] {msg}")
def ensure(cond: bool, msg: str):
    if not cond:
        raise RuntimeError(f"[CHECK FAILED] {msg}")


# -----------------------------
# Orientation utilities
# -----------------------------
def apply_orientation_fix(images_nhw: np.ndarray, mode: str) -> np.ndarray:
    """
    images_nhw: (N,28,28) or (28,28) if N omitted (we handle both)
    mode: none | transpose | transpose_fliplr
    """
    if mode == "none":
        return images_nhw
    if mode == "transpose":
        return images_nhw.transpose(0, 2, 1) if images_nhw.ndim == 3 else images_nhw.T
    if mode == "transpose_fliplr":
        if images_nhw.ndim == 3:
            return np.fliplr(images_nhw.transpose(0, 2, 1))
        return np.fliplr(images_nhw.T)
    raise ValueError("mode must be one of: none, transpose, transpose_fliplr")


# -----------------------------
# Mapping.txt loader
# Kaggle mapping format: "<label> <ascii_code>"
# We'll map raw labels -> 0..C-1
# -----------------------------
def load_mapping(mapping_path: str) -> Tuple[List[str], Dict[int, int]]:
    ensure(os.path.exists(mapping_path), f"Mapping file not found: {mapping_path}")

    label_to_char: Dict[int, str] = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()[:2]
            raw_lbl = int(a)
            asc = int(b)
            label_to_char[raw_lbl] = chr(asc)

    ensure(len(label_to_char) > 0, "Mapping file empty or invalid.")

    raw_sorted = sorted(label_to_char.keys())
    index_to_char = [label_to_char[lbl] for lbl in raw_sorted]
    rawlabel_to_index = {lbl: i for i, lbl in enumerate(raw_sorted)}

    return index_to_char, rawlabel_to_index


def load_emnist_csv(csv_path: str, rawlabel_to_index: Dict[int, int], limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    ensure(os.path.exists(csv_path), f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path, nrows=limit)

    if "label" in df.columns:
        y_raw = df["label"].to_numpy()
        X = df.drop(columns=["label"]).to_numpy()
    else:
        y_raw = df.iloc[:, 0].to_numpy()
        X = df.iloc[:, 1:].to_numpy()

    ensure(X.shape[1] == PIXELS, f"Expected 784 pixels per row. Got {X.shape[1]}")
    y = np.array([rawlabel_to_index[int(v)] for v in y_raw], dtype=np.int64)
    return X.astype(np.float32), y


def build_model(num_classes: int) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_H, IMG_W, 1)),
        tf.keras.layers.Conv2D(32, (5,5), padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(strides=2),
        tf.keras.layers.Conv2D(48, (5,5), padding="valid", activation="relu"),
        tf.keras.layers.MaxPool2D(strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(84, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])


@dataclass
class TrainConfig:
    set_name: str = "letters"
    data_dir: str = DEFAULT_DATA_DIR
    model_path: str = DEFAULT_MODEL_PATH
    labels_path: str = DEFAULT_LABELS_PATH
    train_orientation: str = "transpose"   # ✅ for EMNIST CSV
    epochs: int = 10
    batch: int = 128
    lr: float = 1e-3
    val_split: float = 0.1
    seed: int = 88
    limit: Optional[int] = None
    limit_test: Optional[int] = None


def train(cfg: TrainConfig):
    os.makedirs(DEFAULT_ARTIFACT_DIR, exist_ok=True)

    train_csv = os.path.join(cfg.data_dir, f"emnist-{cfg.set_name}-train.csv")
    test_csv  = os.path.join(cfg.data_dir, f"emnist-{cfg.set_name}-test.csv")
    mapping   = os.path.join(cfg.data_dir, f"emnist-{cfg.set_name}-mapping.txt")

    index_to_char, rawlabel_to_index = load_mapping(mapping)
    num_classes = len(index_to_char)
    debug(f"Set={cfg.set_name} classes={num_classes} train_orientation={cfg.train_orientation}")

    with open(cfg.labels_path, "w", encoding="utf-8") as f:
        json.dump(index_to_char, f, ensure_ascii=False, indent=2)
    debug(f"Saved labels -> {cfg.labels_path}")

    X_train_flat, y_train = load_emnist_csv(train_csv, rawlabel_to_index, limit=cfg.limit)
    X_test_flat,  y_test  = load_emnist_csv(test_csv,  rawlabel_to_index, limit=cfg.limit_test)

    ensure(X_train_flat.shape[1] == 784, "Train pixels not 784.")
    ensure(y_train.min() >= 0 and y_train.max() < num_classes, "Train labels out of range.")

    X_train = X_train_flat.reshape(-1, 28, 28)
    X_test  = X_test_flat.reshape(-1, 28, 28)

    # ✅ Apply TRAIN orientation fix (EMNIST CSV -> upright)
    X_train = apply_orientation_fix(X_train, cfg.train_orientation)
    X_test  = apply_orientation_fix(X_test,  cfg.train_orientation)

    X_train = (X_train / 255.0).astype(np.float32)[..., np.newaxis]
    X_test  = (X_test  / 255.0).astype(np.float32)[..., np.newaxis]

    ensure(X_train.shape[1:] == (28,28,1), f"Bad X_train shape: {X_train.shape}")

    x_tr, x_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=cfg.val_split,
        random_state=cfg.seed,
        shuffle=True,
        stratify=y_train
    )
    debug(f"Split: train={x_tr.shape} val={x_val.shape} test={X_test.shape}")

    model = build_model(num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg.lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(cfg.model_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    model.fit(x_tr, y_tr,
              validation_data=(x_val, y_val),
              epochs=cfg.epochs,
              batch_size=cfg.batch,
              callbacks=callbacks,
              verbose=1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
    print(f"Saved model -> {cfg.model_path}")


# -----------------------------
# Inference
# -----------------------------
def load_artifacts(model_path: str, labels_path: str):
    ensure(os.path.exists(model_path), f"Model not found: {model_path}")
    ensure(os.path.exists(labels_path), f"labels.json not found: {labels_path}")
    model = tf.keras.models.load_model(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        index_to_char = json.load(f)
    return model, index_to_char


def preprocess_char_crop_to_28(char_gray: np.ndarray, infer_orientation: str) -> Optional[np.ndarray]:
    """Return (1,28,28,1) float32 in [0,1] or None if no ink."""
    if char_gray.ndim != 2:
        char_gray = cv2.cvtColor(char_gray, cv2.COLOR_BGR2GRAY)

    # Light denoise
    g = cv2.GaussianBlur(char_gray, (3,3), 0)
    
    # Make ink bright (like EMNIST) if the background is bright
    if g.mean() > 127:
        g = 255 - g

    # Use Otsu to locate ink region (bbox)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None

    # Crop to ink
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    g_crop = g[y0:y1+1, x0:x1+1]

    # Enhance contrast before binarization to preserve strokes
    # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for better local contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    g_enhanced = clahe.apply(g_crop)
    
    # Use Otsu thresholding - it's better at preserving strokes than adaptive
    _, g_binary = cv2.threshold(g_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # If threshold creates mostly white (inverted), flip it
    if g_binary.mean() > 127:
        g_binary = 255 - g_binary
    
    # Light morphological closing to connect broken strokes (but not too aggressive)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    g_binary = cv2.morphologyEx(g_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Preserve aspect ratio better - pad to square but maintain proportions
    h, w = g_binary.shape
    # Add padding proportional to size to preserve shape
    pad_h = max(2, h // 8)
    pad_w = max(2, w // 8)
    g_binary = cv2.copyMakeBorder(g_binary, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    h, w = g_binary.shape
    
    # Pad to square (maintain aspect ratio)
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    sq = np.zeros((side, side), dtype=np.uint8)
    sq[top:top+h, left:left+w] = g_binary

    # Add border for better centering when resizing
    border = max(4, side // 7)
    sq = cv2.copyMakeBorder(sq, border, border, border, border, cv2.BORDER_CONSTANT, value=0)

    # Resize using bilinear to preserve stroke smoothness, then binarize intelligently
    sq = cv2.resize(sq, (28, 28), interpolation=cv2.INTER_LINEAR)
    
    # Use Otsu thresholding for final binarization (preserves more detail than fixed threshold)
    _, sq = cv2.threshold(sq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # EMNIST format: black background (0) with white ink (255)
    # Check border pixels to determine background color (more reliable than mean)
    # Border should be background, not ink
    border_pixels = np.concatenate([
        sq[0, :],      # top border
        sq[-1, :],     # bottom border
        sq[:, 0],      # left border
        sq[:, -1]      # right border
    ])
    border_mean = border_pixels.mean()
    
    # If border is mostly white (>127), we have white background - need to invert
    # If border is mostly black (<127), we have black background - correct!
    if border_mean > 127:
        # White background detected - invert to get black background with white ink
        sq = 255 - sq

    # Match inference orientation (for real photos usually "none" or "auto" selection)
    sq = apply_orientation_fix(sq, infer_orientation)

    x = (sq.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)
    return x


def segment_chars_from_word(gray: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int,int,int,int]]]:
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = 255 - bw  # ink white

    ys, xs = np.where(bw > 0)
    ensure(len(xs) > 0, "No ink found in image.")
    x0,x1 = xs.min(), xs.max()
    y0,y1 = ys.min(), ys.max()
    bw = bw[y0:y1+1, x0:x1+1]
    crop_gray = gray[y0:y1+1, x0:x1+1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    sep = cv2.erode(bw, kernel, iterations=1)

    contours, _ = cv2.findContours(sep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 40:   # noise
            continue
        boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: b[0])
    ensure(len(boxes) > 0, "Failed to segment characters (letters may be touching).")
    return crop_gray, boxes


def choose_best_orientation_for_word(model, crop_gray, boxes, candidates: List[str]) -> str:
    """
    Try each orientation; pick the one with highest mean(max_prob) across chars.
    Also considers aspect ratio heuristics - letters are typically taller than wide.
    """
    best_mode = candidates[0]
    best_score = -1.0
    scores_by_mode = {}

    for mode in candidates:
        scores = []
        aspect_ratios = []
        for (x,y,w,h) in boxes:
            char_gray = crop_gray[y:y+h, x:x+w]
            # Check aspect ratio of original character
            aspect = h / w if w > 0 else 1.0
            aspect_ratios.append(aspect)
            
            xin = preprocess_char_crop_to_28(char_gray, mode)
            if xin is None:
                continue
            probs = model.predict(xin, verbose=0)[0]
            scores.append(float(probs.max()))
        
        if not scores:
            continue
            
        score = float(np.mean(scores))
        # Bonus for "none" if characters are typically taller (normal letters)
        # Penalize transpose if characters are already upright (tall)
        mean_aspect = float(np.mean(aspect_ratios)) if aspect_ratios else 1.0
        if mode == "none" and mean_aspect > 1.2:  # Tall characters suggest upright
            score += 0.1
        elif mode == "transpose" and mean_aspect > 1.2:  # Don't transpose tall chars
            score -= 0.15
        
        scores_by_mode[mode] = score
        debug(f"Orientation {mode}: mean confidence={score:.4f}, mean_aspect={mean_aspect:.2f}")
        if score > best_score:
            best_score = score
            best_mode = mode

    debug(f"Chosen infer_orientation = {best_mode} (score={best_score:.4f})")
    return best_mode


def predict_word(image_path: str, model_path: str, labels_path: str,
                 infer_orientation: str = "none",  # Changed default: real photos are upright
                 dump_chars: bool = False,
                 debug_boxes: bool = True):
    model, index_to_char = load_artifacts(model_path, labels_path)

    img = cv2.imread(image_path)
    ensure(img is not None, f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    crop_gray, boxes = segment_chars_from_word(gray)

    if debug_boxes:
        dbg = cv2.cvtColor(crop_gray, cv2.COLOR_GRAY2BGR)
        for (x,y,w,h) in boxes:
            cv2.rectangle(dbg, (x,y), (x+w, y+h), (0,255,0), 2)
        os.makedirs(DEFAULT_ARTIFACT_DIR, exist_ok=True)
        out_path = os.path.join(DEFAULT_ARTIFACT_DIR, "debug_word_boxes.png")
        cv2.imwrite(out_path, dbg)
        debug(f"Saved boxes -> {out_path}")

    if infer_orientation == "auto":
        infer_orientation = choose_best_orientation_for_word(
            model, crop_gray, boxes,
            candidates=["none", "transpose", "transpose_fliplr"]
        )

    if dump_chars:
        dump_dir = os.path.join(DEFAULT_ARTIFACT_DIR, "processed_chars")
        os.makedirs(dump_dir, exist_ok=True)

    chars = []
    for i, (x,y,w,h) in enumerate(boxes):
        char_gray = crop_gray[y:y+h, x:x+w]
        xin = preprocess_char_crop_to_28(char_gray, infer_orientation)
        if xin is None:
            chars.append("?")
            continue

        if dump_chars:
            # save what model sees (28x28)
            img28 = (xin[0,:,:,0] * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(dump_dir, f"{i:02d}.png"), img28)

        probs = model.predict(xin, verbose=0)[0]
        pred = int(np.argmax(probs))
        chars.append(index_to_char[pred])

    print("".join(chars))


def predict_char(image_path: str, model_path: str, labels_path: str,
                 infer_orientation: str = "auto", topk: int = 3):
    model, index_to_char = load_artifacts(model_path, labels_path)

    img = cv2.imread(image_path)
    ensure(img is not None, f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if infer_orientation == "auto":
        # try all and pick best confidence for this single char
        best_mode, best_conf, best_probs = None, -1.0, None
        for mode in ["none", "transpose", "transpose_fliplr"]:
            xin = preprocess_char_crop_to_28(gray, mode)
            if xin is None:
                continue
            probs = model.predict(xin, verbose=0)[0]
            conf = float(probs.max())
            if conf > best_conf:
                best_mode, best_conf, best_probs = mode, conf, probs
        ensure(best_probs is not None, "No ink found.")
        debug(f"Chosen infer_orientation={best_mode} conf={best_conf:.4f}")
        probs = best_probs
    else:
        xin = preprocess_char_crop_to_28(gray, infer_orientation)
        ensure(xin is not None, "No ink found.")
        probs = model.predict(xin, verbose=0)[0]

    top_idx = probs.argsort()[-topk:][::-1]
    for i in top_idx:
        print(f"{index_to_char[int(i)]}  {float(probs[int(i)]):.4f}")


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--set", dest="set_name", default="letters",
                   help="letters | balanced | byclass | digits | mnist | bymerge")
    t.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--val-split", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=88)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--limit-test", type=int, default=None)
    t.add_argument("--train-orientation", choices=["none","transpose","transpose_fliplr"], default="transpose")
    t.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    t.add_argument("--labels-path", default=DEFAULT_LABELS_PATH)

    pc = sub.add_parser("predict-char")
    pc.add_argument("--image", required=True)
    pc.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    pc.add_argument("--labels-path", default=DEFAULT_LABELS_PATH)
    pc.add_argument("--infer-orientation", choices=["none","transpose","transpose_fliplr","auto"], default="auto")
    pc.add_argument("--topk", type=int, default=3)

    pw = sub.add_parser("predict-word")
    pw.add_argument("--image", required=True)
    pw.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    pw.add_argument("--labels-path", default=DEFAULT_LABELS_PATH)
    pw.add_argument("--infer-orientation", choices=["none","transpose","transpose_fliplr","auto"], default="none")
    pw.add_argument("--dump-chars", action="store_true", help="save model inputs to artifacts/processed_chars/")
    pw.add_argument("--no-debug-boxes", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(DEFAULT_ARTIFACT_DIR, exist_ok=True)

    if args.cmd == "train":
        cfg = TrainConfig(
            set_name=args.set_name,
            data_dir=args.data_dir,
            model_path=args.model_path,
            labels_path=args.labels_path,
            train_orientation=args.train_orientation,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            val_split=args.val_split,
            seed=args.seed,
            limit=args.limit,
            limit_test=args.limit_test,
        )
        train(cfg)

    elif args.cmd == "predict-char":
        predict_char(args.image, args.model_path, args.labels_path, args.infer_orientation, topk=args.topk)

    elif args.cmd == "predict-word":
        predict_word(args.image, args.model_path, args.labels_path,
                     infer_orientation=args.infer_orientation,
                     dump_chars=args.dump_chars,
                     debug_boxes=(not args.no_debug_boxes))


if __name__ == "__main__":
    main()
