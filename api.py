import os
import uuid
import tempfile
import shutil
import subprocess
import hashlib
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# AttentionHTR model/ klasörü
ATTN_MODEL_DIR = Path("/home/beys/Text_Reco/HTR/model")

# Model dosyası (absolute)
SAVED_MODEL = ATTN_MODEL_DIR / "saved_model" / "AttentionHTR-General-sensitive.pth"


@app.get("/health")
def health():
    return {"ok": True}


def _latest_new_log(before_set: set[str]) -> Path:
    """
    result/** altında log_predictions_*.txt dosyalarını arar.
    before_set: test.py çalışmadan önce var olan log path listesi.
    Yeni oluşan log varsa onu döndürür; yoksa en yeni log’u döndürür.
    """
    all_logs = sorted(
        ATTN_MODEL_DIR.glob("result/**/log_predictions_*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not all_logs:
        raise RuntimeError(f"No prediction log file found under {ATTN_MODEL_DIR / 'result'}")

    new_logs = [p for p in all_logs if p.as_posix() not in before_set]
    return new_logs[0] if new_logs else all_logs[0]


def attentionhtr_predict_single(image_path: str) -> str:
    if not SAVED_MODEL.exists():
        raise FileNotFoundError(f"Saved model not found: {SAVED_MODEL}")
    if not (ATTN_MODEL_DIR / "test.py").exists():
        raise FileNotFoundError(f"test.py not found under: {ATTN_MODEL_DIR}")

    # test.py çalışmadan önce mevcut log’ların listesini al (recursive)
    before = {p.as_posix() for p in ATTN_MODEL_DIR.glob("result/**/log_predictions_*.txt")}

    req_id = uuid.uuid4().hex[:10]
    dataset_name = f"api_{req_id}"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        input_dir = tmp / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        src = Path(image_path)
        dst = input_dir / src.name
        shutil.copyfile(src, dst)

        # (debug) modelin gerçekten gördüğü dosya aynı mı?
        dst_bytes = dst.read_bytes()
        print(f"[TMP_INPUT] name={dst.name} bytes={len(dst_bytes)} md5={hashlib.md5(dst_bytes).hexdigest()}")

        gt_path = tmp / "gt.txt"
        gt_path.write_text(f"{dst.name}\t-\n", encoding="utf-8")

        out_lmdb = tmp / dataset_name

        # 1) LMDB üret
        subprocess.check_call(
            [
                "python3",
                str(ATTN_MODEL_DIR / "create_lmdb_dataset.py"),
                "--inputPath",
                str(input_dir),
                "--gtFile",
                str(gt_path),
                "--outputPath",
                str(out_lmdb),
            ],
            cwd=str(ATTN_MODEL_DIR),
        )

        # 2) test.py çalıştır
        # Not: absolute saved_model veriyoruz; log klasörü sanitize edilip result/ altında oluşuyor (sende gördüğümüz gibi)
        subprocess.check_call(
            [
                "python3",
                str(ATTN_MODEL_DIR / "test.py"),
                "--eval_data",
                str(out_lmdb),
                "--Transformation",
                "TPS",
                "--FeatureExtraction",
                "ResNet",
                "--SequenceModeling",
                "BiLSTM",
                "--Prediction",
                "Attn",
                "--saved_model",
                str(SAVED_MODEL),
                "--sensitive",
            ],
            cwd=str(ATTN_MODEL_DIR),
        )

    # 3) Bu request’in oluşturduğu log’u bul (result/** altında)
    log_file = _latest_new_log(before)
    print(f"[LOG] using: {log_file}")

    # 4) prediction çek (son data satırı)
    lines = log_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    pred = ""
    for line in reversed(lines):
        if line.startswith("batch,") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) >= 3:
            pred = parts[2].strip()
            break

    return pred


@app.post("/predict-word")
async def predict_word_api(
    image: UploadFile = File(...),
    infer_orientation: str = "auto",  # frontend bozmasın diye duruyor; şimdilik kullanılmıyor
):
    ext = os.path.splitext(image.filename)[1].lower() or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)

    data = await image.read()
    sha = hashlib.md5(data).hexdigest()
    print(f"[UPLOAD] filename={image.filename} saved_as={path} bytes={len(data)} md5={sha}")

    with open(path, "wb") as f:
        f.write(data)

    try:
        text = attentionhtr_predict_single(path)
        return {"prediction": text}
    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": f"AttentionHTR failed: {e}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
