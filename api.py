import io
import re
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from model.model import Model
from model.utils import CTCLabelConverter


# --------------------
# FastAPI
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# MODEL CONFIG
# --------------------
class Opt:
    Transformation = "TPS"
    FeatureExtraction = "ResNet"
    SequenceModeling = "BiLSTM"
    Prediction = "CTC"

    imgH = 32
    imgW = 256
    PAD = True
    input_channel = 1
    output_channel = 512
    hidden_size = 256
    num_fiducial = 20
    batch_max_length = 120

    # boşluk DAHİL – cümle için şart
    character = "0123456789abcdefghijklmnopqrstuvwxyz .,!?'-\""


opt = Opt()

converter = CTCLabelConverter(opt.character)
opt.num_class = len(converter.character)

# --------------------
# MODEL INIT
# --------------------
model = Model(opt)

# Eğitilmiş model varsa açılır, yoksa kapalı kalır
# model.load_state_dict(torch.load("path/to/your_model.pth", map_location=device))

model = model.to(device)
model.eval()

# --------------------
# CTC → CÜMLE POSTPROCESS
# --------------------
def ctc_sentence_postprocess(text: str) -> str:
    """
    Ham CTC çıktısını okunabilir cümleye çevirir
    """
    # çoklu boşluk → tek boşluk
    text = re.sub(r"\s+", " ", text)

    # noktalama öncesi boşlukları sil
    text = re.sub(r"\s+([.,!?])", r"\1", text)

    # baş/son boşluk temizle
    text = text.strip()

    # cümle başı büyük harf
    if text:
        text = text[0].upper() + text[1:]

    return text


# --------------------
# IMAGE PREPROCESS
# --------------------
def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("L")
    image = image.resize((opt.imgW, opt.imgH))

    img = torch.tensor(
        torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        .float()
        .view(opt.imgH, opt.imgW) / 255.0
    )

    img = img.unsqueeze(0).unsqueeze(0)
    img = (img - 0.5) / 0.5

    return img.to(device)


# --------------------
# ENDPOINTS
# --------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict")
async def predict_sentence(image: UploadFile = File(...)):
    data = await image.read()
    pil_image = Image.open(io.BytesIO(data))

    img_tensor = preprocess_image(pil_image)

    with torch.no_grad():
        dummy_text = torch.zeros((1, opt.batch_max_length), dtype=torch.long).to(device)
        preds = model(img_tensor, dummy_text)
        preds = preds.log_softmax(2)

    preds_size = torch.IntTensor([preds.size(1)])
    _, preds_index = preds.max(2)
    preds_index = preds_index.transpose(1, 0)

    raw_text = converter.decode(preds_index, preds_size)[0]
    sentence = ctc_sentence_postprocess(raw_text)

    return {
        "raw": raw_text,
        "sentence": sentence
    }
