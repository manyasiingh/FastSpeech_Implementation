import os, io, re, random, unicodedata
import numpy as np, soundfile as sf, librosa
from datasets import load_dataset, Audio

DATASET, SPLIT = "SPRINGLab/IndicTTS_Telugu", "train"
TEXT_COL, GENDER_COL, GENDER = "text", "gender" , 0    # <- set from the Cell 5 printout
SR, SEED, TARGET_H, MIN_S, MAX_S = 22050, 42, 6.0, 1.5, 20.0   # 6h total = 5h train + 1h val
random.seed(SEED)

ds = load_dataset(DATASET, split=SPLIT)
acol = [k for k, v in ds.features.items() if isinstance(v, Audio)][0]
ds = ds.cast_column(acol, Audio(decode=False))
texts = ds[TEXT_COL]
genders = list(ds[GENDER_COL]) if GENDER_COL else None

def clean(t):
    t = unicodedata.normalize("NFC", t)
    if re.search(r"[A-Za-z0-9\u0C66-\u0C6F]", t): return ""          # skip lines with Latin letters or digits
    t = re.sub(r"[\u200c\u200d]", "", t)
    t = re.sub(r"[^\u0C00-\u0C7F ]", " ", t)                          # keep Telugu block and spaces only
    return re.sub(r"\s+", " ", t).strip()

ok = [i for i in range(len(ds)) if len(clean(texts[i])) > 3 and (genders is None or genders[i] == GENDER)]
print(len(ds), "total ->", len(ok), "candidate utterances")
random.shuffle(ok)

os.makedirs("raw_data/Telugu/te_spk", exist_ok=True)
acc, n = 0.0, 0
for i in ok:
    if acc >= TARGET_H * 3600: break
    a = ds[i][acol]
    y, sr = sf.read(io.BytesIO(a["bytes"])) if a.get("bytes") else sf.read(a["path"])
    if y.ndim > 1: y = y.mean(1)
    d = len(y) / sr
    if not (MIN_S <= d <= MAX_S): continue
    y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR) if sr != SR else y.astype(np.float32)
    sf.write(f"raw_data/Telugu/te_spk/te_{i}.wav", y, SR)
    open(f"raw_data/Telugu/te_spk/te_{i}.lab", "w", encoding="utf-8").write(clean(texts[i]))
    acc += d; n += 1
val = round(n / 6); open("val_size.txt", "w").write(str(val))
print(f"{n} utterances, {acc/3600:.2f} h, val_size={val}")
