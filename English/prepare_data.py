import os, io, re, random, unicodedata, collections
import numpy as np, soundfile as sf, librosa
from datasets import load_dataset, Audio

DATASET, SPLIT = "ai4bharat/Svarah", "test"
TEXT_COL, AUDIO_COL, DUR_COL = "text", "audio_filepath", "duration"
SPK_COLS = ["gender", "age-group", "primary_language", "native_place_state",
            "native_place_district", "highest_qualification", "job_category", "occupation_domain"]
SR, SEED, TARGET_H, MIN_S, MAX_S = 22050, 42, 6.0, 1.5, 11.0    # 6h total = 5h train + 1h val
random.seed(SEED)

ds = load_dataset(DATASET, split=SPLIT)
ds = ds.cast_column(AUDIO_COL, Audio(decode=False))      # decode lazily, only for selected clips

texts, durs = ds[TEXT_COL], ds[DUR_COL]
cols = {c: ds[c] for c in SPK_COLS}
key = [tuple(str(cols[c][i]) for c in SPK_COLS) for i in range(len(ds))]
ids = {k: j for j, k in enumerate(sorted(set(key)))}
spk = [f"spk{ids[k]:03d}" for k in key]                  # speaker proxy

def clean(t):
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    if re.search(r"\d", t): return ""                    # drop transcripts with digits (e.g. "11th")
    return re.sub(r"\s+", " ", re.sub(r"[^A-Za-z' ]", " ", t)).strip()

ok = [i for i in range(len(ds)) if MIN_S <= durs[i] <= MAX_S and len(clean(texts[i])) > 3]
cnt = collections.Counter(spk[i] for i in ok)
ok = [i for i in ok if cnt[spk[i]] >= 5]                 # drop proxy-speakers with <5 usable utterances
print(len(ds), "total ->", len(ok), "usable utterances,", len(set(spk[i] for i in ok)), "speaker proxies")
random.shuffle(ok)

acc, n, used = 0.0, 0, set()
for i in ok:
    if acc >= TARGET_H * 3600: break
    a = ds[i][AUDIO_COL]
    y, sr = sf.read(io.BytesIO(a["bytes"])) if a.get("bytes") else sf.read(a["path"])
    if y.ndim > 1: y = y.mean(1)
    y = y.astype(np.float32)
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    os.makedirs(f"raw_data/English/{spk[i]}", exist_ok=True)
    sf.write(f"raw_data/English/{spk[i]}/svarah_{i}.wav", y, SR)
    open(f"raw_data/English/{spk[i]}/svarah_{i}.lab", "w").write(clean(texts[i]))
    acc += len(y) / SR; n += 1; used.add(spk[i])
val = round(n / 6); open("val_size.txt", "w").write(str(val))
print(f"{n} utterances, {acc/3600:.2f} h, {len(used)} speaker proxies, val_size={val}")
