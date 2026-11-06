import re, os, json, argparse, yaml, unicodedata, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.io import wavfile
from argparse import Namespace
from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device
from text import text_to_sequence
from text.symbols import symbols

p = argparse.ArgumentParser()
p.add_argument("-c", "--config", required=True); p.add_argument("--restore_step", type=int, required=True)
p.add_argument("--dict", default="mfa_models/dict_extended.txt"); p.add_argument("--out", required=True)
a = p.parse_args()
cfg = yaml.load(open(a.config), Loader=yaml.FullLoader)
pre, mod, tr = cfg["preprocess"], cfg["model"], cfg["train"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S = set(symbols)

# VERIFY these against the original assignment PDF: its text extraction mangled the Telugu
TESTS = {
 "T1": "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
 "T2a": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
 "T2b": "ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చింది.",
 "T3": "పులి అడవిలో నిశ్శబ్దంగా నడిచింది.",
 "T4": "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."}
NORM = [("2026", "రెండు వేల ఇరవై ఆరు"), ("12వ", "పన్నెండవ"), ("10:45", "పది నలభై ఐదు")]   # hand-written for the test sentences

D = {}
for line in open(a.dict, encoding="utf-8"):
    w, _, rest = line.rstrip("\n").partition("\t"); D[w] = rest.split()


TABLE = {}
for line in open("mfa_models/Telugu_Dictionary_g2g.txt", encoding="utf-8"):
    t = line.strip().split()
    if len(t) >= 2: TABLE.setdefault(t[0], t[1:])
MAXLEN = max(map(len, TABLE))
def g2g(w):
    ph, i = [], 0
    while i < len(w):
        for L in range(min(MAXLEN, len(w) - i), 0, -1):
            k = w[i:i + L]
            if k in TABLE: ph += TABLE[k]; i += L; break
        else: ph.append(w[i]); i += 1
    return ph

def to_phones(text):
    for k, v in NORM: text = text.replace(k, v)
    text = unicodedata.normalize("NFC", re.sub(r"[\u200c\u200d]", "", text))
    ph, unk = [], []
    for tok in re.findall(r"[^\s,;.?!]+|[,;.?!]", text):
        if tok in ",;.?!": ph.append("sp"); continue
        if tok in D: ph += [x for x in D[tok] if "@" + x in S]
        else: unk.append(tok); ph += [x for x in g2g(tok) if "@" + x in S]
    while ph and ph[-1] == "sp": ph.pop()
    return ph, unk

tf = [f for f in fm.findSystemFonts() if any(k in os.path.basename(f).lower() for k in ("telugu", "gautami", "lohit-te"))]
NATIVE = bool(tf)
if NATIVE:
    fm.fontManager.addfont(tf[0]); plt.rcParams["font.family"] = fm.FontProperties(fname=tf[0]).get_name()
else:
    from indic_transliteration import sanscript
def label(x):
    if NATIVE or x in ("sp", "spn", "sil"): return x
    s = sanscript.transliterate(x, sanscript.TELUGU, sanscript.IAST)
    return s if s.strip() else "·" + "".join(f"{ord(c):X}" for c in x)
print("native Telugu font:", NATIVE)

model = get_model(Namespace(restore_step=a.restore_step), (pre, mod, tr), device, train=False)
vocoder = get_vocoder(mod, device)
stats = json.load(open(os.path.join(pre["path"]["preprocessed_path"], "stats.json")))
hop, SR = pre["preprocessing"]["stft"]["hop_length"], pre["preprocessing"]["audio"]["sampling_rate"]
for d in ["test_generated_audios", "variance_adaptor_plots/duration", "variance_adaptor_plots/pitch", "variance_adaptor_plots/energy"]:
    os.makedirs(f"{a.out}/{d}", exist_ok=True)

def plot_track(vals, durs, labels, title, ylabel, path, bars=False):
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 3.5))
    if bars: ax.bar(np.arange(len(labels)), vals); x = np.arange(len(labels))
    else:
        edges = np.concatenate([[0], np.cumsum(durs)])
        ax.plot(np.repeat(vals, durs.astype(int)))
        for e in edges: ax.axvline(e, color="gray", lw=0.3)
        x = (edges[:-1] + edges[1:]) / 2
    ax.set_xticks(x); ax.set_xticklabels([label(l) for l in labels], rotation=90, fontsize=7)
    ax.set_title(title); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

for sid, text in TESTS.items():
    labels, unk = to_phones(text)
    print(sid, len(labels), "phones | words not in dictionary:", unk)
    seq = np.array(text_to_sequence("{" + " ".join(labels) + "}", ["english_cleaners"]))[None]
    batch = to_device((None, [text], np.array([0]), seq, np.array([seq.shape[1]]), seq.shape[1]), device)
    with torch.no_grad(): out = model(*(batch[2:]))
    mel, p_pred, e_pred, d_round, mel_len = out[1], out[2], out[3], out[5], out[9]
    wav = vocoder_infer(mel.transpose(1, 2), vocoder, mod, pre, lengths=(mel_len * hop).cpu().numpy())[0]
    wavfile.write(f"{a.out}/test_generated_audios/{sid}.wav", SR, wav)
    d, pv, ev = d_round[0].cpu().numpy(), p_pred[0].cpu().numpy(), e_pred[0].cpu().numpy()
    if pre["preprocessing"]["pitch"]["normalization"]: pv = pv * stats["pitch"][3] + stats["pitch"][2]
    if pre["preprocessing"]["energy"]["normalization"]: ev = ev * stats["energy"][3] + stats["energy"][2]
    V = f"{a.out}/variance_adaptor_plots"
    plot_track(d, d, labels, f"{sid} predicted duration", "frames", f"{V}/duration/{sid}.png", bars=True)
    plot_track(pv, d, labels, f"{sid} predicted pitch (F0)", "Hz", f"{V}/pitch/{sid}.png")
    plot_track(ev, d, labels, f"{sid} predicted energy", "energy", f"{V}/energy/{sid}.png")
