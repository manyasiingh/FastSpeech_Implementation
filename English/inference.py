import re, os, json, argparse, yaml, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.io import wavfile
from argparse import Namespace
from g2p_en import G2p
from FastSpeech_Implementation.English.utils.model import get_model, get_vocoder, vocoder_infer
from FastSpeech_Implementation.English.utils.tools import to_device
from FastSpeech_Implementation.English.text import text_to_sequence

p = argparse.ArgumentParser()
p.add_argument("-c", "--config", required=True); p.add_argument("--restore_step", type=int, required=True)
p.add_argument("--speaker_id", type=int, default=0); p.add_argument("--out", required=True)
a = p.parse_args()
cfg = yaml.load(open(a.config), Loader=yaml.FullLoader)
pre, mod, tr = cfg["preprocess"], cfg["model"], cfg["train"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TESTS = {
 "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
 "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
 "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
 "E3": "The rhyth m of the orchestra changed unexpectedly.",
 "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."}
NORM = [("10:45 AM", "ten forty five A M"), ("12 November 2026", "twelfth November twenty twenty six")]

def read_lexicon(path):
    lex = {}
    for line in open(path):
        t = re.split(r"\s+", line.strip("\n")); lex.setdefault(t[0].lower(), t[1:])
    return lex
lexicon, g2p = read_lexicon(pre["path"]["lexicon_path"]), G2p()

def to_phones(text):
    for k, v in NORM: text = text.replace(k, v)
    ph = []
    for w in re.split(r"([,;.\-\?\!\s+])", text):
        ph += lexicon[w.lower()] if w.lower() in lexicon else [x for x in g2p(w) if x != " "]
    s = re.sub(r"\{[^\w\s]?\}", "{sp}", "{" + "}{".join(ph) + "}").replace("}{", " ")
    return re.findall(r"[^{}\s]+", s), s

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
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title(title); ax.set_ylabel(ylabel); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

for sid, text in TESTS.items():
    labels, seq_str = to_phones(text); print(sid, seq_str)
    seq = np.array(text_to_sequence(seq_str, ["english_cleaners"]))[None]
    batch = to_device((None, [text], np.array([a.speaker_id]), seq, np.array([seq.shape[1]]), seq.shape[1]), device)
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
