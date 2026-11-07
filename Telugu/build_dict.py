import collections, glob, json, zipfile
entries = {}
for line in open("mfa_models/Telugu_Dictionary_g2g.txt", encoding="utf-8"):
    t = line.strip().split()
    if len(t) >= 2: entries.setdefault(t[0], t[1:])
z = zipfile.ZipFile("mfa_models/Telugu_Acoustic_Model.zip")
meta = json.load(z.open([n for n in z.namelist() if n.endswith("meta.json")][0]))
model_phones = set(meta.get("phones", []))
words = collections.Counter()
for f in glob.glob("raw_data/Telugu/*/*.lab"): words.update(open(f, encoding="utf-8").read().split())
oov = [w for w in words if w not in entries]
maxlen = max(len(w) for w in entries)

def pron(w):
    ph, i = [], 0
    while i < len(w):
        for L in range(min(maxlen, len(w) - i), 0, -1):          # longest table entry first
            k = w[i:i + L]
            if k in entries: ph += entries[k]; i += L; break
        else:
            if w[i] in model_phones: ph.append(w[i]); i += 1
            else: return None
    return ph

added = 0
for w in oov:
    ph = pron(w)
    if ph is not None and (not model_phones or all(x in model_phones for x in ph)): entries[w] = ph; added += 1
print(len(words), "word types |", len(oov), "not in table | added", added)
with open("mfa_models/dict_extended.txt", "w", encoding="utf-8") as o:
    for w, p in entries.items(): o.write(f"{w}\t{' '.join(p)}\n")