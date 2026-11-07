import glob, json, tgt
ph = set()
for f in glob.glob("preprocessed_data/Telugu/TextGrid/*/*.TextGrid"):
    ph |= {i.text for i in tgt.io.read_textgrid(f).get_tier_by_name("phones")}
ph -= {"", "sp", "spn", "sil"}
dict_ph = {x for l in open("mfa_models/dict_extended.txt", encoding="utf-8") for x in l.rstrip("\n").split("\t", 1)[-1].split()}
allph = sorted(ph | dict_ph)
json.dump(allph, open("text/telugu_phones.json", "w", encoding="utf-8"), ensure_ascii=False)
print(len(ph), "phones in alignments |", len(allph), "total")
s = open("text/symbols.py").read()
if "telugu_phones.json" not in s:
    open("text/symbols.py", "a").write('''

# Telugu: phone set from the IndicMFA dictionary and alignments
import json as _json, os as _os
_p = _os.path.join(_os.path.dirname(__file__), "telugu_phones.json")
if _os.path.exists(_p):
    symbols = ([_pad] + list(_special) + list(_punctuation) + list(_letters)
               + ["@" + _s for _s in _json.load(open(_p, encoding="utf-8"))] + _silences)
''')