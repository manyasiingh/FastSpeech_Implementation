import yaml
L = "English"; VAL = int(open("val_size.txt").read())
c = lambda n: yaml.safe_load(open(f"config/LibriTTS/{n}.yaml"))     # multi-speaker base configs from the repo
pre, mod, tr = c("preprocess"), c("model"), c("train")
pre["dataset"] = L
pre["path"].update(corpus_path=f"./raw_data/{L}", raw_path=f"./raw_data/{L}", preprocessed_path=f"./preprocessed_data/{L}",
                   lexicon_path="lexicon/librispeech-lexicon.txt")
pre["preprocessing"]["val_size"] = VAL
tr["path"].update(ckpt_path=f"./output/ckpt/{L}", log_path=f"./output/log/{L}", result_path=f"./output/result/{L}")
tr["optimizer"].update(batch_size=16, warm_up_step=2000, anneal_steps=[20000, 26000], anneal_rate=0.3)
tr["step"].update(total_step=30000, log_step=100, synth_step=2500, val_step=500, save_step=2500)
yaml.safe_dump({"preprocess": pre, "model": mod, "train": tr}, open("config.yaml", "w"), allow_unicode=True)
print("multi_speaker:", mod["multi_speaker"], "| sr:", pre["preprocessing"]["audio"]["sampling_rate"], "| val_size:", VAL)
