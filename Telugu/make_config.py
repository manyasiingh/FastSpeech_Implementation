import yaml
L = "Telugu"; VAL = int(open("val_size.txt").read())
c = lambda n: yaml.safe_load(open(f"config/LJSpeech/{n}.yaml"))     # single-speaker base configs
pre, mod, tr = c("preprocess"), c("model"), c("train")
pre["dataset"] = L
pre["path"].update(corpus_path=f"./raw_data/{L}", raw_path=f"./raw_data/{L}", preprocessed_path=f"./preprocessed_data/{L}")
pre["preprocessing"]["val_size"] = VAL
mod["multi_speaker"] = False
mod["vocoder"]["speaker"] = "universal"                              # no Telugu-specific vocoder here: say so in the report
tr["path"].update(ckpt_path=f"./output/ckpt/{L}", log_path=f"./output/log/{L}", result_path=f"./output/result/{L}")
tr["optimizer"].update(batch_size=16, warm_up_step=2000, anneal_steps=[20000, 26000], anneal_rate=0.3)
tr["step"].update(total_step=30000, log_step=100, synth_step=2500, val_step=500, save_step=2500)
mod["max_seq_len"] = 1800     # clips up to 20 s (about 86 frames/s)
yaml.safe_dump({"preprocess": pre, "model": mod, "train": tr}, open("config.yaml", "w"), allow_unicode=True)
print("multi_speaker:", mod["multi_speaker"], "| sr:", pre["preprocessing"]["audio"]["sampling_rate"], "| val_size:", VAL)
