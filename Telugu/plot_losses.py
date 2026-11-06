import sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
log, out = sys.argv[1], sys.argv[2]
tags = ["total_loss", "mel_loss", "mel_postnet_loss", "pitch_loss", "energy_loss", "duration_loss"]
for split, fn in [("train", "training_loss.png"), ("val", "validation_loss.png")]:
    ea = EventAccumulator(f"{log}/{split}", size_guidance={"scalars": 0}); ea.Reload()
    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    for a_, t in zip(ax.flat, tags):
        s = ea.Scalars(f"Loss/{t}"); a_.plot([x.step for x in s], [x.value for x in s]); a_.set_title(t); a_.set_xlabel("step")
    fig.suptitle(f"{split} loss"); fig.tight_layout(); fig.savefig(f"{out}/{fn}", dpi=150)
