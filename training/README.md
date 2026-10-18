# Training pipeline — rejection-sampling SFT

The data pipeline that moves the score. Runs on free Kaggle GPU.

```
training/
├── harvest_trajectories.py   # run baseline over public tasks, keep ONLY resolved runs
├── train_qlora.py            # QLoRA SFT on the kept trajectories -> adapters/main_lora/
└── data/                     # sft_trajectories.jsonl lands here (git-ignored)
```

## Flow

1. **Harvest.** `python training/harvest_trajectories.py`
   - Runs the baseline agent (`agent.yaml`) on each public task, k attempts each.
   - Keeps a trajectory ONLY if it resolved (all FAIL_TO_PASS pass, no PASS_TO_PASS regress).
   - Writes chat-formatted SFT records to `training/data/sft_trajectories.jsonl`.
   - **Two WIRE points** connect it to the real harness runner — see the `# WIRE:` comments.

2. **Train.** `python training/train_qlora.py`
   - QLoRA rank-16 on `gemma-4-e4b` (keep in sync with `agent.yaml`'s model).
   - Trains on assistant tokens only (clone the winning policy, not the context).
   - Writes `adapter_config.json` + `adapter_model.safetensors` into `adapters/main_lora/`.

3. **Repackage.** Re-zip the submission so the adapter ships with it.

## Why rejection sampling
The resolution check is objective (tests pass or they don't), so every kept
trajectory is a verified example of correct behaviour. The model teaches itself
from its own successes — no human labels, no reward model, high signal.

## Free-compute notes
- e4b + rank-16 + a few hundred examples + 1-2 epochs fits free Kaggle quota and
  avoids overfitting the small task set.
- Harvest is the expensive part (many agent runs). Cache resolved trajectories and
  only re-harvest tasks you haven't yet solved.
