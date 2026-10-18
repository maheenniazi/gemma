# Submission layout — Gemma 4 Developer Agent Competition

Baseline, prompt-engineered agent. Scaffolded from the confirmed starter-kit format.

```
submission/
├── agent.yaml                 # root agent (ADK Agent Config); model + tools + includes
├── eval_config.yaml           # optional per-task budget caps
├── configs/
│   └── sampling.yaml          # generation config (temp 0, high output cap)
├── prompts/
│   ├── system.md              # root agent operating procedure (the strategy)
│   └── analyzer.md            # read-only sub-agent instruction
├── sub_agents/
│   └── code_analyzer.yaml     # read-only exploration agent (skip_summarization)
└── adapters/
    └── main_lora/             # (empty until QLoRA is trained)
        ├── adapter_config.json
        └── adapter_model.safetensors
```

## Decisions baked in
- **Model: `gemma-4-e4b-it-qat-w4a16-ct`** — chosen for the free-Kaggle plan (fast
  decode -> more tasks finished in 12h; cheap to QLoRA). Flip to `31b` in BOTH
  `agent.yaml` and `sub_agents/code_analyzer.yaml` if you want to A/B it.
- **Strategy (system.md):** tests are the literal spec; detect bugfix vs new-file;
  localize -> minimal patch -> verify -> stop.
- **Anti-footgun rules encoded:** scratch in /tmp; never edit tests/conftest/pytest.ini;
  minimal diff; incremental edits; stop when green.

## Not yet done
- `adapters/main_lora/` is a placeholder. The baseline runs WITHOUT an adapter
  (comment out `adapter: main_lora` in agent.yaml for a pure-prompt run).
- Reconcile `eval_config.yaml` keys against the real starter-kit schema.
- Harvest trajectories from the public tasks, then QLoRA an e4b adapter and drop it here.
