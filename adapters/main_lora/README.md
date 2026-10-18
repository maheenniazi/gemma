Placeholder for the trained QLoRA adapter.

Once you train an e4b QLoRA on harvested trajectories, drop the two artifacts here:
  - adapter_config.json
  - adapter_model.safetensors

Until then, run the baseline WITHOUT an adapter by commenting out `adapter: main_lora`
in ../../agent.yaml. The agent then runs pure prompt-engineered.
