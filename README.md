# sharpa-wam

Running [DexTacWAM](https://dextacwam.github.io/), a visuo-tactile world-action model, on a single Franka FR3 arm with a Sharpa Wave hand. Currently simulation only (MuJoCo).

The policy runs as a server on the host GPU, using DexTacWAM's own deploy server. The robot/sim client runs inside the FR3Py devcontainer and talks to the server over localhost.

## Notes

- **No ready-made policy is available.** The DexTacWAM authors released only the stage 1 tactile encoder. They did not release the stage 2 world model or the stage 3 action expert, because those are specific to their robot and tasks. Running it on a new robot means collecting data and training those stages yourself.
- **Data requirement.** The authors report needing about 100 teleoperated demonstrations per task.
- **GPU memory.** Inference needs about 15 GB of GPU memory, more than the machine attached to the Franka/Sharpa setup can provide. For real-robot experiments, host the policy on a separate GPU server and stream observations and actions to it over the network (for example through an SSH tunnel). DexTacWAM's server/client split already supports this.

## Sim proof of concept: picking up a bar

- **Task:** the FR3 + Sharpa picks a red bar (pole) up off two posts in MuJoCo.
- **Data:** 100 scripted demonstrations. Each frame stores a head-camera PNG, a 1×5×192×256 tactile array (synthetic, rendered from contact forces), a 75-D action and a 45-D state.
- **Training:** only the action-expert weights are trained. The 2.1B-parameter Genie-Envisioner base world model is frozen, along with the released tactile encoder, so training fits on a single L40S (about 18 GB per GPU). Stage 2 world-model fine-tuning was skipped. The run took 4,000 steps, about 45 minutes on two L40S.
- **Result:** the 4,000-step checkpoint picked up the bar in **20 of 20** trials, at bar positions not in the training data. See [`dextacwam_fr3_sharpa_demo.mp4`](FR3Py/examples/dextacwam/dextacwam_fr3_sharpa_demo.mp4).

Caveats: the demos are scripted and the tactile is synthetic. This shows the pipeline runs on this robot; it is not a benchmark, and it doesn't yet show that touch helps (no vision-only comparison).

## Reproduce

Weights, datasets and checkpoints are not in git. Download the pretrained weights as described in `DexTacWAM/README.md` (Installation).

```bash
# 1. Collect demos (in the devcontainer)
docker exec -e MUJOCO_GL=egl fr3py-sharpa-container bash -c \
  'cd /home/FR3Py/examples/dextacwam && python3 collect_demos.py --n 100'

# 2. Convert, compute stats, write config (host, DexTacWAM venv)
cd DexTacWAM && source .venv/bin/activate
python fr3_integration/npz_to_lerobot.py --src ../FR3Py/examples/dextacwam/data_raw \
  --out data/datasets_lerobot/fr3_sharpa_bar_pick \
  --prompt "Pick up the red bar from the two posts with the right hand."
python scripts/get_statistics.py --relative --arm-layout right_only \
  --data_root data/datasets_lerobot/fr3_sharpa_bar_pick/data/chunk-000 \
  --data_name fr3_sharpa_bar_pick --data_type eef --action_key actions --state_key state \
  --save_path fr3_integration/configs/fr3_sharpa_bar_pick_relative_stats.json \
  --n-previous 4 --action-chunk 54
python fr3_integration/make_bar_pick_config.py

# 3. Train the stage 3 action expert
torchrun --nproc_per_node=2 main.py \
  --config_file fr3_integration/configs/fr3_sharpa_bar_pick_stage3.yaml \
  --runner_class_path runner/tactile_dit_trainer.py --runner_class TactileDiTTrainer

# 4. Evaluate closed-loop in sim (videos go to FR3Py/examples/dextacwam/rollouts/)
STEP=4000 bash fr3_integration/eval_checkpoint.sh
```

torchrun may exit with code 120 after training finishes; this is a known shutdown glitch noted in the DexTacWAM README. The run succeeded if the final `step_*` checkpoint exists.
