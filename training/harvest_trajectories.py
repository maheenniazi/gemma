#!/usr/bin/env python3
"""
Trajectory harvesting for the Gemma 4 Developer Agent Competition.

GOAL
----
Run the baseline agent over the PUBLIC tasks, keep ONLY the trajectories that
actually resolve the task (all FAIL_TO_PASS pass, no PASS_TO_PASS regress), and
write them out as supervised fine-tuning (SFT) examples for a QLoRA run.

This is rejection sampling: the model teaches itself from its own successes.
Because the resolution check is objective (tests pass/fail), the harvested data
is high-signal — every kept trajectory is a proof-of-correct behaviour.

WHERE THIS RUNS
---------------
A free Kaggle GPU notebook. It assumes the competition harness exposes a way to
run one task end-to-end and tell you whether it resolved. The two functions you
must wire to the real harness are marked `# WIRE:` below — everything else is
harness-agnostic and ready.

OUTPUT
------
    training/data/sft_trajectories.jsonl   # one JSON object per kept turn-sequence
Each line is a chat-formatted example ready for TRL / Unsloth SFT.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
TASKS_PATH = os.environ.get("TASKS_PATH", "tasks.jsonl")
OUT_DIR = Path(os.environ.get("OUT_DIR", "training/data"))
OUT_PATH = OUT_DIR / "sft_trajectories.jsonl"
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))         # 0 = all
RUNS_PER_TASK = int(os.environ.get("RUNS_PER_TASK", "3")) # sample k attempts, keep winners
SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH", "prompts/system.md")


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class Turn:
    """One step of an agent trajectory."""
    role: str                 # "assistant" | "tool"
    content: str              # text / tool-call / tool-result
    tool_name: str | None = None


@dataclass
class Trajectory:
    instance_id: str
    repo: str
    resolved: bool
    turns: list[Turn] = field(default_factory=list)
    wall_clock_s: float = 0.0
    n_turns: int = 0


# ----------------------------------------------------------------------------
# WIRE POINTS — connect these two to the real competition harness
# ----------------------------------------------------------------------------
def run_agent_on_task(task: dict[str, Any], system_prompt: str) -> Trajectory:
    """
    WIRE: invoke the baseline agent on a single task and capture its trajectory.

    Must:
      1. check out `task['repo']` at `task['base_commit']` in /workspace,
      2. apply `task['test_patch']` so the hidden tests exist,
      3. run the agent (agent.yaml) to completion or budget,
      4. capture every assistant turn + tool call/result as Turn objects,
      5. run the FAIL_TO_PASS / PASS_TO_PASS checks and set `resolved`.

    Replace the body below with the harness' Python entrypoint (or a subprocess
    call to its runner that emits a trajectory JSON you parse here).
    """
    raise NotImplementedError(
        "Wire run_agent_on_task() to the competition harness runner."
    )


def task_resolved(task: dict[str, Any]) -> bool:
    """
    WIRE: return True iff all FAIL_TO_PASS tests pass AND no PASS_TO_PASS regress.
    Usually the harness already returns this from the run; this stub is here for
    the case where you verify out-of-band.
    """
    raise NotImplementedError("Wire task_resolved() to the harness pass/fail check.")


# ----------------------------------------------------------------------------
# SFT formatting
# ----------------------------------------------------------------------------
def trajectory_to_sft_example(traj: Trajectory, system_prompt: str,
                              task: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a resolved trajectory into a chat-format SFT record.

    We train the model to reproduce the WINNING action sequence given the task.
    The user turn is the task framing; assistant/tool turns are the trajectory.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _format_task_prompt(task)},
    ]
    for t in traj.turns:
        if t.role == "assistant":
            messages.append({"role": "assistant", "content": t.content})
        elif t.role == "tool":
            # Tool results are context the model conditions on, not targets.
            messages.append({"role": "tool", "content": t.content})
    return {
        "instance_id": traj.instance_id,
        "repo": traj.repo,
        "n_turns": traj.n_turns,
        "wall_clock_s": round(traj.wall_clock_s, 1),
        "messages": messages,
    }


def _format_task_prompt(task: dict[str, Any]) -> str:
    """Frame a task the same way the eval harness will present it."""
    ps = (task.get("problem_statement") or "").strip()
    hints = (task.get("hints_text") or "").strip()
    parts = [f"Repository: {task['repo']}", f"Base commit: {task['base_commit']}"]
    if ps:
        parts.append(f"\nProblem statement:\n{ps}")
    if hints:
        parts.append(f"\nHints:\n{hints}")
    parts.append(
        "\nMake the hidden validation tests pass with a minimal, correct patch. "
        "Remember: the tests are the specification."
    )
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def load_tasks(path: str) -> list[dict[str, Any]]:
    tasks = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def harvest() -> None:
    system_prompt = Path(SYSTEM_PROMPT_PATH).read_text()
    tasks = load_tasks(TASKS_PATH)
    if MAX_TASKS:
        tasks = tasks[:MAX_TASKS]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kept = 0
    attempted = 0
    seen_resolved: set[str] = set()

    with open(OUT_PATH, "w") as out:
        for i, task in enumerate(tasks):
            iid = task["instance_id"]
            for run in range(RUNS_PER_TASK):
                attempted += 1
                t0 = time.time()
                try:
                    traj = run_agent_on_task(task, system_prompt)
                except NotImplementedError:
                    print("!! Harness not wired yet — see WIRE points. Aborting.")
                    return
                except Exception as e:  # a crashed run is just a non-win; continue
                    print(f"[{iid}] run {run} crashed: {e!r}")
                    continue
                traj.wall_clock_s = time.time() - t0

                if traj.resolved and iid not in seen_resolved:
                    rec = trajectory_to_sft_example(traj, system_prompt, task)
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    seen_resolved.add(iid)
                    kept += 1
                    print(f"[{iid}] RESOLVED (run {run}, {traj.n_turns} turns) -> kept")
                    break  # one win per task is enough; move on
                else:
                    status = "resolved-dup" if traj.resolved else "unresolved"
                    print(f"[{iid}] run {run}: {status}")

            if (i + 1) % 10 == 0:
                print(f"--- progress: {i+1}/{len(tasks)} tasks, "
                      f"{kept} kept / {attempted} runs ---")

    print(f"\nDONE. Kept {kept} resolved trajectories from {len(tasks)} tasks "
          f"({attempted} total runs). Written to {OUT_PATH}")
    print("Next: run training/train_qlora.py on this file to produce the adapter.")


if __name__ == "__main__":
    harvest()
