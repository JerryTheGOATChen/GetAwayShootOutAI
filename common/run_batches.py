"""Train the batches in BATCHES (common/workers.py) one after another: PPO, then the genetic algorithm (or just
the batch's `methods`), then the chart, for each. Leave it running (overnight); the PC is kept awake until it is done.

    py -m common.run_batches                  # all of them
    py -m common.run_batches 02-your-run      # just these ones

Run it again after stopping it (Ctrl+C) and it carries on where it stopped: a finished trainer just ends, an
unfinished one resumes from its checkpoint. To train a batch again from scratch, delete its folder first.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from common.workers import BATCHES, keep_awake, keep_your_run

STEPS = "50e6"  # unless the batch says otherwise (its steps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batches", nargs="*", default=list(BATCHES), help="which batches (default: all, in order)")
    args = ap.parse_args()
    unknown = [b for b in args.batches if b not in BATCHES]
    if unknown:
        ap.exit(1, f"unknown batch {', '.join(unknown)}; the batches are {', '.join(BATCHES)}\n")

    keep_awake()  # the trainers hold it too; this covers the moments in between
    for batch in args.batches:
        keep_your_run(batch)  # now, so they all practise from the same run even if you set a new best meanwhile
    try:
        for batch in args.batches:
            print(f"\n=== batch {batch} ===", flush=True)
            env = {**os.environ, "GETAWAY_BATCH": batch}
            methods = BATCHES[batch].get("methods", ("ppo", "genetic"))
            steps = str(BATCHES[batch].get("steps", STEPS))
            for cmd in [["-m", f"{m}.train", "--steps", steps, "--resume"] for m in methods] + [["-m", "common.compare", batch]]:
                proc = subprocess.Popen([sys.executable, "-u", *cmd], env=env)
                try:
                    code = proc.wait()
                except KeyboardInterrupt:
                    proc.wait()  # the trainer got the Ctrl+C too, and is saving its progress
                    raise
                if code != 0:
                    sys.exit(f"stopped during batch {batch}. Run py -m common.run_batches again to carry on.")
        print("\nall batches done. Charts: runs/<batch>/compare.png")
    except KeyboardInterrupt:
        print("\nstopped. Run py -m common.run_batches again to carry on.")


if __name__ == "__main__":
    main()
