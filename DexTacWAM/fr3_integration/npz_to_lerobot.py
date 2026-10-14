#!/usr/bin/env python3
"""Convert FR3 + Sharpa sim demos (.npz from FR3Py/examples/dextacwam/collect_demos.py)
into the LeRobot parquet layout DexVTAMDataset reads.

Output: <out>/{meta/{info.json,tasks.jsonl,episodes.jsonl}, data/chunk-000/episode_NNNNNN.parquet}
Columns per frame: head_img {bytes: PNG}, tactile (1,5,H,W) nested uint8, actions (75,),
state (45,), frame_index, episode_index -- the same schema as the test fixture in
tests/test_dex_vtam_cache_consistency.py.
"""

import argparse
import io
from multiprocessing import Pool
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_dex_vtam_cache_consistency import _to_nested_object_array_4d  # noqa: E402


def _convert_episode(ep, f, out):
    z = np.load(f)
    rows = []
    for t in range(len(z["state"])):
        buf = io.BytesIO()
        Image.fromarray(z["head_img"][t]).save(buf, format="PNG")
        rows.append({
            "head_img": {"bytes": buf.getvalue(), "path": ""},
            "tactile": _to_nested_object_array_4d(z["tactile"][t]),
            "actions": z["actions"][t].astype(np.float32),
            "state": z["state"][t].astype(np.float32),
            "frame_index": t,
            "episode_index": ep,
        })
    pd.DataFrame(rows).to_parquet(out / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    print(f"[convert] {f.name} -> episode {ep} ({len(rows)} frames)", flush=True)
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="dir of episode_*.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", required=True)
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("episode_*.npz"))

    jobs = [(ep, f, out) for ep, f in enumerate(files)]
    with Pool(min(32, len(jobs))) as pool:
        lengths = pool.starmap(_convert_episode, jobs)
    episodes = [{"episode_index": ep, "tasks": [args.prompt], "length": n}
                for ep, n in enumerate(lengths)]

    with open(out / "meta" / "info.json", "w") as fh:
        json.dump({"total_chunks": 1, "chunks_size": 1000, "fps": 30}, fh)
    with open(out / "meta" / "tasks.jsonl", "w") as fh:
        fh.write(json.dumps({"task_index": 0, "task": args.prompt}) + "\n")
    with open(out / "meta" / "episodes.jsonl", "w") as fh:
        for e in episodes:
            fh.write(json.dumps(e) + "\n")


if __name__ == "__main__":
    main()
