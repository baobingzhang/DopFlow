"""In-memory (mmap) batches from cache/. Item rule matches wf_data.Pairs: frame f and f+1 have csi, flow, class. Windows f-1, f, f+1 with nearest padding."""
import json, os, numpy as np, torch
from pathlib import Path
from paths import EXP
ALL = [f"A{i:02d}" for i in range(1, 45)]
SPLITS = {"train": ["S00", "S01", "S03", "S04", "S05", "S07", "S09", "S10"], "val": ["S02"], "test": ["S06", "S08"]}
class Cache:
    def __init__(s, root=None):
        C = Path(root or os.environ.get("DOPFLOW_CACHE", EXP/"cache")); s.dop = np.load(C/"dop.npy", mmap_mode="r"); s.mi = np.load(C/"mask_in.npy", mmap_mode="r"); s.flow = np.load(C/"flow.npy", mmap_mode="r"); s.meta = np.load(C/"meta.npy")
        s.keys = [tuple(k) for k in json.load(open(C/"keys.json"))]; s.pos = {k: i for i, k in enumerate(s.keys)}
    def items(s, subjects, actions=ALL):
        """indices f whose flow(f) and flow(f+1) exist (the class files exist with flow), same rule and count as Pairs."""
        out = []
        for i, (S, A, f) in enumerate(s.keys):
            if S in subjects and A in actions and s.meta[i, 4] > 0:
                j = s.pos.get((S, A, f+1))
                if j is not None and s.meta[j, 4] > 0: out.append(i)
        return np.array(out)
    def win(s, i, offs):
        S, A, f = s.keys[i]; r = []
        for o in offs:
            st = -1 if o > 0 else 1; j = None
            for q in range(abs(o)+1):
                j = s.pos.get((S, A, f+o+st*q))
                if j is not None: break
            r.append(j)
        return r
    def winmat(s, idx, offs): return np.array([s.win(int(i), offs) for i in idx])
def split_time():
    import yaml
    c = yaml.safe_load(open(EXP/"wiflow_patched/dataset_configs/seemo_config_E9_time.yml")); return c
