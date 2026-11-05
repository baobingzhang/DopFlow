"""Build per-frame caches for birdview+: Doppler band spectra, mask inputs, flow, targets. Output in cache/."""
import multiprocessing as mp; mp.set_start_method("fork", force=True)
import json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from paths import EXP
from dopflow_doppler import spectra, mask_input, band
import os
AP = int(os.environ.get("DOPFLOW_AP", 100)); HZ = int(os.environ.get("DOPFLOW_HZ", 1000)); ST = 1000//HZ; VIEW = os.environ.get("DOPFLOW_VIEW", "birdviewplus"); LINK = os.environ.get("DOPFLOW_LINK", "")
root = EXP/"data/WiFlow"/VIEW; C = Path(os.environ.get("DOPFLOW_CACHE", EXP/"cache")); C.mkdir(exist_ok=True); ONLY = os.environ.get("DOPFLOW_ONLY", ""); rd = lambda p: torch.load(p, map_location="cpu", weights_only=True)
fr = [(S.name, A.name, int(p.stem[5:]), p) for S in sorted(root.glob("S*")) if not ONLY or S.name in ONLY.split(",") for A in sorted(S.glob("A*")) if (A/"csi").exists() for p in sorted((A/"csi").glob("*.pt"))]
DEV = "cuda" if torch.cuda.is_available() else "cpu"; print("frames", len(fr), flush=True); N = len(fr); b, f = band()
class D(torch.utils.data.Dataset):
    def __len__(s): return N
    def __getitem__(s, i):
        S, A, k, p = fr[i]; c = rd(p).squeeze(0)[..., :AP:ST]; fl = p.parent.parent/"flow"/p.name; has = fl.exists()
        g = torch.zeros(2, 1, 1) if LINK else (rd(fl).float() if has else torch.zeros(2, 128, 168)); g = g*((g**2).sum(0, keepdim=True).sqrt() > 1)
        return c, g, has, i
M = lambda n, sh, t="float32": np.lib.format.open_memmap(C/f"{n}.npy", mode="w+", dtype=t, shape=(N,)+sh)
T = len(range(0, AP, ST)); T2 = T//max(1, T//20); dop = M("dop", (4, 2, int(b.sum()))); mi = M("mask_in", (12, 2, 28, T2)); flow = None if LINK else M("flow", (2, 128, 168)); meta = np.zeros((N, 6), np.float32)
for c, g, has, idx in torch.utils.data.DataLoader(D(), batch_size=64, num_workers=16):
    c = c.to(DEV); j = idx.numpy(); dop[j] = spectra(c)[..., torch.from_numpy(b).to(DEV)].cpu().numpy(); mi[j] = mask_input(c).cpu().numpy()
    if LINK: continue
    flow[j] = g.numpy()
    mk = g.abs().sum(1) > 0; nz = mk.sum((1, 2)); meta[j, 0] = nz.numpy(); meta[j, 1] = ((g[:, 0]*mk).sum((1, 2))/nz.clamp(min=1)).numpy(); meta[j, 2] = ((g[:, 1]*mk).sum((1, 2))/nz.clamp(min=1)).numpy()
    meta[j, 3] = (g**2).sum(1).sqrt().flatten(1).max(1).values.numpy(); meta[j, 4] = has.numpy()
for a in (dop, mi, flow):
    if a is not None: a.flush()
if LINK:
    for n in ("flow.npy", "meta.npy", "keys.json"):
        if not (C/n).exists(): os.symlink(Path(LINK).resolve()/n, C/n)
else:
    np.save(C/"meta.npy", meta); json.dump([x[:3] for x in fr], open(C/"keys.json", "w"))
print("done", flush=True)
