"""Evaluate a saved DopFlow run on given subjects; per-frame metrics, per action, optional causal whole-field EMA. usage: dopflow_eval.py RESULT_JSON SUBJECTS OUT.json [--ema]"""
import json, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from paths import EXP
from dopflow_data import Cache
from dopflow_model import DopFlow
from dopflow_doppler import band
J = json.load(open(sys.argv[1])); subj = sys.argv[2].split(","); out = sys.argv[3]; ema = "--ema" in sys.argv; a = J["args"]; offs = tuple(int(x) for x in a["offs"].split(",")); dev = "cuda" if torch.cuda.is_available() else "cpu"
D = Cache(); b, f = band(); m = DopFlow(offs, bool(a["doppler"]), a["dec"], a["slots"], bool(a["residual"]), bool(a["lin"]), gru=a["gru"], c=a["c"], freqs=f[b], vsrc=a.get("vsrc", "dop"), dropout=a.get("dropout", 0.0), tau=a.get("tau", 0.0)).to(dev); m.load_state_dict(torch.load(EXP/"runs"/J["run"]/"model.pth", map_location=dev, weights_only=True)); m.eval()
groups = dict(collect=range(1, 5), knee=range(5, 9), mill=range(9, 13), together=range(13, 17), fast=range(17, 21), slow=range(21, 29), waving=range(29, 33), off=range(33, 37), void=range(37, 45))
def per(p, g):
    d = p-g; e = (d**2).sum(1).sqrt(); k = g.abs().sum(1) >= .5
    return torch.stack([e.mean((1, 2)), (e*k).sum((1, 2))/k.sum((1, 2)).clamp(min=1), (e*~k).sum((1, 2))/(~k).sum((1, 2)).clamp(min=1), (d**4).sum(1).sqrt().mean((1, 2))], 1).cpu().numpy()
it = D.items(subj); w = D.winmat(it, offs); seq = [D.keys[i][0]+D.keys[i][1] for i in it]; E = []; Ee = {x: [] for x in (.8, .6, .4, .2)} if ema else {}; prev = {}; ps = None
with torch.no_grad():
    for i in range(0, len(it), 64):
        B = len(it[i:i+64]); dop = torch.from_numpy(np.array(D.dop[w[i:i+64].reshape(-1)])).to(dev).reshape(B, len(offs), *D.dop.shape[1:]); mi = torch.from_numpy(np.array(D.mi[w[i:i+64].reshape(-1)])).to(dev).reshape(B, len(offs), *D.mi.shape[1:]); g = torch.from_numpy(np.array(D.flow[it[i:i+64]])).to(dev); p = m(dop, mi)["final"]; E.append(per(p, g))
        if ema:
            for x in Ee:
                y = p.clone()
                for j in range(B):
                    s = seq[i+j]
                    if s in prev.setdefault(x, {}): y[j] = x*p[j]+(1-x)*prev[x][s]
                    prev[x][s] = y[j]; 
                Ee[x].append(per(y, g))
E = np.concatenate(E); acts = np.array([D.keys[i][1] for i in it]); S = lambda a_: dict(EPE=float(a_[:, 0].mean()), EPE_M=float(a_[:, 1].mean()), EPE_S=float(a_[:, 2].mean()), EPE_A=float(a_[:, 3].mean()), n=len(a_))
r = dict(run=J["run"], subjects=subj, all=S(E), per_action={k: S(E[np.isin(acts, [f"A{x:02d}" for x in v])]) for k, v in groups.items() if np.isin(acts, [f"A{x:02d}" for x in v]).any()})
if ema: r["ema"] = {str(x): S(np.concatenate(v)) for x, v in Ee.items()}
json.dump(r, open(out, "w"), indent=1); np.savez(out.replace(".json", ".npz"), metrics=E, seq=np.array(seq)); print(json.dumps(r["all"]))
