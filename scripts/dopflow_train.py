"""Train DopFlow from the cache. Writes a result JSON with val metrics only unless --eval-subjects is given."""
import argparse, hashlib, json, sys, time, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from paths import EXP
from dopflow_data import Cache, SPLITS
from dopflow_model import DopFlow, n_params
from dopflow_doppler import band
ap = argparse.ArgumentParser()
for k, t, d in (("run", str, ""), ("stage", str, "p5"), ("steps", int, 5000), ("seed", int, 0), ("bs", int, 32), ("lr", float, 1e-3), ("lam_v", float, 1.0), ("lam_dice", float, 1.0), ("lam_f", float, 0.0), ("offs", str, "-1,0,1"), ("doppler", int, 1), ("lin", int, 1), ("dec", str, "blob"), ("slots", int, 1), ("residual", int, 0), ("gru", int, 128), ("shuffle_eval", int, 0), ("train_subj", str, ""), ("eval_subjects", str, "S02"), ("freeze_lin", int, 0), ("c", int, 8), ("vsrc", str, "dop"), ("dropout", float, 0.0), ("clip", float, 1.0), ("tau", float, 0.0), ("time_split", int, 0), ("rx", str, "")): ap.add_argument("--"+k, type=t, default=d)
a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"; offs = tuple(int(x) for x in a.offs.split(","))
D = Cache(); b, f = band(); f = f[b]; tr_s = a.train_subj.split(",") if a.train_subj else SPLITS["train"]; assert a.time_split or not set(tr_s) & {"S02"}
TS = None
if a.time_split:
    from dopflow_data import split_time
    TS = split_time(); tr_s = TS["train"]["subjects"]
tr = D.items(tr_s, TS["train"]["actions"]) if TS else D.items(tr_s); eval_s = a.eval_subjects.split(","); wtr = D.winmat(tr, offs)
ok = D.meta[tr, 0] > 50; print("train items", len(tr), "moving", ok.sum(), flush=True)
m = DopFlow(offs, bool(a.doppler), a.dec, a.slots, bool(a.residual), bool(a.lin), gru=a.gru, c=a.c, freqs=f, vsrc=a.vsrc, dropout=a.dropout, tau=a.tau).to(dev)
# ridge initialisation from training subjects only (moving frames), as in gate 1
def lfeat(ix):
    P = torch.from_numpy(np.array(D.dop[ix.reshape(-1)])).to(dev).reshape(*ix.shape, *D.dop.shape[1:]); return m.lin_feat(P).cpu().numpy()
if a.doppler and a.lin:
    X = np.concatenate([lfeat(wtr[i:i+4096]) for i in range(0, len(wtr), 4096)]); Xo = X[ok]; mu, sd = Xo.mean(0), Xo.std(0)+1e-9; A = np.c_[(Xo-mu)/sd, np.ones(len(Xo))]
    W = np.linalg.solve(A.T@A+np.eye(A.shape[1]), A.T@D.meta[tr][ok][:, 1:3]); m.set_lin(torch.tensor(mu, device=dev).float(), torch.tensor(sd, device=dev).float(), torch.tensor(W[:-1].T, device=dev).float(), torch.tensor(W[-1], device=dev).float())
if a.freeze_lin: [p.requires_grad_(False) for p in m.lin.parameters()]
np_ = n_params(m); print("params", np_, flush=True)
RX = [int(x) for x in a.rx.split(",")] if a.rx else None
if RX:
    RXD = torch.zeros(1, 1, 4, 1, 1, device=dev); RXD[:, :, RX] = 1; RXM = torch.zeros(1, 1, 12, 1, 1, 1, device=dev)
    for r_ in RX: RXM[:, :, r_*3:r_*3+3] = 1
def batch(ix, w):
    B = len(ix); dop = torch.from_numpy(np.array(D.dop[w.reshape(-1)])).to(dev).reshape(B, len(offs), *D.dop.shape[1:]); mi = torch.from_numpy(np.array(D.mi[w.reshape(-1)])).to(dev).reshape(B, len(offs), *D.mi.shape[1:])
    if RX: dop = dop*RXD; mi = mi*RXM
    g = torch.from_numpy(np.array(D.flow[ix])).to(dev); return dop, mi, g
def dice(p, y): return 1-(2*(p*y).sum((2, 3))+1)/((p+y).sum((2, 3))+1)
o = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-5); sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, a.steps)
if a.slots == 2: IM = np.load(EXP/"cache/inst_mask.npy", mmap_mode="r"); IV = np.load(EXP/"cache/inst_vel.npy")
rng = np.random.default_rng(a.seed); st = 0; skip = 0; t0 = time.time(); m.train()
while st < a.steps:
    for s_ in range(0, len(tr)-a.bs+1, a.bs):
        if s_ == 0: perm = rng.permutation(len(tr))
        k = np.sort(perm[s_:s_+a.bs])
        if D.meta[tr[k], 3].max() > 40: skip += 1; continue
        dop, mi, g = batch(tr[k], wtr[k]); out = m(dop, mi); y = (g.abs().sum(1, keepdim=True) > 0).float()
        if a.dec == "pixel": l = ((out["final"]-g)**4).mean()
        elif a.slots == 2:
            # permutation-invariant loss against the two largest connected components
            M = out["mask"]; tm = torch.from_numpy(np.array(IM[tr[k]])).to(dev).float(); tv = torch.from_numpy(IV[tr[k]].copy()).to(dev); Md = M[:, :, ::2, ::2].clamp(1e-6, 1-1e-6); V_ = out["v"]; has = (tv[..., 0] > 50).float()
            def cost(p):
                c = 0.
                for q, r in enumerate(p):
                    c = c+F.binary_cross_entropy(Md[:, q], tm[:, r], reduction="none").mean((1, 2))+a.lam_dice*dice(Md[:, q:q+1], tm[:, r:r+1])[:, 0]+a.lam_v*(F.smooth_l1_loss(V_[:, q], tv[:, r, 1:], reduction="none").mean(-1)*has[:, r])
                return c
            l = torch.minimum(cost((0, 1)), cost((1, 0))).mean()
            if a.lam_f: l = l+a.lam_f*(out["final"]-g).abs().sum(1).mean()
        else:
            M = out["mask"]; Ms = M.sum(1, keepdim=True).clamp(0, 1) if a.slots > 1 else M
            l = F.binary_cross_entropy(Ms.clamp(1e-6, 1-1e-6), y)+a.lam_dice*dice(Ms, y).mean(); t = torch.from_numpy(D.meta[tr[k]][:, 1:3].copy()).to(dev); ne = torch.from_numpy(D.meta[tr[k], 0] > 50).to(dev)
            if ne.any():
                Vm = out["v"].mean(1) if a.slots > 1 else out["v"][:, 0]; l = l+a.lam_v*F.smooth_l1_loss(Vm[ne], t[ne])
            if a.lam_f: l = l+a.lam_f*(out["final"]-g).abs().sum(1).mean()
        o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), a.clip); o.step(); sch.step(); st += 1
        if st % 1000 == 0: print(st, round(l.item(), 4), round(time.time()-t0), flush=True)
        if st >= a.steps: break
run = a.run or f"{a.stage}_{int(time.time())}"; R = EXP/"runs"/run; R.mkdir(parents=True, exist_ok=True); torch.save(m.state_dict(), R/"model.pth")
def evaluate(subj, shuffle=False):
    m.eval(); it = D.items(subj, TS["test"]["actions"]) if TS else D.items(subj); w = D.winmat(it, offs); seq = np.array([D.keys[i][0]+D.keys[i][1] for i in it]); E = []; Vp = []
    if shuffle:
        g_ = np.random.default_rng(0)
        for s_ in np.unique(seq):
            q = np.where(seq == s_)[0]; w[q] = w[g_.permutation(q)]
    with torch.no_grad():
        for i in range(0, len(it), 64):
            dop, mi, g = batch(it[i:i+64], w[i:i+64]); out = m(dop, mi); p = out["final"]; d = p-g; e = (d**2).sum(1).sqrt(); mk = g.abs().sum(1) >= .5
            E.append(torch.stack([e.mean((1, 2)), (e*mk).sum((1, 2))/mk.sum((1, 2)).clamp(min=1), (e*~mk).sum((1, 2))/(~mk).sum((1, 2)).clamp(min=1), (d**4).sum(1).sqrt().mean((1, 2)), mk.sum((1, 2)).float()], 1).cpu())
            if out.get("v") is not None:
                Vp.append((out["v"].mean(1) if a.slots > 1 else out["v"][:, 0]).cpu())
    E = torch.cat(E).numpy(); r = dict(n=len(it), EPE=float(E[:, 0].mean()), EPE_M=float(E[:, 1].mean()), EPE_S=float(E[:, 2].mean()), EPE_A=float(E[:, 3].mean()))
    mv = D.meta[it, 0] > 50
    if Vp:
        Vp = torch.cat(Vp).numpy(); T = D.meta[it][:, 1:3]; r2 = lambda y, h: float(1-((y-h)**2).sum()/((y-y.mean())**2).sum()); ang = np.degrees(np.abs(np.angle(np.exp(1j*(np.arctan2(Vp[mv, 1], Vp[mv, 0])-np.arctan2(T[mv, 1], T[mv, 0]))))))
        r.update(R2_u=r2(T[mv, 0], Vp[mv, 0]), R2_v=r2(T[mv, 1], Vp[mv, 1]), angle=float(np.median(ang)))
    return r
res = dict(run=run, args=vars(a), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), params=np_, secs=time.time()-t0, skipped_batches=skip, peak_mem_gb=torch.cuda.max_memory_allocated()/1e9, eval={s: evaluate([s]) for s in eval_s})
if a.shuffle_eval: res["eval_shuffled"] = {s: evaluate([s], True) for s in eval_s}
Out = EXP/"results/dopflow"/a.stage; Out.mkdir(parents=True, exist_ok=True); json.dump(res, open(Out/f"{run}.json", "w"), indent=1); print(json.dumps({k: res[k] for k in ("run", "params", "secs", "eval")}), flush=True)
