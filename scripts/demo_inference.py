"""Run a trained DopFlow on one real frame from the WiFlow dataset.

The first run downloads three CSI windows and one flow label (about 2.6 MB) from the
official dataset archive, using HTTP range requests. Subject S05 was held out from the training of the included weights.
"""
import hashlib, json, struct, sys, time, urllib.request, zlib, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dopflow_doppler import spectra, mask_input, band
from dopflow_model import DopFlow, n_params
ROOT = Path(__file__).resolve().parents[1]; CACHE = ROOT/"demo_data"; MAN = json.load(open(ROOT/"demo/sample_manifest.json")); WEIGHTS = ROOT/"weights/dopflow_S05_seed0.pth"
def get(a, b):
    r = urllib.request.Request(MAN["url"], headers={"Range": f"bytes={a}-{b}", "User-Agent": "DopFlow-demo"})
    with urllib.request.urlopen(r, timeout=120) as f:
        assert f.status == 206, "the server did not honour the range request"; return f.read()
def fetch(e):
    dst = CACHE/Path(e["member"]).parent.parent.name/Path(e["member"]).parent.name/Path(e["member"]).name
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() == e["sha256"]: return dst
    print("downloading", e["member"], flush=True); h = get(e["offset"], e["offset"]+29); assert h[:4] == b"PK\x03\x04"; nl, el = struct.unpack("<HH", h[26:30])
    start = e["offset"]+30+nl+el; raw = get(start, start+e["compressed"]-1); data = raw if e["method"] == 0 else zlib.decompress(raw, -15)
    assert hashlib.sha256(data).hexdigest() == e["sha256"], "checksum mismatch: " + e["member"]
    dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(data); return dst
def flow_rgb(F, clip=8.):
    """Hue is the direction and saturation the magnitude, clipped at `clip` px per frame."""
    from matplotlib.colors import hsv_to_rgb
    a = (np.arctan2(F[1], F[0])/(2*np.pi)) % 1; m = np.clip(np.hypot(F[0], F[1])/clip, 0, 1); return hsv_to_rgb(np.stack([a, m, np.ones_like(m)], -1))
def main():
    try: paths = [fetch(e) for e in MAN["files"]]
    except Exception as x: sys.exit(f"Could not download the sample from {MAN['url']}: {x}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rd = lambda p: torch.load(p, map_location=dev, weights_only=True)
    csi = [rd(p).squeeze(0)[..., :100][None] for p in paths[:3]]; g = rd(paths[3]).float()[None]; g = g*((g**2).sum(1, keepdim=True).sqrt() > 1)  # windows i-1, i, i+1; label as in dopflow_cache.py
    b, f = band(); bt = torch.from_numpy(b).to(dev); m = DopFlow((-1, 0, 1), True, "blob", 1, False, True, gru=128, c=8, freqs=f[b], vsrc="both", dropout=.2, tau=.2).to(dev)
    m.load_state_dict(torch.load(WEIGHTS, map_location=dev, weights_only=True)); m.eval()
    with torch.no_grad():
        t = time.perf_counter(); d = torch.stack([spectra(c)[..., bt] for c in csi], 1); i = torch.stack([mask_input(c) for c in csi], 1); o = m(d, i); ms = (time.perf_counter()-t)*1e3
    p = o["final"]; e = ((p-g)**2).sum(1).sqrt(); mk = g.abs().sum(1) >= .5; tv = (g[:, :, mk[0]]).mean(2)[0]; v = o["v"][0, 0]
    print(f"\nSample: subject {MAN['subject']}, action {MAN['action']} (Fast walk), frame {MAN['frame']}; device {dev}; {n_params(m):,} parameters")
    print(f"Inference: {ms:.1f} ms from raw CSI to flow, output {tuple(p.shape[1:])}")
    print(f"Moving-region EPE {e[mk].mean():.3f} px, static-region EPE {e[~mk].mean():.3f} px, moving pixels {int(mk.sum())}")
    print(f"Velocity: predicted ({v[0]:.2f}, {v[1]:.2f}) px/frame, label mean ({tv[0]:.2f}, {tv[1]:.2f}) px/frame")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(10, 2.9), layout="constrained")
        for a, im, tt in zip(ax, (flow_rgb(g[0].cpu().numpy()), o["mask"][0, 0].cpu().numpy(), flow_rgb(p[0].cpu().numpy())), ("Label flow", "Predicted mask", "Predicted flow")):
            a.imshow(im, cmap="gray", vmin=0, vmax=1); a.set_title(tt); a.set_xticks([]); a.set_yticks([])
        fig.savefig(ROOT/"demo_output.png", dpi=150); print("Saved", ROOT/"demo_output.png")
    except ImportError: print("Install matplotlib to save the figure.")
if __name__ == "__main__": main()
