"""Raw CSI to flow with an untrained DopFlow, on random input: shapes, parameter count and latency."""
import sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dopflow_doppler import spectra, mask_input, band
from dopflow_model import DopFlow, n_params
b, f = band(); bt = torch.from_numpy(b)
m = DopFlow((-1, 0, 1), True, "blob", 1, False, True, gru=128, c=8, freqs=f[b], vsrc="both", dropout=.2, tau=.2).eval()
csi = [torch.randn(1, 16, 114, 100, dtype=torch.complex64) for _ in range(3)]  # windows i-1, i, i+1
with torch.no_grad():
    t = time.perf_counter(); d = torch.stack([spectra(c)[..., bt] for c in csi], 1); i = torch.stack([mask_input(c) for c in csi], 1); o = m(d, i); ms = (time.perf_counter()-t)*1e3
print("input CSI", tuple(csi[0].shape), "x 3 windows"); print("flow", tuple(o["final"].shape), "mask", tuple(o["mask"].shape), "velocity", tuple(o["v"].shape)); print("parameters", n_params(m)); print(f"latency {ms:.1f} ms on {torch.get_num_threads()} CPU thread(s)")
assert n_params(m) == 1143317 and o["final"].shape == (1, 2, 128, 168)
