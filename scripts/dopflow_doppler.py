"""Differentiable-free Doppler spectra and cached mask inputs, matching doppler_feats.py. csi: B,16,114,T complex."""
import os, numpy as np, torch
FS = float(os.environ.get("DOPFLOW_HZ", 1000))
KK = int(os.environ.get("DOPFLOW_K", 100))
def freqs(K=KK, fs=FS): return np.fft.fftshift(np.fft.fftfreq(K, 1/fs))
def band(K=KK, fs=FS, lo=10, hi=150):
    f = freqs(K, fs); return (np.abs(f) >= lo) & (np.abs(f) <= hi), f
def spectra(csi):
    """returns B,4,2,T power spectra; estimator 0 = quotient, 1 = conjugate multiplication."""
    B, A, C, T = csi.shape; w = torch.hann_window(T, periodic=False, device=csi.device); out = []
    for r in range(4):
        a = csi[:, r*4:r*4+3]; ref = csi[:, r*4+3:r*4+4]
        q = a/(ref+1e-9+1e-9j); m = q.abs().median(dim=-1, keepdim=True).values; q = q*(q.abs() < 5*m+1e-6).float()
        q = q-q.mean(-1, keepdim=True); Pq = torch.fft.fftshift(torch.fft.fft(q*w, dim=-1), dim=-1).abs()**2
        c = a*ref.conj(); c = c/(c.abs().mean(-1, keepdim=True)+1e-9); c = c-c.mean(-1, keepdim=True)
        Pc = torch.fft.fftshift(torch.fft.fft(c*w, dim=-1), dim=-1).abs()**2
        out.append(torch.stack([Pq.mean((1, 2)), Pc.mean((1, 2))], 1))
    return torch.stack(out, 1)
def linear_features(P, f):
    """P: ...,4,2,F band power; f: band frequencies. Order matches feat(): Q centroid, Q asym, C centroid, C asym (16 dims)."""
    f = torch.as_tensor(f, dtype=P.dtype, device=P.device); s = P.sum(-1)+1e-12
    cent = (P*f).sum(-1)/s; asym = (P[..., f > 0].sum(-1)-P[..., f < 0].sum(-1))/s
    return torch.nan_to_num(torch.cat([cent[..., 0], asym[..., 0], cent[..., 1], asym[..., 1]], -1))
def mask_input(csi, sub=4, dec=None):
    """Robust quotient averaged over subcarrier groups and time bins: B,12,2,28,20 real/imag. Keeps the full 100 ms aperture."""
    B, A, C, T = csi.shape; qs = []; dec = dec or max(1, T//20)
    for r in range(4):
        q = csi[:, r*4:r*4+3]/(csi[:, r*4+3:r*4+4]+1e-9+1e-9j); m = q.abs().median(dim=-1, keepdim=True).values; qs.append(q*(q.abs() < 5*m+1e-6).float())
    q = torch.cat(qs, 1); C2 = C//sub; T2 = T//dec
    q = q[:, :, :C2*sub, :T2*dec].reshape(B, 12, C2, sub, T2, dec).mean((3, 5))
    return torch.stack([q.real, q.imag], 2)
