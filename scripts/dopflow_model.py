"""DopFlow: F(x) = M(x) * V + optional R(x), V = V_lin + dV. See the execution plan, section 8."""
import torch, torch.nn as nn, torch.nn.functional as F
def bl(i, o, st=1): return nn.Sequential(nn.Conv2d(i, o, 3, st, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
def mask_channels(x):
    """x: B,12,2,28,20 -> B,36,28,20 amp / sin / cos, amp clipped at per-sample q99."""
    z = torch.complex(x[:, :, 0], x[:, :, 1]); a = z.abs(); q = torch.quantile(a.flatten(1), .99, dim=1).view(-1, 1, 1, 1)+1e-9; p = z.angle()
    return torch.cat([(a/q).clip(0, 1), p.sin(), p.cos()], 1)
class CSIEnc(nn.Module):
    def __init__(s, w=48, d=128):
        super().__init__()
        s.net = nn.Sequential(bl(36, w, 2), bl(w, w), bl(w, 2*w, 2), bl(2*w, 2*w), bl(2*w, 2*w, 2), bl(2*w, 2*w), nn.AdaptiveAvgPool2d((4, 3)), nn.Flatten(), nn.Linear(2*w*12, d), nn.LayerNorm(d), nn.GELU())
    def forward(s, x): return s.net(mask_channels(x))
class DopEnc(nn.Module):
    """log spectra B,4,2,F: conv over frequency shared across links, plus a learned link embedding."""
    def __init__(s, c=24, d=128, e=8):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv1d(1, c, 3, padding=1), nn.GELU(), nn.Conv1d(c, c, 3, padding=1), nn.GELU(), nn.Conv1d(c, c, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(4), nn.Flatten())
        s.link = nn.Parameter(torch.randn(8, e)*.1); s.fc = nn.Sequential(nn.Linear(8*(4*c+e), d), nn.LayerNorm(d), nn.GELU())
    def forward(s, P):
        B, R, E, N = P.shape; x = torch.log(P+1e-6); x = (x-x.mean(-1, keepdim=True))/(x.std(-1, keepdim=True)+1e-6)
        h = s.conv(x.reshape(B*R*E, 1, N)).reshape(B, R*E, -1); return s.fc(torch.cat([h, s.link[None].expand(B, -1, -1)], -1).flatten(1))
class DopFlow(nn.Module):
    def __init__(s, offs=(-1, 0, 1), doppler=True, dec="blob", slots=1, residual=False, lin=True, w=48, d=128, c=8, gru=128, freqs=None, vsrc="dop", dropout=0.0, tau=0.0):
        super().__init__(); s.tau = tau; s.vsrc = vsrc if doppler else "csi"; s.drop = nn.Dropout(dropout); s.offs, s.doppler, s.decn, s.slots, s.residual, s.lin_on = tuple(offs), doppler, dec, slots, residual, lin; W = len(offs)
        s.csi = CSIEnc(w, d); s.fuse = nn.Sequential(nn.Linear(W*d, d), nn.LayerNorm(d), nn.GELU())
        s.to_grid = nn.Linear(d, 16*21*c); s.c = c; out = slots if dec == "blob" else 2
        s.dec = nn.Sequential(bl(c+2, 2*c+16), bl(2*c+16, 2*c+16), nn.Conv2d(2*c+16, out, 3, 1, 1))
        if residual: s.res = nn.Conv2d(2*c+16, 2, 3, 1, 1)
        y, x = torch.meshgrid(torch.linspace(-1, 1, 16), torch.linspace(-1, 1, 21), indexing="ij"); s.register_buffer("grid", torch.stack([x, y])[None])
        s.register_buffer("f", torch.as_tensor(freqs).float() if freqs is not None else torch.zeros(30))
        s.denc = DopEnc(d=d); s.gru = nn.GRU(2*d if s.vsrc == "both" else d, gru, batch_first=True); s.head = nn.Linear(gru, 2*slots); nn.init.zeros_(s.head.weight); nn.init.zeros_(s.head.bias)
        n = 16*W; s.register_buffer("lmu", torch.zeros(n)); s.register_buffer("lsd", torch.ones(n)); s.lin = nn.Linear(n, 2)
    def set_lin(s, mu, sd, W, b):
        s.lmu.copy_(mu); s.lsd.copy_(sd); s.lin.weight.data.copy_(W); s.lin.bias.data.copy_(b)
    def lin_feat(s, P):
        fp = s.f; sm = P.sum(-1)+1e-12; cent = (P*fp).sum(-1)/sm; asym = (P[..., fp > 0].sum(-1)-P[..., fp < 0].sum(-1))/sm
        return torch.nan_to_num(torch.cat([cent[..., 0], asym[..., 0], cent[..., 1], asym[..., 1]], -1)).flatten(1)
    def forward(s, dop, mi):
        B, W = dop.shape[:2]; tok = s.csi(mi.flatten(0, 1)).reshape(B, W, -1); g = s.fuse(tok.flatten(1)); x = s.to_grid(g).reshape(B, s.c, 16, 21)
        h = torch.cat([x, s.grid.expand(B, -1, -1, -1)], 1); feat = s.dec[1](s.dec[0](h)); o = s.dec[2](feat); up = lambda t: F.interpolate(t, scale_factor=8, mode="bilinear")
        if s.decn == "pixel": fl = up(o); return dict(final=fl, flow=fl, mask=None, v=None)
        dseq = s.denc(dop.flatten(0, 1)).reshape(B, W, -1) if s.vsrc != "csi" else None
        seq = {"dop": dseq, "csi": tok, "both": torch.cat([dseq, tok], -1) if dseq is not None else None}[s.vsrc]; seq = s.drop(seq)
        dv = s.head(s.gru(seq)[0][:, -1]).reshape(B, s.slots, 2); V = dv
        if s.doppler and s.lin_on: V = V+s.lin((s.lin_feat(dop)-s.lmu)/s.lsd)[:, None]
        M = torch.sigmoid(up(o)); Mf = M if s.training or not s.tau else M*(M > s.tau).float(); fl = (Mf[:, :, None]*V[..., None, None]).sum(1)
        if s.residual: fl = fl+up(s.res(feat))
        return dict(final=fl, mask=M, v=V)
def n_params(m): return sum(p.numel() for p in m.parameters())
