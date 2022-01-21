import torch

__all__ = ["torch_peaks", "torch_cubic", "torch_hermite", "torch_akima"]


def torch_peaks(y, method):
    y1 = y[..., 1:-1] - y[..., :-2]
    y1 = torch.where(y1 < 0, y1 * 0 - 1, torch.where(y1 > 0, y1 * 0 + 1, y1))
    y2 = y[..., 1:-1] - y[..., 2:]
    y2 = torch.where(y2 < 0, y2 * 0 - 1, torch.where(y2 > 0, y2 * 0 + 1, y2))
    if method == "bfill":
        y2 = torch.where(y2 == 0, -y1, y2)
        first_nonzero = (
            y2.abs() * torch.arange(y2.shape[-1], 0, -1, device=y2.device)
        ).argmax(-1, keepdim=True)
        last_nonzero = (y2.abs() * torch.arange(y2.shape[-1], device=y2.device)).argmax(
            -1, keepdim=True
        )
        y2_without_first_nonzero = y2.scatter(-1, first_nonzero, 0)
        y2_without_last_nonzero = y2.scatter(-1, last_nonzero, 0)
        ind = torch.where(y2_without_last_nonzero != 0)
        indp1 = torch.where(y2_without_first_nonzero != 0)
        y1[indp1] = -y2[ind]
    elif method == "ffill":
        y1 = torch.where(y1 == 0, -y2, y1)
        first_nonzero = (
            y1.abs() * torch.arange(y1.shape[-1], 0, -1, device=y1.device)
        ).argmax(-1, keepdim=True)
        last_nonzero = (y1.abs() * torch.arange(y1.shape[-1], device=y1.device)).argmax(
            -1, keepdim=True
        )
        y1_without_first_nonzero = y1.scatter(-1, first_nonzero, 0)
        y1_without_last_nonzero = y1.scatter(-1, last_nonzero, 0)
        ind = torch.where(y1_without_last_nonzero != 0)
        indp1 = torch.where(y1_without_first_nonzero != 0)
        y2[ind] = -y1[indp1]
    else:
        raise ValueError("method must be either 'bfill' or 'ffill'!")
    out1 = torch.where(y1 > 0, y1 * 0 + 1, y1 * 0)
    out2 = torch.where(y2 > 0, out1, y2 * 0)
    peaks = torch.nonzero(out2)
    batch, cols = peaks[..., :-1], peaks[..., -1]
    cols.add_(1)
    return batch, cols


def torch_cubic(x, y, xs):
    nf = y.size(-1)
    if nf == 0:
        return torch.zeros_like(xs, dtype=y.dtype, device=y.device)
    if nf == 1:
        return torch.ones_like(xs, dtype=y.dtype, device=y.device) * y[..., [0]]
    if nf == 2:
        a = y[..., :1]
        b = (y[..., 1:] - y[..., :1]) / (x[..., 1:] - x[..., :1])
        c = torch.zeros(*y.shape[:-1], 1, dtype=y.dtype, device=y.device)
        d = torch.zeros(*y.shape[:-1], 1, dtype=y.dtype, device=y.device)
    else:
        dx = x[1:] - x[:-1]
        inv_dx = dx.reciprocal()
        inv_dx2 = inv_dx ** 2
        dy = y[..., 1:] - y[..., :-1]
        dy_scaled = 3 * dy * inv_dx2
        D = torch.empty(nf, dtype=y.dtype, device=y.device)
        D[:-1] = inv_dx
        D[-1] = 0
        D[1:] += inv_dx
        D *= 2
        f = torch.empty_like(y)
        f[..., :-1] = dy_scaled
        f[..., -1] = 0
        f[..., 1:] += dy_scaled
        U = inv_dx.clone().detach()
        L = inv_dx.clone().detach()
        for i in range(1, nf):
            w = L[i - 1] / D[i - 1]
            D[i] = D[i] - w * U[i - 1]
            f[i] = f[i] - w * f[i - 1]
        out = f / D
        for i in range(nf - 2, -1, -1):
            out[i] = (f[i] - U[i] * out[i + 1]) / D[i]
        a = y[..., :-1]
        b = out[..., :-1]
        c = (3 * dy * inv_dx - 2 * out[..., :-1] - out[..., 1:]) * inv_dx
        d = (-2 * dy * inv_dx + out[..., :-1] + out[..., 1:]) * inv_dx2
    maxlen = b.size(-1) - 1
    index = torch.bucketize(xs.detach(), x) - 1
    index = index.clamp(0, maxlen)
    t = xs - x[index]
    inner = c[..., index] + d[..., index] * t
    inner = b[..., index] + inner * t
    return a[..., index] + inner * t


def torch_hermite(x, y, xs):
    nf = y.size(-1)
    if nf == 0:
        return torch.zeros_like(xs, dtype=y.dtype, device=y.device)
    if nf == 1:
        return torch.ones_like(xs, dtype=y.dtype, device=y.device) * y[..., [0]]
    else:
        delta = x[..., 1:] - x[..., :-1]
        last_nonzero = (-delta).signbit().cumsum(-1)[..., [-1]].clamp(min=1)
        m = (y[..., 1:] - y[..., :-1]) / delta
        replacement = torch.take_along_dim(m, last_nonzero - 1, dim=-1)
        m = torch.where(torch.isnan(m), replacement, m)
        m = torch.cat(
            [m[..., [0]], (m[..., 1:] + m[..., :-1]) / 2, m[..., [-1]]], axis=-1
        )
        idxs = torch.searchsorted(x, xs) - 1
        idxs.clamp_(min=0, max=nf - 2)
        xi = torch.take_along_dim(x, idxs, dim=-1)
        dx = torch.take_along_dim(x, idxs + 1, dim=-1) - xi
        t = (xs - xi) / dx
        tt = t[..., None, :] ** torch.arange(4, device=t.device)[:, None]
        A = torch.tensor(
            [[1, 0, -3, 2], [0, 1, -2, 1], [0, 0, 3, -2], [0, 0, -1, 1]],
            dtype=t.dtype,
            device=t.device,
        )
        hh = A @ tt
        yi = torch.take_along_dim(y, idxs, dim=-1)
        mi = torch.take_along_dim(m, idxs, dim=-1)
        yi1 = torch.take_along_dim(y, idxs + 1, dim=-1)
        mi1 = torch.take_along_dim(m, idxs + 1, dim=-1)
        return (
            hh[..., 0, :] * yi
            + hh[..., 1, :] * mi * dx
            + hh[..., 2, :] * yi1
            + hh[..., 3, :] * mi1 * dx
        )


def torch_akima(x, y, xs):
    nf = y.size(-1)
    if nf == 0:
        return torch.zeros_like(xs, dtype=y.dtype, device=y.device)
    if nf == 1:
        return torch.ones_like(xs, dtype=y.dtype, device=y.device) * y[..., [0]]
    if nf == 3:
        return torch_hermite(x, y, xs)
    delta = x[..., 1:] - x[..., :-1]
    m = (y[..., 1:] - y[..., :-1]) / delta
    if nf == 2:
        return m * (xs - x[..., [0]]) + y[..., [0]]
    else:
        mm = 2.0 * m[..., [0]] - m[..., [1]]
        mmm = 2.0 * mm - m[..., [0]]
        mp = 2.0 * m[..., [-1]] - m[..., [-2]]
        mpp = 2.0 * mp - m[..., [-1]]
        m = torch.cat([mmm, mm, m, mp, mpp], axis=-1)
        m1 = m[..., 1:].clone().detach()
        m2 = m[..., :-1].clone().detach()
        dm = torch.abs(m1 - m2)
        f1 = dm[..., 2:].clone().detach()
        f2 = dm[..., :-2].clone().detach()
        # accounts for repeated points (delta == 0)
        ind1 = torch.where(torch.isnan(m1) & (~torch.isnan(m2)))
        ind2 = torch.where(torch.isnan(m2) & (~torch.isnan(m1)))
        m1[ind1] = m2[ind2[:-1] + (ind2[-1] + 1,)]
        m2[ind2] = m1[ind1[:-1] + (ind1[-1] - 1,)]

        f1[ind1[:-1] + (ind1[-1] - 2,)] = torch.abs(m1[..., 1:] - m1[..., :-1])[
            ind1[:-1] + (ind1[-1] - 1,)
        ]
        f2[ind2] = torch.abs(m2[..., 1:] - m2[..., :-1])[ind2]
        f1[ind1[:-1] + (ind1[-1] - 1,)] = f2[ind2[:-1] + (ind2[-1] + 1,)]
        f2[ind2[:-1] + (ind2[-1] - 1,)] = f1[ind1[:-1] + (ind1[-1] - 3,)]
        # determines piecewise polynomial coefficients b, c, and d
        f12 = f1 + f2
        f12 = torch.where(torch.isnan(f12), torch.zeros_like(f12), f12)
        ind = torch.nonzero(f12 > 1e-9 * f12.max(), as_tuple=True)
        batch_ind, x_ind = ind[:-1], ind[-1]
        b = m[..., 1:-1].clone().detach()
        b[ind] = (
            f1[ind] * m2[batch_ind + (x_ind + 1,)]
            + f2[ind] * m1[batch_ind + (x_ind + 1,)]
        ) / f12[ind]
        c = (3.0 * m[..., 2:-2] - 2.0 * b[..., :-2] - b[..., 1:-1]) / delta
        d = (b[..., :-2] + b[..., 1:-1] - 2.0 * m[..., 2:-2]) / delta ** 2
        idxs = torch.searchsorted(x, xs) - 1
        idxs.clamp_(min=0, max=nf - 2)
        xi = torch.take_along_dim(x, idxs, dim=-1)
        yi = torch.take_along_dim(y, idxs, dim=-1)
        di = torch.take_along_dim(d, idxs, dim=-1)
        ci = torch.take_along_dim(c, idxs, dim=-1)
        bi = torch.take_along_dim(b, idxs, dim=-1)
        t = xs - xi
        return ((t * di + ci) * t + bi) * t + yi
