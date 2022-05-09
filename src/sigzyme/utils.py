from numba import cuda, njit, float64
import numpy as np
import torch

__all__ = [
    "torch_peaks",
    "torch_natural",
    "torch_not_a_knot",
    "torch_hermite",
    "torch_akima",
]


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


@njit(
    (
        float64[:, :],
        float64[:, :],
        float64[:, :],
        float64[:, :],
        float64[:, :],
        float64[:, :],
        float64[:, :],
    ),
    cache=False,
)
def thomas(U, D, L, f, b, cb, db):
    """
    U: (M, N - 1)
    D: (M, N)
    L: (M, N - 1)
    f: (M, N)
    b: (M, N)
    cb: (M, N - 1)
    db: (M, N - 1)
    """
    M, N = b.shape
    zero = float64(0.0)
    for j in range(M):
        dD = zero
        df = zero
        for i in range(N):
            if D[j, i] != zero:
                D[j, i] = D[j, i] - dD
                f[j, i] = f[j, i] - df
                if i < N - 1:
                    dD = L[j, i] * U[j, i] / D[j, i]
                    df = L[j, i] * f[j, i] / D[j, i]
        good = 0
        if D[j, -1] != zero:
            b[j, -1] = f[j, -1] / D[j, -1]
            good += 1
        else:
            b[j, -1] = zero
        w = b[j, -1]
        for i in range(N - 2, -1, -1):
            if D[j, i] != zero:
                good += 1
                b[j, i] = (f[j, i] - U[j, i] * w) / D[j, i]
                if good > 1:
                    cb[j, i] = w + b[j, i] + b[j, i]
                    db[j, i] = w + b[j, i]
                else:
                    cb[j, i] = zero
                    db[j, i] = zero
                w = b[j, i]
            else:
                b[j, i] = zero
                cb[j, i] = zero
                db[j, i] = zero


@cuda.jit(
    (
        float64[:, :],
        float64[:, :],
        float64[:, :],
        float64[:, :],
        float64[:, :],
        float64[:, :],
        float64[:, :],
    )
)
def cu_thomas(U, D, L, f, b, cb, db):
    """
    U: (M, N - 1)
    D: (M, N)
    L: (M, N - 1)
    f: (M, N)
    b: (M, N)
    """
    M, N = b.shape
    pos = cuda.grid(1)
    tx = cuda.gridsize(1)
    zero = float64(0.0)
    for j in range(pos, M, tx):
        dD = zero
        df = zero
        for i in range(N):
            if D[j, i] != zero:
                D[j, i] = D[j, i] - dD
                f[j, i] = f[j, i] - df
                if i < N - 1:
                    dD = L[j, i] * U[j, i] / D[j, i]
                    df = L[j, i] * f[j, i] / D[j, i]
        good = 0
        if D[j, -1] != zero:
            b[j, -1] = f[j, -1] / D[j, -1]
            good += 1
        else:
            b[j, -1] = zero
        w = b[j, -1]
        for i in range(N - 2, -1, -1):
            if D[j, i] != zero:
                good += 1
                b[j, i] = (f[j, i] - U[j, i] * w) / D[j, i]
                if good > 1:
                    cb[j, i] = w + b[j, i] + b[j, i]
                    db[j, i] = w + b[j, i]
                else:
                    cb[j, i] = zero
                    db[j, i] = zero
                w = b[j, i]
            else:
                b[j, i] = zero
                cb[j, i] = zero
                db[j, i] = zero


def torch_natural(x, y, xs):
    return _torch_cubic(x, y, xs, bc_type="natural")


def torch_not_a_knot(x, y, xs):
    return _torch_cubic(x, y, xs, bc_type="not-a-knot")


def _torch_cubic(x, y, xs, bc_type="not-a-knot"):
    *batch_shape, nf = torch.atleast_2d(y).shape
    if nf == 0:
        return torch.zeros_like(xs, dtype=y.dtype, device=y.device)
    if nf == 1:
        return torch.ones_like(xs, dtype=y.dtype, device=y.device) * y[..., [0]]
    delta = x[..., 1:] - x[..., :-1]
    dy = y[..., 1:] - y[..., :-1]
    m = dy / delta
    if nf == 2:
        return m * (xs - x[..., [0]]) + y[..., [0]]
    else:
        dxr = delta.reciprocal()
        dyr = 3 * dy * dxr ** 2
        # write the banded matrix
        dxr1 = torch.zeros_like(y)
        dxr1[..., :-1] = 2 * dxr
        dxr2 = torch.zeros_like(y)
        dxr2[..., 1:] = 2 * dxr
        dyr1 = torch.zeros_like(y)
        dyr1[..., :-1] = dyr
        dyr2 = torch.zeros_like(y)
        dyr2[..., 1:] = dyr
        ind1 = torch.where(torch.isfinite(dxr2) & (~torch.isfinite(dxr1)))
        ind2 = torch.where(torch.isfinite(dxr1) & (~torch.isfinite(dxr2)))
        dxr2[ind2] = dxr2[ind1]
        dyr2[ind2] = dyr2[ind1]
        D = dxr1 + dxr2
        f = dyr1 + dyr2
        U = dxr.clone().detach()
        L = dxr.clone().detach()
        # boundary conditions
        if bc_type == "natural":
            pass
        elif bc_type == "not-a-knot":
            # indices of two first and two last valid knots
            id01 = (
                torch.isfinite(U).to(float).abs()
                * torch.arange(U.shape[-1], 0, -1, device=U.device)
            ).argmax(-1, keepdim=True)
            U2 = U.clone().detach()
            U2.scatter_(-1, id01, np.inf)
            id02 = (
                torch.isfinite(U2).to(float).abs()
                * torch.arange(U2.shape[-1], 0, -1, device=U2.device)
            ).argmax(-1, keepdim=True)
            id11 = (
                torch.isfinite(L).to(float).abs()
                * torch.arange(L.shape[-1], device=L.device)
            ).argmax(-1, keepdim=True)
            L2 = L.clone().detach()
            L2.scatter_(-1, id11, np.inf)
            id12 = (
                torch.isfinite(L2).to(float).abs()
                * torch.arange(L2.shape[-1], device=L2.device)
            ).argmax(-1, keepdim=True)
            d01 = 1 / (
                torch.take_along_dim(delta, id01, -1)
                + torch.take_along_dim(delta, id02, -1)
            )
            d10 = 1 / (
                torch.take_along_dim(delta, id11, -1)
                + torch.take_along_dim(delta, id12, -1)
            )
            m01 = torch.take_along_dim(m, id01, -1)
            m02 = torch.take_along_dim(m, id02, -1)
            m11 = torch.take_along_dim(m, id11, -1)
            m12 = torch.take_along_dim(m, id12, -1)
            h01 = torch.take_along_dim(dxr, id01, -1)
            h02 = torch.take_along_dim(dxr, id02, -1)
            h11 = torch.take_along_dim(dxr, id11, -1)
            h12 = torch.take_along_dim(dxr, id12, -1)
            # when nf=3 and bc=not-a-knot, use a quadratic instead
            quad_mask = id02 > id12
            h02[quad_mask] = 0
            h12[quad_mask] = 0
            d01[quad_mask] = 0
            d10[quad_mask] = 0
            # first knot
            D.scatter_(-1, id01, h01)
            f.scatter_(-1, id01, m01 * d01 + 2 * m01 * h01 + m02 * h02 * d01 / h01)
            U.scatter_(-1, id01, h01 + h02)
            # last knot
            D[..., [-1]] = h11
            f[..., [-1]] = m11 * d10 + 2 * m11 * h11 + m12 * h12 * d10 / h11
            L.scatter_(-1, id11, h11 + h12)
        else:
            raise ValueError(f"Unknown boundary condition {bc_type}.")
        D = torch.where(torch.isfinite(D), D, torch.nan_to_num(0 * D))
        f = torch.where(torch.isfinite(f), f, torch.nan_to_num(0 * f))
        U = torch.where(torch.isfinite(U), U, torch.nan_to_num(0 * U))
        L = torch.where(torch.isfinite(L), L, torch.nan_to_num(0 * L))
        M = torch.tensor(batch_shape, device=y.device).prod().item()
        b = torch.zeros((M, nf), device=y.device, dtype=y.dtype)
        cb = torch.zeros((M, nf - 1), device=y.device, dtype=y.dtype)
        db = torch.zeros((M, nf - 1), device=y.device, dtype=y.dtype)
        # solve the tridiagonal system
        if y.device.type == "cpu":
            U0 = U.view(M, nf - 1).to(torch.float64).numpy()
            D0 = D.view(M, nf).to(torch.float64).numpy()
            L0 = L.view(M, nf - 1).to(torch.float64).numpy()
            f0 = f.view(M, nf).to(torch.float64).numpy()
            b0 = b.to(torch.float64).numpy()
            cb0 = cb.to(torch.float64).numpy()
            db0 = db.to(torch.float64).numpy()
            thomas(U0, D0, L0, f0, b0, cb0, db0)
            b = torch.tensor(b0, device=b.device, dtype=b.dtype).view(y.shape)
            cb = torch.tensor(cb0, device=b.device, dtype=b.dtype).view(dy.shape)
            db = torch.tensor(db0, device=b.device, dtype=b.dtype).view(dy.shape)
        else:
            U0 = cuda.as_cuda_array(U.view(M, nf - 1).to(torch.float64))
            D0 = cuda.as_cuda_array(D.view(M, nf).to(torch.float64))
            L0 = cuda.as_cuda_array(L.view(M, nf - 1).to(torch.float64))
            f0 = cuda.as_cuda_array(f.view(M, nf).to(torch.float64))
            b0 = cuda.as_cuda_array(b.to(torch.float64))
            cb0 = cuda.as_cuda_array(cb.to(torch.float64))
            db0 = cuda.as_cuda_array(db.to(torch.float64))
            cu_thomas[128, 32](U0, D0, L0, f0, b0, cb0, db0)
            b = torch.tensor(b0, device=b.device, dtype=b.dtype).view(y.shape)
            cb = torch.tensor(cb0, device=b.device, dtype=b.dtype).view(dy.shape)
            db = torch.tensor(db0, device=b.device, dtype=b.dtype).view(dy.shape)
        # calculate coefficients
        c = (3.0 * m - cb) * dxr
        d = (-2.0 * m + db) * dxr ** 2
        idxs = torch.searchsorted(x, xs) - 1
        idxs.clamp_(min=0, max=nf - 2)
        xi = torch.take_along_dim(x, idxs, dim=-1)
        yi = torch.take_along_dim(y, idxs, dim=-1)
        di = torch.take_along_dim(d, idxs, dim=-1)
        ci = torch.take_along_dim(c, idxs, dim=-1)
        bi = torch.take_along_dim(b, idxs, dim=-1)
        t = xs - xi
        ys = ((t * di + ci) * t + bi) * t + yi
        return ys


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
