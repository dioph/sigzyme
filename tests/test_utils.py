import numpy as np
from scipy import interpolate
import torch

from sigzyme.utils import torch_akima, torch_cubic, torch_hermite


def test_akima_scipy():
    t = torch.arange(5).to(float)
    y = torch.tensor([3, 2, 3, 4, 3]).to(float)
    t0 = torch.arange(0, 4.5, 0.5).to(float)
    y0 = torch_akima(t, y, t0)
    true_y0 = interpolate.Akima1DInterpolator(t, y)(t0)
    assert np.allclose(y0, true_y0)


def test_hermite_scipy():
    t = torch.arange(5).to(float)
    y = torch.tensor([3, 2, 3, 4, 3]).to(float)
    t0 = torch.arange(0, 4.5, 0.5).to(float)
    y0 = torch_hermite(t, y, t0)
    true_y0 = interpolate.CubicHermiteSpline(t, y, dydx=np.gradient(y, t))(t0)
    assert np.allclose(y0, true_y0)


def test_cubic_scipy():
    t = torch.arange(5).to(float)
    y = torch.tensor([3, 2, 3, 4, 3]).to(float)
    t0 = torch.arange(0, 4.5, 0.5).to(float)
    y0 = torch_cubic(t, y, t0)
    true_y0 = interpolate.CubicSpline(t, y, bc_type="natural")(t0)
    assert np.allclose(y0, true_y0)
