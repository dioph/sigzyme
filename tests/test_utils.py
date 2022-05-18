import numpy as np
import pytest
from scipy import interpolate
import torch

from sigzyme.utils import (
    torch_akima,
    torch_natural,
    torch_not_a_knot,
    torch_hermite,
    torch_peaks,
)


@pytest.fixture
def t():
    return torch.arange(5).to(float)


@pytest.fixture
def y():
    return torch.tensor([3, 2, 3, 4, 3]).to(float)


@pytest.fixture
def x_rep():
    x1 = torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    x2 = torch.tensor([-2, -1, 1, 2, 3, 3, 3, 7, 8])
    x3 = torch.tensor([-2, -2, 2, 2, 2, 2, 2, 8, 8])
    return torch.stack([x1, x2, x3]).to(float)


@pytest.fixture
def x_norep():
    x1 = torch.tensor([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    x2 = torch.tensor([-2, -1, 1, 2, 3, 7, 8])
    x3 = torch.tensor([-2, 2, 8])
    return [x1, x2, x3]


@pytest.fixture
def y_rep():
    y1 = torch.tensor([2, 3, 3, 2, 3, 4, 3, 3, 4])
    y2 = torch.tensor([3, 2, 2, 3, 4, 4, 4, 4, 3])
    y3 = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
    return torch.stack([y1, y2, y3]).to(float)


@pytest.fixture
def y_norep():
    y1 = torch.tensor([2, 3, 3, 2, 3, 4, 3, 3, 4])
    y2 = torch.tensor([3, 2, 2, 3, 4, 4, 3])
    y3 = torch.tensor([1, 1, 1])
    return [y1, y2, y3]


def test_akima_scipy(t, y):
    t0 = torch.arange(0, 4.5, 0.5).to(float)
    y0 = torch_akima(t, y, t0)
    true_y0 = interpolate.Akima1DInterpolator(t, y)(t0)
    assert np.allclose(y0, true_y0)


def test_hermite_scipy(t, y):
    t0 = torch.arange(0, 4.5, 0.5).to(float)
    y0 = torch_hermite(t, y, t0)
    true_y0 = interpolate.CubicHermiteSpline(t, y, dydx=np.gradient(y, t))(t0)
    assert np.allclose(y0, true_y0)


def test_natural_scipy(t, y):
    t0 = torch.arange(0, 4.5, 0.5).to(float)
    y0 = torch_natural(t, y, t0)
    true_y0 = interpolate.CubicSpline(t, y, bc_type="natural")(t0)
    assert np.allclose(y0, true_y0)


def test_nak_scipy(t, y):
    t0 = torch.arange(0, 4.5, 0.5).to(float)
    y0 = torch_not_a_knot(t, y, t0)
    true_y0 = interpolate.CubicSpline(t, y, bc_type="not-a-knot")(t0)
    assert np.allclose(y0, true_y0)


def test_akima_with_repetition(x_rep, y_rep, x_norep, y_norep):
    t0 = torch.arange(0, 5.5, 0.5).expand(3, -1).contiguous()
    y0 = torch_akima(x_rep, y_rep, t0)
    for i in range(3):
        true_y0 = interpolate.Akima1DInterpolator(x_norep[i], y_norep[i])(t0[i])
        assert np.allclose(y0[i], true_y0)


def test_hermite_with_repetition(x_rep, y_rep, x_norep, y_norep):
    t0 = torch.arange(0, 5.5, 0.5).expand(3, -1).contiguous()
    y0 = torch_hermite(x_rep, y_rep, t0)
    for i in range(3):
        true_y0 = interpolate.CubicHermiteSpline(
            x_norep[i], y_norep[i], dydx=np.gradient(y_norep[i], x_norep[i])
        )(t0[i])
        assert np.allclose(y0[i], true_y0)


def test_natural_with_repetition(x_rep, y_rep, x_norep, y_norep):
    t0 = torch.arange(0, 5.5, 0.5).expand(3, -1).contiguous()
    y0 = torch_natural(x_rep, y_rep, t0)
    for i in range(3):
        true_y0 = interpolate.CubicSpline(x_norep[i], y_norep[i], bc_type="natural")(
            t0[i]
        )
        assert np.allclose(y0[i], true_y0)


def test_nak_with_repetition(x_rep, y_rep, x_norep, y_norep):
    t0 = torch.arange(0, 5.5, 0.5).expand(3, -1).contiguous()
    y0 = torch_not_a_knot(x_rep, y_rep, t0)
    for i in range(3):
        true_y0 = interpolate.CubicSpline(x_norep[i], y_norep[i], bc_type="not-a-knot")(
            t0[i]
        )
        assert np.allclose(y0[i], true_y0)


def test_peaks_constant():
    batch, cols = torch_peaks(torch.ones(10), method="ffill")
    assert len(batch) == 0
    assert len(cols) == 0


def test_peaks_plateaus():
    plateau_sizes = torch.tensor([1, 2, 3, 4, 8, 20, 111])
    x = torch.zeros(len(plateau_sizes) * 2 + 1)
    x[1::2] = plateau_sizes
    repeats = torch.ones(x.shape, dtype=int)
    repeats[1::2] = x[1::2]
    x = torch.repeat_interleave(x, repeats)
    batch, cols = torch_peaks(x, method="ffill")
    assert torch.allclose(cols, torch.tensor([1, 3, 6, 10, 15, 24, 45]))
    batch, cols = torch_peaks(x, method="bfill")
    assert torch.allclose(cols, torch.tensor([1, 4, 8, 13, 22, 43, 155]))
