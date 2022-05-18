import pytest
import torch

from sigzyme.core import TorchEMD, TorchCEEMDAN


def test_emd_two_sinusoids_plus_trend():
    t = torch.arange(0, 10, 0.001)
    y0 = t
    y1 = torch.sin(2 * torch.pi * t / 0.2)
    y2 = 0.75 * torch.sin(2 * torch.pi * t / 0.5 + 1)
    X = torch.atleast_2d(y0 + y1 + y2)
    emd = TorchEMD()
    modes = emd(t, X)
    assert len(modes) == 2
    assert torch.allclose(X, sum(modes) + emd.residue, atol=1e-7)
    nsr_0 = torch.sqrt(torch.sum((emd.residue - y0) ** 2) / torch.sum(y0 ** 2))
    nsr_1 = torch.sqrt(torch.sum((modes[0][0] - y1) ** 2) / torch.sum(y1 ** 2))
    nsr_2 = torch.sqrt(torch.sum((modes[1][0] - y2) ** 2) / torch.sum(y2 ** 2))
    assert nsr_0 < 0.02
    assert nsr_1 < 0.10
    assert nsr_2 < 0.20


def test_emd_max_modes():
    t = torch.arange(0, 10, 0.001)
    y0 = t
    y1 = torch.sin(2 * torch.pi * t / 0.2)
    y2 = 0.75 * torch.sin(2 * torch.pi * t / 0.5 + 1)
    X = torch.atleast_2d(y0 + y1 + y2)
    emd = TorchEMD()
    modes = emd(t, X, max_modes=1)
    assert len(modes) == 1
    assert torch.allclose(X, sum(modes) + emd.residue, atol=1e-7)
    nsr_02 = torch.sqrt(
        torch.sum((emd.residue - (y0 + y2)) ** 2) / torch.sum((y0 + y2) ** 2)
    )
    assert nsr_02 < 0.02


def test_emd_gpu():
    if not torch.cuda.is_available():
        pytest.skip("skipping CUDA tests")
    torch.manual_seed(42)
    t = torch.linspace(0, 10, 64).to(float)
    t0 = t.to("cuda")
    X = torch.atleast_2d(torch.randn(t.shape)).to(float)
    X0 = X.to("cuda")
    emd = TorchEMD()
    cpu_modes = emd(t, X)
    gpu_modes = emd(t0, X0)
    assert torch.allclose(gpu_modes.cpu(), cpu_modes, atol=1e-6)


def test_ceemdan_two_tones():
    # Test if nothing but the two tones are recovered by CEEMDAN
    t = torch.arange(1000).to(float)
    s2 = torch.sin(2 * torch.pi * 0.065 * t)
    s1 = torch.zeros_like(s2)
    s1[500:750] += torch.sin(2 * torch.pi * 0.255 * torch.arange(250))
    data = s1 + s2
    imfs = TorchCEEMDAN(ensemble_size=50, random_seed=42)(t, data)
    assert len(imfs) == 2
    # Test if the residual noise in the first mode is close to zero
    left_mse = torch.mean(torch.square(imfs[0][0][11:490]))
    right_mse = torch.mean(torch.square(imfs[0][0][761:990]))
    assert left_mse < 1e-4
    assert right_mse < 1e-4
    # Test the closeness between the original tones and the recovered IMFs
    err1 = (imfs[0] - s1)[:, 3:-3]
    err2 = (imfs[1] - s2)[:, 3:-3]
    err = sum(imfs) - data
    rrse_1 = torch.linalg.norm(err1) / torch.linalg.norm(s1[3:-3])
    rrse_2 = torch.linalg.norm(err2) / torch.linalg.norm(s2[3:-3])
    rrse_x = torch.linalg.norm(err) / torch.linalg.norm(data)
    assert rrse_1 < 0.10
    assert rrse_2 < 0.05
    assert rrse_x < 1e-16
