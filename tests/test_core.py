import torch

from sigzyme.core import TorchEMD


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
    t = torch.linspace(0, 10, 1024).to(float)
    t0 = t.to('cuda')
    X = torch.atleast_2d(torch.randn(t.shape)).to(float)
    X0 = X.to('cuda')
    emd = TorchEMD()
    cpu_modes = torch.vstack(emd(t, X))
    gpu_modes = torch.vstack(emd(t0, X0))
    assert torch.allclose(gpu_modes.cpu(), cpu_modes)
    