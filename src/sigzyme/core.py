import warnings

import torch

from .utils import torch_not_a_knot, torch_peaks


class TorchEMD(object):
    def __init__(
        self,
        max_iter=2000,
        pad_width=2,
        theta_1=0.05,
        theta_2=0.50,
        alpha=0.05,
        spline=torch_not_a_knot,
    ):
        self.max_iter = max_iter
        self.pad_width = pad_width
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.alpha = alpha
        self.spline = spline

    def get_envelope(self, batch, cols, y):
        # use linear indices to calculate n_ext
        ravel = (batch * self.coefs).sum(dim=-1)
        n_ext = torch.bincount(ravel, minlength=self.n_signals).reshape(
            self.batch_shape
        )
        # fancy indexing by repeating the last elements of the batches with fewer extrema
        index = torch.zeros(
            n_ext.shape + (n_ext.max() + 1,), dtype=int, device=self.device
        )
        index.scatter_(-1, n_ext.view(*self.shape), 1)
        index = index[..., 1:].flip(dims=(-1,)).cumsum(-1).flip(dims=(-1,))
        index = index.view(-1).cumsum(0).view(n_ext.shape + (n_ext.max(),))
        # use the fancy indexing to get extrema for each batch
        t_ext = self.time[cols[index - 1]]
        y_ext = torch.take_along_dim(y, cols[index - 1], dim=-1)
        # catch monotonicity
        if t_ext.shape[-1] < 1:
            return n_ext, y
        # pad extrema before interpolating to counter edge effects
        t_left = torch.arange(self.pad_width, device=self.device).expand(
            *self.batch_shape, -1
        )
        t_left = torch.clamp_max(t_left, t_ext.shape[-1] - 1)
        t_left = torch.take_along_dim(t_ext, t_left, dim=-1).fliplr()
        t_left = 2 * self.time[0] - t_left
        t_right = n_ext.view(*self.shape) - torch.arange(
            1, self.pad_width + 1, device=self.device
        )
        t_right = torch.clamp_min(t_right, 0)
        t_right = torch.take_along_dim(t_ext, t_right, dim=-1)
        t_right = 2 * self.time[-1] - t_right
        t_ext = torch.cat([t_left, t_ext, t_right], axis=-1)
        y_left = torch.arange(self.pad_width, device=self.device).expand(
            *self.batch_shape, -1
        )
        y_left = torch.clamp_max(y_left, y_ext.shape[-1] - 1)
        y_left = torch.take_along_dim(y_ext, y_left, dim=-1).fliplr()
        y_right = n_ext.view(*self.shape) - torch.arange(
            1, self.pad_width + 1, device=self.device
        )
        y_right = torch.clamp_min(y_right, 0)
        y_right = torch.take_along_dim(y_ext, y_right, dim=-1)
        y_ext = torch.cat([y_left, y_ext, y_right], axis=-1)
        env = self.spline(t_ext, y_ext, self.t_interp)
        # replace interpolation of monotonic residues with the original signal
        # env = torch.where(torch.isnan(env), y, env)
        return n_ext, env

    def sift(self, y):
        """
        Parameters
        ----------
        y: Tensor (..., size)

        Returns
        -------
        mu: Tensor (..., size)
        is_imf: Tensor (...,)
        is_monotonic: Tensor (...,)
        """
        # upper envelope
        peak_batch, peak_cols_f = torch_peaks(y, "ffill")
        peak_batch, peak_cols_b = torch_peaks(y, "bfill")
        peak_cols = (peak_cols_b + peak_cols_f).div(2, rounding_mode="floor")
        n_peaks, upper = self.get_envelope(peak_batch, peak_cols, y)
        # lower envelope
        dip_batch, dip_cols_f = torch_peaks(-y, "ffill")
        dip_batch, dip_cols_b = torch_peaks(-y, "bfill")
        dip_cols = (dip_cols_b + dip_cols_f).div(2, rounding_mode="floor")
        n_dips, lower = self.get_envelope(dip_batch, dip_cols, y)
        # find zero crossings
        n_zero = torch.count_nonzero(torch.diff(torch.signbit(y)), axis=-1)
        is_monotonic = (n_peaks < 2) | (n_dips < 2)
        mu = (upper + lower) / 2
        amp = (upper - lower) / 2
        sigma = torch.abs(mu / amp)
        # stopping criteria
        is_imf = (sigma > self.theta_1).sum(axis=-1) < (self.alpha * self.size)
        is_imf &= (sigma < self.theta_2).all(axis=-1)
        is_imf &= torch.abs(n_peaks + n_dips - n_zero) <= 1
        return mu, is_imf, is_monotonic

    def iter(self, y):
        """
        Parameters
        ----------
        y: Tensor (..., size)

        Returns
        -------
        mode: Tensor (..., size)
        is_monotonic: Tensor (...,)
        """
        mode = y.clone().detach()
        for it in range(self.max_iter):
            if it == 0:
                mu, is_imf, is_monotonic = self.sift(mode)
            else:
                mu, is_imf, tmp = self.sift(mode)
                is_monotonic = (is_monotonic | tmp)
            if (is_monotonic | is_imf).all():
                print(it)
                break
            mode[~is_imf] = mode[~is_imf] - mu[~is_imf]
        return mode, is_monotonic

    def fit(self, t, X):
        size = X.shape[-1]
        if t.ndim != 1:
            raise ValueError("'t' should be 1D.")
        if size != t.shape[0]:
            raise ValueError(
                f"'t' should have the same size as the last dim of 'X' "
                f"(got {t.shape[0]} and {size})."
            )
        self.time = t
        self.size = size
        self.device = X.device
        return self

    def transform(self, X, max_modes=None):
        if X.shape[-1] != self.size:
            raise ValueError(
                f"the last dim of 'X' should have the same size as 'time' "
                f"(got {X.shape[-1]} and {self.size})."
            )
        *self.batch_shape, self.size = X.shape
        self.shape = torch.tensor(self.batch_shape + [1], dtype=int, device=self.device)
        self.n_signals = self.shape.prod()
        self.coefs = self.shape[1:].flipud().cumprod(dim=0).flipud()
        # t_interp must have same batches as t_ext for the searchsorted step
        self.t_interp = self.time.expand(X.shape).contiguous()
        if max_modes is None:
            max_modes = torch.inf
        imfs = []
        is_monotonic = torch.zeros(self.batch_shape, dtype=bool)
        residue = X.clone().detach()
        while not is_monotonic.all() and len(imfs) < max_modes:
            print("k =", len(imfs))
            mode, is_monotonic = self.iter(residue)
            if is_monotonic.all():
                break
            mode[is_monotonic] = 0.0
            if len(imfs) > torch.log2(torch.tensor(self.size)):
                warnings.warn(
                    "Stopping criteria n_modes > log2(n_samples) has been reached before"
                    " convergence. If you're using float32, you may want to use float64."
                )
                break
            imfs.append(mode)
            residue = residue - mode
        # define useful attributes
        self.modes = imfs
        self.residue = residue
        self.n_modes = len(imfs)
        return self.modes

    def __call__(self, t, X, max_modes=None):
        self.fit(t, X)
        return self.transform(X, max_modes)


class TorchCEEMDAN(object):
    def __init__(
        self,
        epsilon=0.2,
        ensemble_size=50,
        min_energy=0.0,
        random_seed=None,
        **emd_kwargs,
    ):
        self.epsilon = epsilon
        self.ensemble_size = ensemble_size
        self.min_energy = min_energy
        if random_seed is not None:
            torch.manual_seed(random_seed)
        self.emd = TorchEMD(**emd_kwargs)

    def _realization(self, noise_modes, k, residue):
        noisy_residue = residue.copy()
        if len(noise_modes) > k:
            beta = self.epsilon * np.std(residue)
            if k == 0:
                beta /= np.std(noise_modes[k])
            noisy_residue = noisy_residue + beta * noise_modes[k]
        try:
            mode = self.emd(noisy_residue, max_modes=1)[0]
        except IndexError:
            # in case noisy_residue happens to be monotonic even though residue was not
            mode = noisy_residue.copy()
        return noisy_residue - mode

    def __call__(self, t, X, max_modes=None):
        size = X.shape[-1]
        if t.ndim != 1:
            raise ValueError("'t' should be 1D.")
        if size != t.shape[0]:
            raise ValueError(
                f"'t' should have the same size as the last dim of 'X' "
                f"(got {t.shape[0]} and {size})."
            )
        self.time = t
        self.size = size
        self.device = X.device
        if max_modes is None:
            max_modes = torch.inf
        sigma_x = X.std(-1)
        white_noise = torch.randn((self.ensemble_size, self.size), dtype=X.dtype)
        white_noise_modes = self.emd(white_noise)

        imfs = []
        residue = X / sigma_x
        while len(imfs) < max_modes:
            k = len(imfs)

            # Averages the ensemble of trials for the next mode
            mu = 0
            tasks = [(noise_modes, k, residue) for noise_modes in white_noise_modes]
            mus = pool.map(self._realization, tasks)
            mu = sum(mus) / self.ensemble_size
            imfs.append(residue - mu)
            residue = mu.copy()

            # Checks stopping criteria (if the residue is an IMF or too small)
            if np.var(residue) < self.min_energy:
                break
            residue_imfs = self.emd(residue)
            if len(residue_imfs) <= 1:
                if len(imfs) < max_modes and len(residue_imfs) == 1:
                    imfs.append(residue)
                break

        # Undoes the initial normalization
        for i in range(len(imfs)):
            imfs[i] *= sigma_x
        self.signal = signal
        self.modes = imfs
        self.residue = signal - sum(imfs)
        self.n_modes = len(imfs)
        return self.modes


class TorchSSA(object):
    def __init__(self, L):
        self.L = L

    def fit(self, t):
        self.N = t.size(0)
        self.K = N - self.L + 1
        rangeL = torch.arange(self.L, device=x.device)
        self.ids = torch.arange(self.K, device=x.device).expand(self.L, self.K)
        self.ids = self.ids + rangeL.view(-1, 1)
        self.x_ids = torch.repeat_interleave(rangeL, self.K)
        self.y_ids = self.ids.flipud().flatten()
        return self

    def transform(x, max_components=None, min_sigma=0, save_memory=True):
        if max_components is None:
            max_components = self.L
        X = x[self.ids]
        U, S, VT = torch.linalg.svd(X, full_matrices=False)
        self.sigma = S
        d = torch.where(S / S[0] > min_sigma)[0][-1] + 1
        d = min(d, max_components)
        X_elem = torch.full((self.L, N), torch.nan, device=x.device)
        if save_memory:
            results = torch.empty((d, N), device=x.device)
            for i in range(d):
                X_elem[x_ids, y_ids] = (
                    (S[i] * U[:, i].outer(VT[i, :])).flipud().flatten()
                )
                results[i] = torch.nanmean(X_elem, 0)
        else:
            pass
        return results

    def __call__(self, t, x, max_components=None, min_sigma=0):
        self.fit(t)
        return self.transform(x, max_components=None, min_sigma=0)
