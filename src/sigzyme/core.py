import torch
from .utils import torch_peaks


class TorchEMD(object):
    def __init__(
        self,
        max_iter=2000,
        theta_1=0.05,
        theta_2=0.50,
        alpha=0.05,
        spline=torch_akima,
    ):
        self.max_iter = max_iter
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.alpha = alpha
        self.spline = spline

    def get_envelope(self, batch, cols, y, yy):
        # remove peaks on the edges of the original signals
        mask = cols % (self.size - 1) > 0
        batch = batch[mask]
        cols = cols[mask]
        # use linear indices to calculate n_peaks
        ravel = (batch * self.coefs).sum(dim=-1)
        n_ext = torch.bincount(ravel, minlength=self.n_signals).reshape(
            self.batch_shape
        )
        # fancy indexing by repeating the last elements of the batches with fewer peaks
        index = torch.zeros(n_ext.shape + (n_ext.max() + 1,), dtype=int)
        index.scatter_(-1, n_ext.view(*self.shape), 1)
        index = index[..., 1:].flip(dims=(-1,)).cumsum(-1).flip(dims=(-1,))
        index = index.view(-1).cumsum(0).view(n_ext.shape + (n_ext.max(),))
        # the original signal is repeated 3 times, determine actual number of peaks
        n_ext.div_(3, rounding_mode="floor")
        # use the fancy indexing to get peaks for each batch
        t_ext = self.tt[cols[index - 1]]
        y_ext = torch.take_along_dim(yy, cols[index - 1], dim=-1)
        env = self.spline(t_ext, y_ext, self.t_interp)
        # replace interpolation of monotonic residues with the original signal
        env = torch.where(torch.isnan(env), y, env)
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
        yy = torch.cat(
            [y[..., 1:].flip(-1), y, y[..., :-1].flip(-1)],
            axis=-1,
        )
        peak_batch, peak_cols_f = torch_peaks(yy, "ffill")
        peak_batch, peak_cols_b = torch_peaks(yy, "bfill")
        peak_cols = (peak_cols_b + peak_cols_f).div(2, rounding_mode="floor")
        n_peaks, upper = self.get_envelope(peak_batch, peak_cols, y, yy)
        dip_batch, dip_cols_f = torch_peaks(-yy, "ffill")
        dip_batch, dip_cols_b = torch_peaks(-yy, "bfill")
        dip_cols = (dip_cols_b + dip_cols_f).div(2, rounding_mode="floor")
        n_dips, lower = self.get_envelope(dip_batch, dip_cols, y, yy)
        # find zero crossings
        n_zero = torch.count_nonzero(torch.diff(torch.signbit(yy)), axis=-1)
        n_zero.div_(3, rounding_mode="floor")
        is_monotonic = (n_peaks < 2) | (n_dips < 2)
        mu = (upper + lower) / 2
        amp = (upper - lower) / 2
        sigma = torch.abs(mu / amp)

        print(
            torch.vstack(
                [
                    n_peaks,
                    n_dips,
                    n_zero,
                    torch.abs(n_peaks + n_dips - n_zero),
                    (sigma > self.theta_1).sum(-1),
                    (sigma > self.theta_2).sum(-1),
                ]
            )
        )

        # stoppig criteria
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
                mu, is_imf, _ = self.sift(mode)
            if (is_monotonic | is_imf).all():
                break
            mode[~is_imf] = mode[~is_imf] - mu[~is_imf]
            # mode[is_monotonic] = 0.0
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
        self.tt = torch.cat(
            [
                2 * self.time[0] - self.time[1:].flip(-1),
                self.time,
                2 * self.time[-1] - self.time[:-1].flip(-1),
            ]
        )
        self.device = X.device

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
        # t_interp must have same batches as t_peaks for the searchsorted step
        self.t_interp = self.time.expand(X.shape)
        if max_modes is None:
            max_modes = torch.inf
        imfs = []
        is_monotonic = torch.zeros(self.batch_shape, dtype=bool)
        residue = X.clone().detach()
        while not is_monotonic.all() and len(imfs) < max_modes:
            mode, is_monotonic = self.iter(residue)
            if is_monotonic.all():
                break
            imfs.append(mode)
            residue = residue - mode
        # Defines useful attributes
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

    def __call__(self, t, x, max_components=None, min_sigma=0):
        N = t.size(0)
        K = N - self.L + 1
        if max_components is None:
            max_components = self.L
        rangeL = torch.arange(self.L, device=x.device)
        ids = torch.arange(K, device=x.device).expand(self.L, K)
        ids = ids + rangeL.view(-1, 1)
        X = x[ids]
        U, S, VT = torch.linalg.svd(X, full_matrices=False)
        self.sigma = S
        d = torch.where(S / S[0] > min_sigma)[0][-1] + 1
        d = min(d, max_components)
        X_elem = torch.full((self.L, N), torch.nan, device=x.device)
        x_ids = torch.repeat_interleave(rangeL, K)
        y_ids = ids.flipud().flatten()
        results = torch.empty((d, N), device=x.device)
        for i in range(d):
            X_elem[x_ids, y_ids] = (S[i] * U[:, i].outer(VT[i, :])).flipud().flatten()
            results[i] = torch.nanmean(X_elem, 0)
        return results
