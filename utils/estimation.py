import nptyping as npt
import numpy as np
from abc import ABC
from scipy import stats

def relu(x):
    return x * (x > 0)


class MonteCarloEstimators(ABC):
    def __init__(self, r: float, sigma: float, S0: float, K: float, n_rep: int, n_strat: int, T: float = 1) -> None:
        self.r = r
        self.sigma = sigma
        self.S0 = S0
        self.K = K
        self.n_rep = n_rep
        self.n_strat = n_strat
        self.T = 1


    def _BM_to_geam(self, BM, n_timestamps):
        tdeltas = ((np.arange(n_timestamps) + 1) / n_timestamps * self.T).reshape(-1, 1)
        mu = self.r - self.sigma ** 2 / 2
        return self.S0 * np.exp(BM * self.sigma + mu * tdeltas)

    def simple_sample(self, size: int, n_timestamps: int = 1) -> npt.NDArray:
        BM = np.random.normal(scale=self.T / n_timestamps, size=(n_timestamps, size)).cumsum(axis=0)
        return self._BM_to_geam(BM, n_timestamps)


    def transformSY(self, St: npt.NDArray):
        return relu(St.mean() - self.K)

    def CrudeMonteCarlo(self, St: npt.NDArray) -> float:
        Y = self.transformSY(St)
        return Y.mean()

    def _sample_from_strata(self):
        pass

    def stratified_sample(self, sizes: list, n_timestamps: int = 1) -> list[npt.NDArray]:
        # TODO: probably whole method could be presented in a more clear way
        assert sizes.shape[0] == self.n_strat  # TODO: fix it somehow
        # specify covariance matrix and run Cholesky decomposition
        I = np.repeat((np.arange(n_timestamps) + 1), n_timestamps).reshape((n_timestamps, n_timestamps))
        cov_mat = np.concatenate([I[:, :, np.newaxis], I.transpose()[:, :, np.newaxis]], axis=2).min(axis=2) / n_timestamps * self.T
        A = np.linalg.cholesky(cov_mat)

        # Sample xi, u
        normals = np.random.normal(size=(n_timestamps, sizes.sum()))
        u = np.random.uniform(size=sizes.sum())

        # transform in strata
        stratas = np.concatenate([np.ones(sizes[i]) * (i + 1) for i in range(sizes.reshape(-1).shape[0])])
        D = np.sqrt(stats.chi2.ppf(self.r, n_timestamps) * ((stratas + u) / self.n_strat))
        BM = A @ (D * normals / np.square(normals).sum(axis=0))
        return self._BM_to_geam(BM, n_timestamps)

    def StratifiedMonteCarlo(self, n: int = 10 ** 6) -> float:
        # TODO; it's just sample mean for properly sampled data
        pass

    def AntitheticSample(self, size: int):
        BM_half = np.random.normal(scale=self.T, size=(1, size // 2))
        BM = np.concatenate([BM_half. BM], axis=1)
        return self._BM_to_geam(BM, 1)


if __name__ == "__main__":
    MC = MonteCarloEstimators(0.05, 0.25, 100, 100, 1000, 10, 1)
    print(MC.simple_sample(25, 2))