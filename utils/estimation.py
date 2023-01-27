import nptyping as npt
import numpy as np
from abc import ABC
from scipy import stats
from typing import Optional



class MonteCarloEstimators(ABC):
    def __init__(self, r: float = 0.05, sigma: float = 0.25, S0: float = 100.0, K: float = 100.0, n_rep: int = 2 ** 20,
                 n_timestamps: int = 1, strata_size: Optional[npt.NDArray] = None, T: float = 1.0) -> None:
        self.r = r
        self.sigma = sigma
        self.S0 = S0
        self.K = K
        self.n_rep = n_rep
        self.strata_size = strata_size
        self.T = T
        self.n_timestamps = n_timestamps

    @staticmethod
    def _relu(x: npt.NDArray | float) -> npt.NDArray | float:
        return x * (x > 0)

    def _BM_to_geom(self, BM: npt.NDArray) -> npt.NDArray:
        tdeltas = ((np.arange(self.n_timestamps) + 1) / self.n_timestamps * self.T).reshape(-1, 1)
        mu = self.r - self.sigma ** 2 / 2
        return self.S0 * np.exp(BM * self.sigma + mu * tdeltas)

    def _transform_to_cost(self, St: npt.NDArray) -> npt.NDArray:
        return self._relu(St.mean() - self.K)

    def simple_sample(self) -> npt.NDArray:
        BM = np.random.normal(scale=self.T / self.n_timestamps, size=(self.n_timestamps, self.n_rep)).cumsum(axis=0)
        return BM

    def _CMC_estimator(self, Y: npt.NDArray) -> float:
        return Y.mean()

    def CrudeMonteCarlo(self) -> float:
        BM = self.simple_sample()
        GMB = self._BM_to_geom(BM)
        Y = self._transform_to_cost(GMB)
        Y_hat  = self._CMC_estimator(Y)
        return Y_hat

    def _BM_covariance(self) -> npt.NDArray:
        I = np.repeat((np.arange(self.n_timestamps) + 1), self.n_timestamps).reshape((self.n_timestamps, self.n_timestamps))
        cov_mat = np.concatenate([I[:, :, np.newaxis], I.transpose()[:, :, np.newaxis]], axis=2).min(
            axis=2) / self.n_timestamps * self.T
        return cov_mat

    def _sample_from_strata(self,  index: int, A: npt.NDArray) -> npt.NDArray:
        size = self.strata_size[i]
        # Sample xi, u
        normals = np.random.normal(size=(self.n_timestamps, size))
        u = np.random.uniform(size=size)

        # transform in strata
        D = np.sqrt(stats.chi2.ppf(self.r, self.n_timestamps) * ((index + u) / self.n_strat))
        BM = A @ (D * normals / np.square(normals).sum(axis=0))
        return BM

    def stratified_sample(self) -> list[npt.NDArray]:
        # specify covariance matrix and run Cholesky decomposition
        cov_mat = self._BM_covariance()
        A = np.linalg.cholesky(cov_mat)

        samples = [self._sample_from_strata(i, A) for i in self.strata_size]
        return samples

    def _StartMC_estimator(self, Ys: npt.NDArray) -> float:  # TODO: dla ogólności można dodać p_i
        Y_hat = np.array([Y.mean() for Y in Ys]).mean()
        return Y_hat

    def StratifiedMonteCarlo(self) -> float:  # TODO: add saving variances!
        BMs = self.stratified_sample()
        GBMs = [self._BM_to_geom(BM) for BM in BMs]
        Ys = [self._transform_to_cost(GBM) for GBM in GBMs]
        Y_hat = self._StartMC_estimator(Ys)
        return Y_hat

    def antithetic_sample(self) -> npt.NDArray:
        BM_half = np.random.normal(scale=self.T, size=(1, self.n_rep // 2))
        BM = np.concatenate([BM_half, -1 * BM_half], axis=1)
        return BM

    def AnthiteticMonteCarlo(self) -> float:
        BM = self.antithetic_sample()
        GMB = self._BM_to_geom(BM)
        Y = self._transform_to_cost(GMB)
        Y_hat  = self._CMC_estimator(Y)
        return Y_hat

    def _control_variate(self, X: npt.NDArray, Y: npt.NDArray) -> float:
        const = -1 * np.cov(Y, X) / 1  # TODO: parametrise
        Y_hat = Y + const * (X.mean() - 0)
        return Y_hat

    def ControlVariateMonteCarlo(self) -> float:
        BM = self.simple_sample()
        GMB = self._BM_to_geom(BM)
        Y = self._transform_to_cost(GMB)
        Y_hat = self._control_variate(BM, Y)
        return Y_hat


if __name__ == "__main__":
    MC = MonteCarloEstimators(0.05, 0.25, 100, 100, 1000, 10, 1)  # TODO: fix arguments!
    print(MC.CrudeMonteCarlo())
    print(MC.AnthiteticMonteCarlo())
    print(MC.StratifiedMonteCarlo())
    print(MC.ControlVariateMonteCarlo())