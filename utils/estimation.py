import nptyping as npt
import numpy as np
from abc import ABC
from scipy import stats
from typing import Optional



class MonteCarloEstimators(ABC):
    def __init__(self, r: float = 0.05, sigma: float = 0.25, S0: float = 100.0, K: float = 100.0, n_rep: int = 10**4,
                 set_seed: bool = False, n_timestamps: int = 1, n_strat: Optional[int] = 10,
                 strata_size: Optional[npt.NDArray] = None, T: float = 1.0) -> None:
        self.set_seed = set_seed
        self.r = r
        self.sigma = sigma
        self.S0 = S0
        self.K = K
        self.n_rep = n_rep
        self.T = T
        self.n_timestamps = n_timestamps
        self.n_strat = n_strat
        self.strata_size = self._preprare_strata_size(strata_size)

    @staticmethod
    def _relu(x: npt.NDArray | float) -> npt.NDArray | float:
        return x * (x > 0)

    def _preprare_strata_size(self, strata_size: list | npt.NDArray) -> npt.NDArray:
        if strata_size is not None:
            if len(strata_size) != self.n_strat:
                # print(f"Provided strata sizes have length {len(strata_size)}, specified number of strata ({self.n_strat}) ignored")
                self.n_strat = len(strata_size)
            return np.array(strata_size).reshape(-1)
        else:
            # print("Proportional allocation will be used in SSMC as no strata sizes were provided")
            strata_sizes = np.array([self.n_rep // self.n_strat for _ in range(self.n_strat)])
            self.n_strat = strata_sizes.shape[0]
            return strata_sizes

    def _update_strata_size(self, sigma: npt.NDArray) -> None:
        self.strata_size = (sigma / sigma.sum() * self.n_rep).astype(int)
        self.strata_size[self.strata_size == 0] = self.strata_size[self.strata_size != 0].min()

    def _BM_to_geom(self, BM: npt.NDArray) -> npt.NDArray:
        tdeltas = ((np.arange(self.n_timestamps) + 1) / self.n_timestamps * self.T).reshape(-1, 1)
        mu = self.r - self.sigma ** 2 / 2
        return self.S0 * np.exp(BM * self.sigma + mu * tdeltas)

    def _transform_to_cost(self, St: npt.NDArray) -> npt.NDArray:
        return self._relu(St.mean(axis=0) - self.K).reshape(1, -1)

    def simple_sample(self) -> npt.NDArray:
        if self.set_seed:
            np.random.seed(2023)
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
        # print(cov_mat)
        return cov_mat

    def _sample_from_strata(self,  index: int, A: npt.NDArray) -> npt.NDArray:
        size = self.strata_size[index]
        # Sample xi, u
        normals = np.random.normal(size=(self.n_timestamps, size))
        u = np.random.uniform(size=size)

        # transform in strata
        # D = np.sqrt(stats.chi2.ppf((index + u) / self.n_strat, self.n_timestamps))
        D = np.sqrt(stats.chi2.ppf((index + u) / self.n_strat, self.n_timestamps))  # * ((index + u) / self.n_strat))
        BM = A @ (D * normals / np.sqrt(np.square(normals).sum(axis=0))).reshape(normals.shape)
        return BM

    def stratified_sample(self) -> list[npt.NDArray]:
        if self.set_seed:
            np.random.seed(2023)
        # specify covariance matrix and run Cholesky decomposition
        cov_mat = self._BM_covariance()
        A = np.linalg.cholesky(cov_mat)

        samples = [self._sample_from_strata(i, A) for i in range(self.n_strat)]
        return samples

    def _StartMC_estimator(self, Ys: npt.NDArray) -> float:  # TODO: dla ogólności można dodać p_i
        Y_hat = np.array([Y.mean() for Y in Ys]).mean()
        return Y_hat

    def StratifiedMonteCarlo(self, update: bool = True) -> float:  # TODO: add saving variances!
        BMs = self.stratified_sample()
        GBMs = [self._BM_to_geom(BM) for BM in BMs]
        Ys = [self._transform_to_cost(GBM) for GBM in GBMs]
        Y_hat = self._StartMC_estimator(Ys)
        if update:
            self._update_strata_size(np.array([Y.std() for Y in Ys]))
        return Y_hat

    def antithetic_sample(self) -> npt.NDArray:
        if self.set_seed:
            np.random.seed(2023)
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
        const = -1 * np.cov(Y, X)[1, 1] / 1
        Y_hat = Y.mean() + const * (X.mean() - 0)
        return Y_hat

    def ControlVariateMonteCarlo(self) -> float:
        BM = self.simple_sample()
        GMB = self._BM_to_geom(BM)
        Y = self._transform_to_cost(GMB)
        Y_hat = self._control_variate(BM, Y)
        return Y_hat

    def BlackSchols(self):
        d1 = (np.log(self.S0 / self.K) + self.r + self.sigma ** 2 / 2) / self.sigma
        d2 = d1 - self.sigma
        return self.S0 * stats.norm.cdf(d1) - self.K * np.exp(self.r * (-1)) * stats.norm.cdf(d2)


if __name__ == "__main__":
    MC = MonteCarloEstimators(n_rep=1000000, set_seed=False)
    print(MC.CrudeMonteCarlo())
    print(MC.AnthiteticMonteCarlo())
    print(MC.StratifiedMonteCarlo())
    print(MC.StratifiedMonteCarlo(update=False))
    print(MC.ControlVariateMonteCarlo())
    print(MC.BlackSchols())
