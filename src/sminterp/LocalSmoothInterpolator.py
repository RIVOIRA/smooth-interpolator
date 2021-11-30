from typing import Tuple

import numpy as np
from scipy import special


class SamplingNodes:
    def __init__(self, times: np.ndarray, positions: np.ndarray):
        self._times = times
        self._positions = positions

    def finite_velocity_increments(self) -> np.ndarray:
        dt_s, v_s = self.finite_diff_velocities()
        return np.diff(v_s)

    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    def finite_diff_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        dt_s = np.diff(self.times)
        dy_s = np.diff(self.positions)
        v_s = dy_s / dt_s
        return dt_s, v_s


class SamplingDifferentialNodes(SamplingNodes):
    def __init__(
        self,
        times: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ):
        SamplingNodes.__init__(self, times=times, positions=positions)
        self._velocities = velocities
        self._accelerations = accelerations

    @property
    def velocities(self) -> np.ndarray:
        return self._velocities

    @property
    def accelerations(self) -> np.ndarray:
        return self._accelerations

    @staticmethod
    def compute_taylor_derivatives(
        samples: SamplingNodes,
    ) -> Tuple[np.ndarray, np.ndarray]:
        dt_s, v_s = samples.finite_diff_velocities()
        dt_mid = (dt_s[1:] + dt_s[:-1]) / 2.0
        a_s = np.diff(v_s) / dt_mid
        a_s = np.append(a_s, a_s[-1])
        a_s = np.insert(a_s, 0, a_s[0])
        w = dt_s[1:] / (2.0 * dt_mid)
        v_mid = (1.0 - w) * v_s[1:] + w * v_s[:-1]
        v_mid = np.insert(v_mid, 0, v_mid[0] - a_s[0] * dt_s[0])
        v_mid = np.append(v_mid, v_mid[-1] + a_s[-1] * dt_s[-1])
        return v_mid, a_s

    @staticmethod
    def construct(samples: SamplingNodes):
        (
            v_mid,
            a_s,
        ) = SamplingDifferentialNodes.compute_taylor_derivatives(samples)
        differential_nodes = SamplingDifferentialNodes(
            times=samples.times,
            positions=samples.positions,
            velocities=v_mid,
            accelerations=a_s,
        )
        return differential_nodes


class UnitIntervalDifferentialNodes(SamplingDifferentialNodes):
    def __init__(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ):
        unit_times = np.asarray([0.0, 1.0])
        SamplingDifferentialNodes.__init__(
            self, unit_times, positions, velocities, accelerations
        )

    def time_0(self) -> float:
        return self._times[0]

    def time_1(self) -> float:
        return self._times[1]

    def position_0(self) -> float:
        return self._positions[0]

    def position_1(self) -> float:
        return self._positions[1]

    def velocity_0(self) -> float:
        return self._velocities[0]

    def velocity_1(self) -> float:
        return self._velocities[1]

    def acceleration_0(self) -> float:
        return self._accelerations[0]

    def acceleration_1(self) -> float:
        return self._accelerations[1]


class LocalSmoothInterp:
    erf_const = 2.0 / np.sqrt(np.pi)

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        d2y: np.ndarray,
        dy: np.ndarray = None,
        sigma: float = np.sqrt(1 / 5.0),
    ):
        self._sigma = sigma
        alpha = 1.0 / sigma
        self.alpha = alpha
        alpha2 = alpha ** 2
        R = 1 / (1.0 + 2.0 * alpha2)
        self.Rc = R * np.exp(alpha2)
        self.nu1 = R
        mu1 = self.Rc * (special.erfc(alpha) + self.erf_const * alpha * np.exp(-alpha2))
        self.mu1 = mu1
        kappa = 0.5 * (1 - R)
        self.kappa = kappa
        # partial construction to fill the sigma dependant constants
        if x is None or y is None:
            return
        if len(x) < 2:
            raise Exception("Not enough nodes to interpolate")
        if dy is None and d2y is not None:
            dy = self.d2y_to_dy(SamplingNodes(times=x, positions=y), d2y=d2y)
        differential_samples = SamplingDifferentialNodes(
            times=x, positions=y, velocities=dy, accelerations=d2y
        )
        self._differential_samples = differential_samples

    def r_func(self) -> float:
        kappa = self.kappa
        return (1 - kappa) / (1 + kappa)

    @staticmethod
    def phi_c_func(x: np.ndarray) -> np.ndarray:
        phi_c = 1.0 / np.sqrt(1 - x ** 2)
        return phi_c

    @staticmethod
    def phi_func(x: np.ndarray) -> np.ndarray:
        phi = x * LocalSmoothInterp.phi_c_func(x)
        return phi

    def epsilon_dmu_nu_functions(
        self, x: np.ndarray, with_derivatives: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.atleast_1d(x)
        x_capfloor = np.minimum(np.maximum(x, -1.0), 1.0)
        eps_x = x_capfloor
        nu = self.nu1 * eps_x
        dmu = np.zeros_like(x)

        idx_in = np.abs(x_capfloor) < 1.0
        x_in = x_capfloor[idx_in]

        phi_c = LocalSmoothInterp.phi_c_func(x_in)
        phi = x_in * phi_c
        phi_r = self.alpha * phi
        e_r = special.erf(phi_r)
        exp_minus_phi_r2 = np.exp(-(phi_r ** 2))
        q_r = LocalSmoothInterp.erf_const * phi_r * exp_minus_phi_r2

        # Core interp
        eps_x[idx_in] = e_r - self.nu1 * q_r

        phi_cr = self.alpha * phi_c
        phi_cr2 = phi_cr ** 2
        q_c = self.erf_const * phi_cr * np.exp(-phi_cr2)
        dmu[idx_in] = self.Rc * (q_c + special.erfc(phi_cr))
        nu[idx_in] = self.nu1 * (e_r - q_r)
        if not with_derivatives:
            return eps_x, dmu, nu
        deps_x = np.zeros_like(x)
        d2eps_x = np.zeros_like(x)
        phi_cr5 = phi_cr * phi_cr2 ** 2
        sigma2 = self._sigma ** 2
        deps_x[idx_in] = (
            2.0 * self.erf_const * self.nu1 * exp_minus_phi_r2 * phi_cr5 * sigma2
        )
        d2eps_x[idx_in] = (
            -2.0 * x_in * phi_cr2 * (1.0 - 2.5 * sigma2 + phi ** 2) * deps_x[idx_in]
        )
        return eps_x, dmu, nu, deps_x, d2eps_x

    def unit_layer_interp_split(
        self,
        differential_nodes: UnitIntervalDifferentialNodes,
        interpolated_times: np.ndarray,
        with_derivatives: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.array(interpolated_times)
        LL0 = np.zeros_like(t)
        LL1 = np.zeros_like(t)
        LL2 = np.zeros_like(t)
        ind0 = t == 0.0
        ind1 = t == 1.0
        LL0[ind0] = differential_nodes.position_0()
        LL0[ind1] = differential_nodes.position_1()
        ind_in = (0.0 < t) & (t < 1.0)

        x = 2.0 * t[ind_in] - 1

        # Loading basis functions
        if with_derivatives:
            eps_x, dmu, nu, deps_x, d2eps_x = self.epsilon_dmu_nu_functions(
                x, with_derivatives=with_derivatives
            )
        else:
            eps_x, dmu, nu = self.epsilon_dmu_nu_functions(
                x, with_derivatives=with_derivatives
            )

        # Core interp
        mid_position = 0.5 * (
            differential_nodes.position_1() + differential_nodes.position_0()
        )
        mid_diff_position = 0.5 * (
            differential_nodes.position_1() - differential_nodes.position_0()
        )
        LL0[ind_in] = mid_position + mid_diff_position * eps_x

        # Layer one
        mid_velocity = 0.5 * (
            differential_nodes.velocity_1() + differential_nodes.velocity_0()
        )
        mid_diff_velocity = 0.5 * (
            differential_nodes.velocity_1() - differential_nodes.velocity_0()
        )

        g_even = x * eps_x + dmu - 1.0
        g_odd = x - eps_x
        LL1[ind_in] = 0.5 * (mid_velocity * g_odd + mid_diff_velocity * g_even)

        # Layer two
        mid_acceleration = 0.5 * (
            differential_nodes.acceleration_1() + differential_nodes.acceleration_0()
        )
        mid_diff_acceleration = 0.5 * (
            differential_nodes.acceleration_1() - differential_nodes.acceleration_0()
        )
        G_even = x ** 2 - 1.0 - 2.0 * g_even
        G_odd = (x ** 2 + 1 - self.nu1) * eps_x - 2.0 * x * (1 - dmu) + nu

        LL2[ind_in] = 0.125 * (
            mid_acceleration * G_even + mid_diff_acceleration * G_odd
        )
        if not with_derivatives:
            return LL0, LL1, LL2

        dLL0 = np.zeros_like(t)
        dLL1 = np.zeros_like(t)
        dLL2 = np.zeros_like(t)
        d2LL0 = np.zeros_like(t)
        d2LL1 = np.zeros_like(t)
        d2LL2 = np.zeros_like(t)
        dLL1[ind0] = differential_nodes.velocity_0()
        dLL1[ind1] = differential_nodes.velocity_1()
        d2LL2[ind0] = differential_nodes.acceleration_0()
        d2LL2[ind1] = differential_nodes.acceleration_1()
        # Core interp

        dLL0[ind_in] = 2 * mid_diff_position * deps_x
        d2LL0[ind_in] = 4 * mid_diff_position * d2eps_x

        # Layer one
        dLL1[ind_in] = (
            2 * 0.5 * (mid_velocity * (1.0 - deps_x) + mid_diff_velocity * eps_x)
        )
        d2LL1[ind_in] = 4 * 0.5 * (mid_diff_velocity * deps_x - mid_velocity * d2eps_x)

        dG_even = 2.0 * g_odd
        d2G_even = 2.0 * (1.0 - deps_x)

        dG_odd = (1.0 - self.nu1) * deps_x + 2.0 * g_even
        d2G_odd = (1.0 - self.nu1) * d2eps_x + 2.0 * eps_x

        dLL2[ind_in] = (
            2 * 0.125 * (mid_acceleration * dG_even + mid_diff_acceleration * dG_odd)
        )
        d2LL2[ind_in] = (
            4 * 0.125 * (mid_acceleration * d2G_even + mid_diff_acceleration * d2G_odd)
        )
        return LL0, LL1, LL2, dLL0, dLL1, dLL2, d2LL0, d2LL1, d2LL2

    def unit_layer_interp(
        self,
        differential_nodes: UnitIntervalDifferentialNodes,
        interpolated_times: np.ndarray,
        with_derivatives: bool = False,
    ) -> np.ndarray:
        if not with_derivatives:
            LL0, LL1, LL2 = self.unit_layer_interp_split(
                differential_nodes,
                interpolated_times,
                with_derivatives=with_derivatives,
            )
            return LL0 + LL1 + LL2
        else:
            (
                LL0,
                LL1,
                LL2,
                dLL0,
                dLL1,
                dLL2,
                d2LL0,
                d2LL1,
                d2LL2,
            ) = self.unit_layer_interp_split(
                differential_nodes,
                interpolated_times,
                with_derivatives=with_derivatives,
            )
            return LL0 + LL1 + LL2, dLL0 + dLL1 + dLL2, d2LL0 + d2LL1 + d2LL2

    def __call__(
        self,
        interpolated_times: np.ndarray,
        split: bool = False,
        with_derivatives: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        LL0 = np.zeros_like(interpolated_times)
        LL1 = np.zeros_like(interpolated_times)
        LL2 = np.zeros_like(interpolated_times)
        if with_derivatives:
            dLL0 = np.zeros_like(interpolated_times)
            dLL1 = np.zeros_like(interpolated_times)
            dLL2 = np.zeros_like(interpolated_times)
            d2LL0 = np.zeros_like(interpolated_times)
            d2LL1 = np.zeros_like(interpolated_times)
            d2LL2 = np.zeros_like(interpolated_times)
        differential_nodes = self._differential_samples
        indt0 = interpolated_times <= differential_nodes.times[0]
        LL0[indt0] = differential_nodes.positions[0]
        indtN = interpolated_times >= differential_nodes.times[-1]
        LL0[indtN] = differential_nodes.positions[-1]
        if with_derivatives:
            dLL1[indt0] = differential_nodes.velocities[0]
            dLL1[indtN] = differential_nodes.velocities[-1]
            d2LL2[indt0] = differential_nodes.accelerations[0]
            d2LL2[indtN] = differential_nodes.accelerations[-1]

        Ns = len(differential_nodes.times)
        for n in range(Ns - 1):
            dt_n = differential_nodes.times[n + 1] - differential_nodes.times[n]
            ind_n = (differential_nodes.times[n] < interpolated_times) & (
                differential_nodes.times[n + 1] >= interpolated_times
            )
            dt_n2 = dt_n ** 2
            if np.any(ind_n):
                x_n = (interpolated_times[ind_n] - differential_nodes.times[n]) / dt_n
                z_n = np.asarray(
                    [
                        differential_nodes.positions[n],
                        differential_nodes.positions[n + 1],
                    ]
                )
                dz_n = (
                    np.asarray(
                        [
                            differential_nodes.velocities[n],
                            differential_nodes.velocities[n + 1],
                        ]
                    )
                    * dt_n
                )
                d2z_n = (
                    np.asarray(
                        [
                            differential_nodes.accelerations[n],
                            differential_nodes.accelerations[n + 1],
                        ]
                    )
                    * dt_n2
                )
                unit_nodes = UnitIntervalDifferentialNodes(z_n, dz_n, d2z_n)
                if not with_derivatives:
                    LL0_n, LL1_n, LL2_n = self.unit_layer_interp_split(unit_nodes, x_n)
                    LL0[ind_n] = LL0_n
                    LL1[ind_n] = LL1_n
                    LL2[ind_n] = LL2_n
                else:
                    (
                        LL0_n,
                        LL1_n,
                        LL2_n,
                        dLL0_n,
                        dLL1_n,
                        dLL2_n,
                        d2LL0_n,
                        d2LL1_n,
                        d2LL2_n,
                    ) = self.unit_layer_interp_split(
                        unit_nodes,
                        x_n,
                        with_derivatives=with_derivatives,
                    )
                    LL0[ind_n] = LL0_n
                    LL1[ind_n] = LL1_n
                    LL2[ind_n] = LL2_n
                    dLL0[ind_n] = dLL0_n / dt_n
                    dLL1[ind_n] = dLL1_n / dt_n
                    dLL2[ind_n] = dLL2_n / dt_n
                    d2LL0[ind_n] = d2LL0_n / dt_n2
                    d2LL1[ind_n] = d2LL1_n / dt_n2
                    d2LL2[ind_n] = d2LL2_n / dt_n2

        if not with_derivatives:
            if split:
                return LL0, LL1, LL2
            return LL0 + LL1 + LL2
        else:
            if split:
                return LL0, LL1, LL2, dLL0, dLL1, dLL2, d2LL0, d2LL1, d2LL2
            return LL0 + LL1 + LL2, dLL0 + dLL1 + dLL2, d2LL0 + d2LL1 + d2LL2

    def d2y_to_dy(self, nodes: SamplingNodes, d2y: np.ndarray):
        dt_s, v_s = nodes.finite_diff_velocities()
        dy_size = len(v_s) + 1
        dy = np.zeros(dy_size)
        kappa = self.kappa
        for n in range(len(v_s)):
            dy[n] = v_s[n] - 0.25 * dt_s[n] * (
                (1.0 + kappa) * d2y[n] + (1.0 - kappa) * d2y[n + 1]
            )

        dy[-1] = v_s[-1] + 0.25 * dt_s[-1] * (
            (1.0 + kappa) * d2y[-1] + (1.0 - kappa) * d2y[-2]
        )
        return dy


class TaylorLocalSmoothInterp(LocalSmoothInterp):
    def __init__(self, x: np.ndarray, y: np.ndarray, sigma: float = np.sqrt(1 / 5.0)):
        samples = SamplingNodes(times=x, positions=y)
        differential_nodes = SamplingDifferentialNodes.construct(samples)
        LocalSmoothInterp.__init__(
            self,
            x=x,
            y=y,
            d2y=differential_nodes.accelerations,
            dy=differential_nodes.velocities,
            sigma=sigma,
        )
