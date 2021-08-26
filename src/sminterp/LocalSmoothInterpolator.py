from typing import Tuple

import numpy as np
from scipy import special


class LocalSmoothInterp:
    erf_const = 2.0 / np.sqrt(np.pi)

    def __init__(self, sigma: float = np.sqrt(1 / 5.0)):
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

    class SamplingNodes:
        def __init__(self, times: np.ndarray, positions: np.ndarray):
            self.times = times
            self.positions = positions

        def finite_diff_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
            dt_s = np.diff(self.times)
            dy_s = np.diff(self.positions)
            v_s = dy_s / dt_s
            return dt_s, v_s

        def finite_velocity_increments(self) -> np.ndarray:
            dt_s, v_s = self.finite_diff_velocities()
            return np.diff(v_s)

    class SamplingDifferentialNodes(SamplingNodes):
        def __init__(
            self,
            times: np.ndarray,
            positions: np.ndarray,
            velocities: np.ndarray,
            accelerations: np.ndarray,
        ):
            LocalSmoothInterp.SamplingNodes.__init__(
                self, times=times, positions=positions
            )
            self.velocities = velocities
            self.accelerations = accelerations

        @staticmethod
        def compute_taylor_derivatives(
            samples: "SamplingNodes",
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
        def construct(samples: "SamplingNodes"):
            (
                v_mid,
                a_s,
            ) = LocalSmoothInterp.SamplingDifferentialNodes.compute_taylor_derivatives(
                samples
            )
            differential_nodes = LocalSmoothInterp.SamplingDifferentialNodes(
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
            LocalSmoothInterp.SamplingDifferentialNodes.__init__(
                self, unit_times, positions, velocities, accelerations
            )

        def time_0(self) -> float:
            return self.times[0]

        def time_1(self) -> float:
            return self.times[1]

        def position_0(self) -> float:
            return self.positions[0]

        def position_1(self) -> float:
            return self.positions[1]

        def velocity_0(self) -> float:
            return self.velocities[0]

        def velocity_1(self) -> float:
            return self.velocities[1]

        def acceleration_0(self) -> float:
            return self.accelerations[0]

        def acceleration_1(self) -> float:
            return self.accelerations[1]

    def epsilon_dmu_nu_functions(
        self, x: np.ndarray
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
        q_r = LocalSmoothInterp.erf_const * phi_r * np.exp(-(phi_r ** 2))

        # Core interp
        eps_x[idx_in] = e_r - self.nu1 * q_r

        phi_cr = self.alpha * phi_c
        q_c = self.erf_const * phi_cr * np.exp(-(phi_cr ** 2))
        dmu[idx_in] = self.Rc * (q_c + special.erfc(phi_cr))
        nu[idx_in] = self.nu1 * (e_r - q_r)
        return eps_x, dmu, nu

    def unit_layer_interp_split(
        self,
        differential_node: UnitIntervalDifferentialNodes,
        interpolated_times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.array(interpolated_times)
        LL0 = np.zeros_like(t)
        LL1 = np.zeros_like(t)
        LL2 = np.zeros_like(t)
        ind0 = t == 0.0
        ind1 = t == 1.0
        LL0[ind0] = differential_node.position_0()
        LL0[ind1] = differential_node.position_1()
        ind_in = (0.0 < t) & (t < 1.0)

        x = 2.0 * t[ind_in] - 1

        # Loading building basis functions

        eps_x, dmu, nu = self.epsilon_dmu_nu_functions(x)

        # Core interp
        mid_position = 0.5 * (
            differential_node.position_1() + differential_node.position_0()
        )
        mid_diff_position = 0.5 * (
            differential_node.position_1() - differential_node.position_0()
        )
        LL0[ind_in] = mid_position + mid_diff_position * eps_x

        # Layer one
        mid_velocity = 0.5 * (
            differential_node.velocity_1() + differential_node.velocity_0()
        )
        mid_diff_velocity = 0.5 * (
            differential_node.velocity_1() - differential_node.velocity_0()
        )

        LL1[ind_in] = 0.5 * (
            mid_velocity * (x - eps_x) + mid_diff_velocity * (x * eps_x + dmu - 1.0)
        )

        # Layer two
        mid_acceleration = 0.5 * (
            differential_node.acceleration_1() + differential_node.acceleration_0()
        )
        mid_diff_acceleration = 0.5 * (
            differential_node.acceleration_1() - differential_node.acceleration_0()
        )
        G_even = x ** 2 + 1.0 - 2.0 * x * eps_x - 2.0 * dmu
        G_odd = (x ** 2 + 1 - self.nu1) * eps_x - 2.0 * x * (1 - dmu) + nu

        LL2[ind_in] = 0.125 * (
            mid_acceleration * G_even + mid_diff_acceleration * G_odd
        )

        return LL0, LL1, LL2

    def unit_layer_interp(
        self,
        differential_node: UnitIntervalDifferentialNodes,
        interpolated_times: np.ndarray,
    ) -> np.ndarray:
        LL0, LL1, LL2 = self.unit_layer_interp_split(
            differential_node, interpolated_times
        )
        return LL0 + LL1 + LL2

    def layered_interp_split(
        self,
        differential_node: SamplingDifferentialNodes,
        interpolated_times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        LL0 = np.zeros_like(interpolated_times)
        LL1 = np.zeros_like(interpolated_times)
        LL2 = np.zeros_like(interpolated_times)

        LL0[
            interpolated_times <= differential_node.times[0]
        ] = differential_node.positions[0]
        LL0[
            interpolated_times >= differential_node.times[-1]
        ] = differential_node.positions[-1]
        Ns = len(differential_node.times)
        for n in range(Ns - 1):
            dt_n = differential_node.times[n + 1] - differential_node.times[n]
            ind_n = (differential_node.times[n] < interpolated_times) & (
                differential_node.times[n + 1] >= interpolated_times
            )
            if np.any(ind_n):
                x_n = (interpolated_times[ind_n] - differential_node.times[n]) / dt_n
                z_n = np.asarray(
                    [differential_node.positions[n], differential_node.positions[n + 1]]
                )
                dz_n = (
                    np.asarray(
                        [
                            differential_node.velocities[n],
                            differential_node.velocities[n + 1],
                        ]
                    )
                    * dt_n
                )
                d2z_n = (
                    np.asarray(
                        [
                            differential_node.accelerations[n],
                            differential_node.accelerations[n + 1],
                        ]
                    )
                    * dt_n
                    * dt_n
                )
                unit_nodes = LocalSmoothInterp.UnitIntervalDifferentialNodes(
                    z_n, dz_n, d2z_n
                )
                LL0_n, LL1_n, LL2_n = self.unit_layer_interp_split(unit_nodes, x_n)
                LL0[ind_n] = LL0_n
                LL1[ind_n] = LL1_n
                LL2[ind_n] = LL2_n

        return LL0, LL1, LL2

    def layered_interp(
        self,
        differential_node: SamplingDifferentialNodes,
        interpolated_times: np.ndarray,
    ) -> np.ndarray:
        LL0, LL1, LL2 = self.layered_interp_split(differential_node, interpolated_times)
        return LL0 + LL1 + LL2

    def layered_interp_taylor_split(
        self, samples: SamplingNodes, interpolated_times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        differential_node = LocalSmoothInterp.SamplingDifferentialNodes.construct(
            samples
        )
        return self.layered_interp_split(differential_node, interpolated_times)

    def layered_interp_taylor(
        self, samples: SamplingNodes, interpolated_times: np.ndarray
    ) -> np.ndarray:
        differential_node = LocalSmoothInterp.SamplingDifferentialNodes.construct(
            samples
        )
        return self.layered_interp(differential_node, interpolated_times)

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
