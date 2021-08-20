import scipy.optimize as scop
from scipy.sparse import dia_matrix

from src.sminterp.LocalSmoothInterpolator import *


class SmoothConvexInterpolator(LocalSmoothInterp):
    def __init__(self, sigma: float = np.sqrt(1 / 5.0)):
        LocalSmoothInterp.__init__(self, sigma=sigma)

    def f_func(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r_ = self.r_func()
        f = r_ * (x ** 2 + y ** 2) + (1.0 - r_) * x * y
        return f

    def L_func(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.f_func(x, 1.0 - y)

    def L_matrix_func(self, alpha_p: np.ndarray, alpha_m: np.ndarray) -> np.ndarray:
        # L = [L_1^+,     L_2^-_]
        #     [L_2^+,     L_3^-]
        #     ...
        #     [L_{N-2}^+, L_{N-1}^-]
        L_p = self.L_func(alpha_p, alpha_m)
        L_m = self.L_func(alpha_m, alpha_p)
        return np.column_stack((L_p, L_m))

    def K_diagonals_func(
        self, alpha_p: np.ndarray, alpha_m: np.ndarray, sampling_times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha_shape = np.shape(alpha_p)
        L_pm = np.zeros((alpha_shape[0] + 2, 2))
        L_pm[1:-1][:] = self.L_matrix_func(alpha_p, alpha_m)
        r_ = self.r_func()
        dt = np.diff(sampling_times, axis=0)
        dt = np.expand_dims(dt, axis=-1)
        Ldt = L_pm * dt
        K_up = Ldt[1:, 0]
        K_down = Ldt[:-1, 1]
        M = (1.0 + r_) * dt - Ldt
        M[:-1, 0] = M[1:, 0]
        P = M[:-1, :]
        K_diag = np.sum(P, axis=1)
        return K_diag, K_up, K_down

    @staticmethod
    def K_matrix_from_diagonals(
        K_diag: np.ndarray, K_up: np.ndarray, K_down: np.ndarray
    ) -> np.matrix:
        data = np.stack((K_diag, K_up, K_down))
        offsets = [0, -1, 1]
        n_diag = len(K_diag)
        K_matrix = dia_matrix((data, offsets), shape=(n_diag, n_diag)).T
        return K_matrix

    def Rho_vector_func(
        self, alpha_p: np.ndarray, alpha_m: np.ndarray, sampling_times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        K_diag, K_up, K_down = self.K_diagonals_func(alpha_p, alpha_m, sampling_times)
        rho_p = K_up / np.roll(K_diag, -1)
        rho_m = K_down / np.roll(K_diag, 1)
        return rho_p, rho_m, K_diag

    def Omega_vector_func(
        self,
        alpha_p: np.ndarray,
        alpha_m: np.ndarray,
        samples: LocalSmoothInterp.SamplingNodes,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sampling_times = samples.times
        rho_p, rho_m, K_diag = self.Rho_vector_func(alpha_p, alpha_m, sampling_times)
        dv = samples.finite_velocity_increments()
        dv_p = np.zeros_like(rho_p)
        dv_p[0:-1] = dv[1:]
        dv_m = np.zeros_like(rho_m)
        dv_m[1:] = dv[0:-1]
        omega = 1 - (dv_p * rho_p + dv_m * rho_m) / dv
        return omega, rho_p, rho_m, K_diag

    def d2y_approx(
        self,
        alpha_p: np.ndarray,
        alpha_m: np.ndarray,
        samples: LocalSmoothInterp.SamplingNodes,
    ) -> np.ndarray:
        omega, rho_p, rho_m, K_diag = self.Omega_vector_func(alpha_p, alpha_m, samples)
        dv = samples.finite_velocity_increments()
        scaling_factor = 4.0 / (1.0 + self.kappa)
        numerator = scaling_factor * omega * dv / K_diag
        product_rho = rho_p[:-1] * rho_m[1:]
        denominator = 1.0 - np.convolve(np.array([1.0, 1.0]), product_rho)
        return numerator / denominator

    @staticmethod
    def alpha_middle(alpha_p: np.ndarray, alpha_m: np.ndarray) -> np.ndarray:
        return 1.0 - alpha_m - alpha_p

    def calculate_d2y_at_real_nodes(
        self, alpha: np.ndarray, samples: LocalSmoothInterp.SamplingNodes
    ) -> np.ndarray:
        alpha_m, alpha_p = SmoothConvexInterpolator.alpha_split(alpha)
        K_diag, K_up, K_down = self.K_diagonals_func(alpha_p, alpha_m, samples.times)
        K_matrix = self.K_matrix_from_diagonals(K_diag, K_up, K_down)
        dv = samples.finite_velocity_increments()
        d2y = np.linalg.solve(K_matrix.toarray(), 2.0 * dv * (1 + self.r_func()))
        return d2y

    def regularity_metric(
        self,
        alpha_p: np.ndarray,
        alpha_m: np.ndarray,
        samples: LocalSmoothInterp.SamplingNodes,
    ) -> np.ndarray:
        alpha = SmoothConvexInterpolator.alpha_merge(alpha_m=alpha_m, alpha_p=alpha_p)
        d2y = self.calculate_d2y_at_real_nodes(alpha, samples)
        d3y = np.diff(d2y)
        dt = np.diff(samples.times)
        dt = dt[1:-1]
        loss = 0.5 * d3y ** 2
        loss = loss / dt
        loss = loss / SmoothConvexInterpolator.alpha_middle(alpha_p, alpha_m)
        return np.sum(loss)

    @staticmethod
    def alpha_merge(alpha_m: np.ndarray, alpha_p: np.ndarray) -> np.ndarray:
        alpha = np.dstack((alpha_m, alpha_p)).flatten()
        return alpha

    @staticmethod
    def alpha_split(alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha_m = alpha[0::2]
        alpha_p = alpha[1::2]
        return alpha_m, alpha_p

    @staticmethod
    def calculate_augmented_times(
        sampling_times: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        alpha_m, alpha_p = SmoothConvexInterpolator.alpha_split(alpha)
        divided_intervals = sampling_times[1:-2]
        nb_inner_intervals = len(divided_intervals)
        if nb_inner_intervals == 0:
            return sampling_times
        dt = np.diff(sampling_times)
        dt = dt[1:-1]
        new_times = np.zeros((nb_inner_intervals, 3))
        new_times[:, 0] = divided_intervals
        for n in range(nb_inner_intervals):
            t_n = new_times[n, 0]
            dt_n = dt[n]
            new_times[n, 1] = t_n + dt_n * alpha_m[n]
            new_times[n, 2] = t_n + dt_n * (1.0 - alpha_p[n])

        new_times = new_times.reshape((-1,))
        new_times = np.insert(new_times, 0, sampling_times[0])
        new_times = np.concatenate((new_times, sampling_times[-2:]))
        return new_times

    def calculate_augmented_d2y(
        self, alpha: np.ndarray, samples: LocalSmoothInterp.SamplingNodes
    ) -> np.ndarray:
        d2y = self.calculate_d2y_at_real_nodes(alpha, samples)
        new_d2y = np.dstack([d2y] * 3)
        new_d2y = new_d2y.reshape((-1,))
        return new_d2y

    def calculate_augmented_y(
        self, alpha: np.ndarray, samples: LocalSmoothInterp.SamplingNodes
    ) -> np.ndarray:
        d2y = self.calculate_d2y_at_real_nodes(alpha=alpha, samples=samples)
        alpha_m, alpha_p = SmoothConvexInterpolator.alpha_split(alpha)
        sampling_times = samples.times
        inner_interval_times = sampling_times[1:-2]
        nb_inner_intervals = len(inner_interval_times)
        if nb_inner_intervals == 0:
            return samples.positions
        dt, v = samples.finite_diff_velocities()
        dt = dt[1:-1]
        v = v[1:-1]
        new_positions = np.zeros((nb_inner_intervals, 3))
        new_positions[:, 0] = samples.positions[1:-2]
        r_ = self.r_func()
        scaling_factor = 0.25 * (1.0 + self.kappa)
        for n in range(nb_inner_intervals):
            y_n = new_positions[n, 0]
            dt_n = dt[n]
            v_n = v[n]
            alpha_n = 1.0 - alpha_p[n] - alpha_m[n]
            Q_n = (
                scaling_factor
                * dt_n
                * (
                    r_ * alpha_n * d2y[n + 1]
                    + ((1.0 + r_) * alpha_m[n] + alpha_n) * d2y[n]
                )
            )
            Q_n2third = (
                scaling_factor
                * dt_n
                * (
                    r_ * alpha_n * d2y[n]
                    + ((1.0 + r_) * alpha_p[n] + alpha_n) * d2y[n + 1]
                )
            )
            q_n = v_n - (1.0 - alpha_m[n]) * Q_n - alpha_p[n] * Q_n2third
            new_positions[n, 1] = y_n + dt_n * alpha_m[n] * q_n
            q_n1third = v_n + alpha_m[n] * Q_n - alpha_p[n] * Q_n2third
            new_positions[n, 2] = new_positions[n, 1] + alpha_n * dt_n * q_n1third

        new_positions = new_positions.reshape((-1,))
        new_positions = np.insert(new_positions, 0, samples.positions[0])
        new_positions = np.concatenate((new_positions, samples.positions[-2:]))
        return new_positions

    def calculate_differential_augmented_nodes(
        self, alpha: np.ndarray, samples: LocalSmoothInterp.SamplingNodes
    ) -> LocalSmoothInterp.SamplingDifferentialNodes:
        new_t = SmoothConvexInterpolator.calculate_augmented_times(
            alpha=alpha, sampling_times=samples.times
        )
        new_d2y = self.calculate_augmented_d2y(alpha=alpha, samples=samples)
        new_y = self.calculate_augmented_y(alpha=alpha, samples=samples)
        augmented_nodes = LocalSmoothInterp.SamplingNodes(times=new_t, positions=new_y)
        new_dy = self.d2y_to_dy(nodes=augmented_nodes, d2y=new_d2y)
        augmented_diff_nodes = LocalSmoothInterp.SamplingDifferentialNodes(
            times=new_t, positions=new_y, velocities=new_dy, accelerations=new_d2y
        )
        return augmented_diff_nodes

    def convex_interp_split(
        self,
        alpha: np.ndarray,
        samples: LocalSmoothInterp.SamplingNodes,
        interpolated_times: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        augmented_diff_nodes = self.calculate_differential_augmented_nodes(
            alpha=alpha, samples=samples
        )
        LL0, LL1, LL2 = self.layered_interp_split(
            differential_node=augmented_diff_nodes,
            interpolated_times=interpolated_times,
        )
        return LL0, LL1, LL2

    def convex_interp(
        self,
        alpha: np.ndarray,
        samples: LocalSmoothInterp.SamplingNodes,
        interpolated_times: np.ndarray,
    ) -> np.ndarray:
        LL0, LL1, LL2 = self.convex_interp_split(alpha, samples, interpolated_times)
        return LL0 + LL1 + LL2

    def alpha_opt(self, samples: LocalSmoothInterp.SamplingNodes) -> np.ndarray:
        alpha_solver = AlphaSolver(interp=self, samples=samples)
        results = alpha_solver.solve()
        alpha = results.x
        return alpha

    def convex_interp_opt_split(
        self, samples: LocalSmoothInterp.SamplingNodes, interpolated_times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha = self.alpha_opt(samples=samples)
        LL0, LL1, LL2 = self.convex_interp_split(
            alpha=alpha, samples=samples, interpolated_times=interpolated_times
        )
        return LL0, LL1, LL2

    def convex_interp_opt(
        self, samples: LocalSmoothInterp.SamplingNodes, interpolated_times: np.ndarray
    ) -> np.ndarray:
        alpha = self.alpha_opt(samples=samples)
        return self.convex_interp(alpha, samples, interpolated_times)


def jac_f(f, x):
    dx = 1e-2
    y = f(x)
    n_x = len(x)
    y_is_scalar = isinstance(y, float)
    if y_is_scalar:
        n_y = 1
    else:
        n_y = len(y)
    jac_val = np.zeros((n_y, n_x))
    bump_matrix = dx * np.eye(n_x)
    for i in range(n_x):
        bump = bump_matrix[:, i]
        df = f(x + bump) - f(x - bump)
        jac_val[:, i] = df
    if y_is_scalar:
        jac_val = jac_val.reshape((-1,))
    return 0.5 * jac_val / dx


class AlphaSolver:
    def __init__(
        self, interp: SmoothConvexInterpolator, samples: LocalSmoothInterp.SamplingNodes
    ):
        self._interpolator = interp
        self._samples = samples

    @staticmethod
    def bounds_for_alpha(n_inner_intervals: int) -> scop.Bounds:
        lower_bound_1 = np.zeros((2 * n_inner_intervals))
        upper_bound_1 = np.ones((2 * n_inner_intervals))
        bounds = scop.Bounds(lb=lower_bound_1, ub=upper_bound_1)
        return bounds

    @staticmethod
    def linear_constraint_matrix_for_alpha(n_inner_intervals: int) -> np.ndarray:
        M = np.zeros((n_inner_intervals, 2 * n_inner_intervals))
        for i in range(n_inner_intervals):
            for j in range(2 * n_inner_intervals):
                if j == 2 * i:
                    M[i, j] = 1.0
                if j == 2 * i + 1:
                    M[i, j] = 1.0
        return M

    @staticmethod
    def bounds_for_alpha_mid(n_inner_intervals: int) -> Tuple[np.ndarray, np.ndarray]:
        upper_bound = np.ones(n_inner_intervals)
        lower_bound = 0.0 * upper_bound
        return lower_bound, upper_bound

    @staticmethod
    def linear_constraints(n_inner_intervals: int) -> scop.LinearConstraint:
        A = AlphaSolver.linear_constraint_matrix_for_alpha(n_inner_intervals)
        lower_bound, upper_bound = AlphaSolver.bounds_for_alpha_mid(n_inner_intervals)
        linear_constraints = scop.LinearConstraint(A=A, lb=lower_bound, ub=upper_bound)
        return linear_constraints

    def non_linear_constr(self, alpha: np.ndarray):
        samples_val = self._samples
        return self._interpolator.calculate_d2y_at_real_nodes(alpha, samples_val)

    def non_linear_constr_jac(self, alpha: np.ndarray):
        return jac_f(self.non_linear_constr, alpha)

    def non_const_bounds(self):
        dv = self._samples.finite_velocity_increments()
        lb = np.zeros_like(dv)
        lb[dv < 0] = -np.inf
        ub = np.zeros_like(dv)
        ub[dv > 0] = np.inf
        return lb, ub

    def objective_func(self, alpha: np.ndarray):
        alpha_m, alpha_p = SmoothConvexInterpolator.alpha_split(alpha)
        J22 = self._interpolator.regularity_metric(alpha_p, alpha_m, self._samples)
        return J22

    def objective_jac(self, alpha: np.ndarray):
        jac_val = jac_f(self.objective_func, alpha)
        return jac_val

    def solve(self):
        n_inner_intervals = len(self._samples.times) - 3
        alpha_init = np.zeros((2 * n_inner_intervals))
        bounds = AlphaSolver.bounds_for_alpha(n_inner_intervals)
        linear_constr = AlphaSolver.linear_constraints(n_inner_intervals)

        lb, ub = self.non_const_bounds()

        def convexity_constraint(alpha):
            return self.non_linear_constr(alpha)

        def convexity_constraint_jac(alpha):
            return self.non_linear_constr_jac(alpha)

        nonlinear_constr = scop.NonlinearConstraint(
            fun=convexity_constraint,
            lb=lb,
            ub=ub,
            jac=convexity_constraint_jac,
            hess=scop.BFGS(),
        )

        def regularity_fun(alpha):
            return self.objective_func(alpha)

        def regularity_jac(alpha):
            return self.objective_jac(alpha)

        res = scop.minimize(
            fun=regularity_fun,
            x0=alpha_init,
            method="trust-constr",
            jac=regularity_jac,
            constraints=[linear_constr, nonlinear_constr],
            options={"verbose": 1},
            bounds=bounds,
        )

        return res
