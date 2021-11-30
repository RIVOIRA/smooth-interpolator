import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.optimize as scop
from scipy.sparse import dia_matrix

from .LocalSmoothInterpolator import *


class SmoothConvexInterpolator(LocalSmoothInterp):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sigma: float = np.sqrt(1 / 5.0),
        fast_method: bool = True,
        _alpha_init: np.ndarray = None,
    ):
        LocalSmoothInterp.__init__(self, x=None, y=None, dy=None, d2y=None, sigma=sigma)
        self._real_nodes = SamplingNodes(times=x, positions=y)
        if not fast_method:
            alpha_solver = AlphaSolver(interp=self)
            results = alpha_solver.solve(_alpha_init)
            self._alpha = results.x
        else:
            alpha_solver = TurboAlphaSolver(interp=self)
            alpha_m, alpha_p = alpha_solver.solve()
            self._alpha = SmoothConvexInterpolator.alpha_merge(alpha_m, alpha_p)

        augmented_diff_nodes = self.calculate_differential_augmented_nodes()
        self._differential_samples = augmented_diff_nodes

    @property
    def real_nodes(self) -> SamplingNodes:
        return self._real_nodes

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

    def U_vector_func(
        self, alpha_p: np.ndarray, alpha_m: np.ndarray, sampling_times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        K_diag, K_up, K_down = self.K_diagonals_func(alpha_p, alpha_m, sampling_times)
        u_p = K_up / np.roll(K_diag, -1)
        u_m = K_down / np.roll(K_diag, 1)
        return u_p, u_m, K_diag

    def Omega_vector_func(
        self,
        alpha_p: np.ndarray,
        alpha_m: np.ndarray,
        samples: SamplingNodes,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sampling_times = samples.times
        u_p, u_m, K_diag = self.U_vector_func(alpha_p, alpha_m, sampling_times)
        dv = samples.finite_velocity_increments()
        dv_p = np.zeros_like(u_p)
        dv_p[0:-1] = dv[1:]
        dv_m = np.zeros_like(u_m)
        dv_m[1:] = dv[0:-1]
        omega = 1 - (dv_p * u_p + dv_m * u_m) / dv
        return omega, u_p, u_m, K_diag

    def d2y_approx(
        self,
        alpha_p: np.ndarray,
        alpha_m: np.ndarray,
        samples: SamplingNodes,
    ) -> np.ndarray:
        omega, u_p, u_m, K_diag = self.Omega_vector_func(alpha_p, alpha_m, samples)
        dv = samples.finite_velocity_increments()
        scaling_factor = 4.0 / (1.0 + self.kappa)
        numerator = scaling_factor * omega * dv / K_diag
        product_u = u_p[:-1] * u_m[1:]
        denominator = 1.0 - np.convolve(np.array([1.0, 1.0]), product_u)
        return numerator / denominator

    @staticmethod
    def alpha_middle(alpha_p: np.ndarray, alpha_m: np.ndarray) -> np.ndarray:
        return 1.0 - alpha_m - alpha_p

    def calculate_d2y_at_real_nodes(self, alpha) -> np.ndarray:
        samples = self._real_nodes
        alpha_m, alpha_p = SmoothConvexInterpolator.alpha_split(alpha)
        K_diag, K_up, K_down = self.K_diagonals_func(alpha_p, alpha_m, samples.times)
        K_matrix = self.K_matrix_from_diagonals(K_diag, K_up, K_down)
        dv = samples.finite_velocity_increments()
        d2y = np.linalg.solve(K_matrix.toarray(), 2.0 * dv * (1 + self.r_func()))
        return d2y

    @staticmethod
    def alpha_merge(alpha_m: np.ndarray, alpha_p: np.ndarray) -> np.ndarray:
        alpha = np.dstack((alpha_m, alpha_p)).flatten()
        return alpha

    @staticmethod
    def alpha_split(alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        alpha_m = alpha[0::2]
        alpha_p = alpha[1::2]
        return alpha_m, alpha_p

    def calculate_augmented_times(self) -> np.ndarray:
        alpha = self._alpha
        sampling_times = self._real_nodes.times
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

    def calculate_augmented_d2y(self) -> Tuple[np.ndarray, np.ndarray]:
        d2y = self.calculate_d2y_at_real_nodes(self._alpha)
        new_d2y = np.dstack([d2y] * 3)
        new_d2y = new_d2y.reshape((-1,))
        return new_d2y, d2y

    def calculate_augmented_y(self, d2y: np.ndarray) -> np.ndarray:
        alpha = self._alpha
        samples = self._real_nodes
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

    @staticmethod
    def clean_up_duplicates(
        t: np.ndarray, y: np.ndarray, d2y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_prev = t[0]
        t_new = np.array(t[0])
        y_new = np.array(y[0])
        d2y_new = np.array(d2y[0])
        for i, t_i in enumerate(t):
            if math.isclose(t_i, t_prev):
                continue
            t_new = np.append(t_new, t_i)
            t_prev = t_i
            y_new = np.append(y_new, y[i])
            d2y_new = np.append(d2y_new, d2y[i])

        return t_new, y_new, d2y_new

    def calculate_differential_augmented_nodes(
        self,
    ) -> SamplingDifferentialNodes:

        new_t = self.calculate_augmented_times()
        new_d2y, d2y = self.calculate_augmented_d2y()
        new_y = self.calculate_augmented_y(d2y)

        new_t, new_y, new_d2y = SmoothConvexInterpolator.clean_up_duplicates(
            t=new_t, y=new_y, d2y=new_d2y
        )
        augmented_nodes = SamplingNodes(times=new_t, positions=new_y)
        new_dy = self.d2y_to_dy(nodes=augmented_nodes, d2y=new_d2y)
        augmented_diff_nodes = SamplingDifferentialNodes(
            times=new_t, positions=new_y, velocities=new_dy, accelerations=new_d2y
        )
        return augmented_diff_nodes


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
    def __init__(self, interp: SmoothConvexInterpolator):
        self._interpolator = interp

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
        return self._interpolator.calculate_d2y_at_real_nodes(alpha)

    def non_linear_constr_jac(self, alpha: np.ndarray):
        return jac_f(self.non_linear_constr, alpha)

    def non_const_bounds(self):
        dv = self._interpolator.real_nodes.finite_velocity_increments()
        lb = np.zeros_like(dv)
        lb[dv < 0] = -np.inf
        ub = np.zeros_like(dv)
        ub[dv > 0] = np.inf
        return lb, ub

    def objective_func(self, alpha: np.ndarray):
        alpha_m, alpha_p = SmoothConvexInterpolator.alpha_split(alpha)
        alpha = SmoothConvexInterpolator.alpha_merge(alpha_m=alpha_m, alpha_p=alpha_p)
        d2y = self._interpolator.calculate_d2y_at_real_nodes(alpha)
        d3y = np.diff(d2y)
        dt = np.diff(self._interpolator.real_nodes.times)
        dt = dt[1:-1]
        loss = 0.5 * d3y ** 2
        loss = loss / dt
        loss = loss / SmoothConvexInterpolator.alpha_middle(alpha_p, alpha_m)
        J22 = np.sum(loss)
        return J22

    def objective_jac(self, alpha: np.ndarray):
        jac_val = jac_f(self.objective_func, alpha)
        return jac_val

    def solve(self, _alpha_init=None):
        n_inner_intervals = len(self._interpolator.real_nodes.times) - 3
        if _alpha_init is None:
            alpha_init = np.zeros((2 * n_inner_intervals))
        else:
            alpha_init = _alpha_init
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


class SubdivisionCoordinates:
    def __init__(self, c: float, r: float):
        self._c = c
        self._r = r

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, c_val: float):
        self._c = c_val

    def w_from_v(self, v: float) -> float:
        w = (1.0 + 1.0 / self._r) / v
        return w

    def v_from_w(self, w: float) -> float:
        return self.w_from_v(w)

    def p_from_w(self, w: float) -> float:
        p = (self._c * w + 1) * w - 1.0
        return p

    def u_from_p(self, p: float) -> float:
        return 1.0 / p

    def p_from_u(self, u: float) -> float:
        return 1.0 / u

    def w_from_p(self, p: float) -> float:
        if p == np.inf:
            return p
        num = 2.0 * (p + 1.0)
        den = 1.0 + np.sqrt(1.0 + 2.0 * self._c * num)
        return num / den

    def u_max(self) -> float:
        w_max = self.w_from_v(1.0)
        p_min = self.p_from_w(w_max)
        u_max_val = self.u_from_p(p_min)
        return u_max_val

    def w_min(self) -> float:
        return self.w_from_v(1.0)

    def relative_smoothness_loss(self, u: float) -> float:
        if u == 0.0:
            return np.inf
        u_red = u / self.u_max()
        p = self.p_from_u(u)
        w = self.w_from_p(p)
        w_red = w / self.w_min()
        return u_red * (w_red ** 3)

    def drelative_smoothness_loss_du(self, u: float) -> float:
        if u == 0.0:
            return -np.inf
        w = self.w_from_p(1.0 / u)
        dp = 2.0 * self._c * w + 1.0
        w_red = w / self.w_min()
        dloss = (w_red ** 3) / self.u_max()
        return dloss * (1.0 - 3.0 * self.p_from_u(u) / dp / w)

    def d2relative_smoothness_loss_du2(self, u: float) -> float:
        if u == 0.0:
            return np.inf
        w = self.w_from_p(1.0 / u)
        dp = 2.0 * self._c * w + 1.0
        d2loss = 3.0 * w / ((dp * u) ** 3)
        d2loss *= dp + 1.0
        scaling_factor = self.u_max() * (self.w_min() ** 3)
        return d2loss / scaling_factor


class IntervalSubdivision:
    def __init__(self, R_m: float, R_p: float, c_m: float, c_p: float, r: float):
        self._R = np.array([R_m, R_p])
        self._left_coordinate = SubdivisionCoordinates(c=c_m, r=r)
        self._right_coordinate = SubdivisionCoordinates(c=c_p, r=r)

    @property
    def R_m(self):
        return self._R[0]

    @property
    def R_p(self):
        return self._R[1]

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        self._R = R

    @R_m.setter
    def R_m(self, R_m: float):
        self._R[0] = R_m

    @R_p.setter
    def R_p(self, R_p: float):
        self._R[1] = R_p

    def set_c_m(self, c_m_val: float):
        self._left_coordinate.c = c_m_val

    def set_c_p(self, c_p_val: float):
        self._right_coordinate.c = c_p_val

    def left_weight(self):
        left_weight_val = 2 * self._R[0] * self._left_coordinate.u_max()
        if left_weight_val <= 1.0:
            return np.inf
        left_weight_val = 1.0 - 1.0 / left_weight_val
        left_weight_val = 1.0 / left_weight_val
        return left_weight_val

    def right_weight(self):
        right_weight_val = 2 * self._R[1] * self._right_coordinate.u_max()
        if right_weight_val <= 1.0:
            return np.inf
        right_weight_val = 1.0 - 1.0 / right_weight_val
        right_weight_val = 1.0 / right_weight_val
        return right_weight_val

    def relative_smoothness_loss(self, U: np.ndarray) -> float:
        left_loss = self._left_coordinate.relative_smoothness_loss(U[0])
        right_loss = self._right_coordinate.relative_smoothness_loss(U[1])
        return self.left_weight() * left_loss + self.right_weight() * right_loss

    def drelative_smoothness_loss_dum(self, U: np.ndarray) -> float:
        dleft_loss = self._left_coordinate.drelative_smoothness_loss_du(U[0])
        dright_loss = self._right_coordinate.drelative_smoothness_loss_du(U[1])
        dupdum = -self.R_m / self.R_p
        return (
            self.left_weight() * dleft_loss + dupdum * self.right_weight() * dright_loss
        )

    def drelative_smoothness_loss_dup(self, U: np.ndarray) -> float:
        dleft_loss = self._left_coordinate.drelative_smoothness_loss_du(U[0])
        dright_loss = self._right_coordinate.drelative_smoothness_loss_du(U[1])
        dumdup = -self.R_p / self.R_m
        return (
            dumdup * self.left_weight() * dleft_loss + self.right_weight() * dright_loss
        )

    def d2relative_smoothness_loss_dup2(self, U: np.ndarray) -> float:
        d2left_loss = self._left_coordinate.d2relative_smoothness_loss_du2(U[0])
        d2right_loss = self._right_coordinate.d2relative_smoothness_loss_du2(U[1])
        dumdup = -self.R_p / self.R_m
        return (
            dumdup ** 2
        ) * self.left_weight() * d2left_loss + self.right_weight() * d2right_loss

    def u_m_max(self):
        u_m_max = self._left_coordinate.u_max()
        if self.R_m > 0:
            u_m_max = min(u_m_max, 1.0 / self.R_m)
        return u_m_max

    def u_p_max(self):
        u_p_max = self._right_coordinate.u_max()
        if self.R_p > 0:
            u_p_max = min(u_p_max, 1.0 / self.R_p)
        return u_p_max

    def u_p_min(self):
        u_m_max = self.u_m_max()
        u_p_min = (1.0 - self.R_m * u_m_max) / self.R_p
        return u_p_min

    def is_convexity_condition_met(self) -> bool:
        u_m_max = self._left_coordinate.u_max()
        u_p_max = self._right_coordinate.u_max()
        return self.R_m * u_m_max + self.R_p * u_p_max <= 1.0

    def u_opt(self) -> np.ndarray:
        u_m_max = self.u_m_max()
        u_p_max = self.u_p_max()
        if self.is_convexity_condition_met():
            return np.array([u_m_max, u_p_max])
        u_p = (1.0 - self.R_m * u_m_max) / self.R_p
        U_m_max = np.array([u_m_max, u_p])
        if self.left_weight() == np.inf:
            return U_m_max

        u_m = (1.0 - self.R_p * u_p_max) / self.R_m
        U_p_max = np.array([u_m, u_p_max])
        if self.right_weight() == np.inf:
            return U_p_max
        dloss_dum_m_max = self.drelative_smoothness_loss_dum(U_m_max)
        if dloss_dum_m_max <= 0.0:
            return U_m_max
        dloss_dum_p_max = self.drelative_smoothness_loss_dup(U_p_max)
        if dloss_dum_p_max <= 0.0:
            return U_p_max

        def um_func(u_p):
            um_val = (1.0 - self.R_p * u_p) / self.R_m
            return um_val

        def objective_func(u_p):
            u_m = um_func(u_p)
            U = np.array([u_m, u_p])
            return self.drelative_smoothness_loss_dup(U)

        def dobjective_func(u_p):
            u_m = um_func(u_p)
            U = np.array([u_m, u_p])
            return self.d2relative_smoothness_loss_dup2(U)

        u_mid = 0.5 * (u_p_max + self.u_p_min())
        sol = scop.root_scalar(
            f=objective_func, x0=u_mid, fprime=dobjective_func, method="newton"
        )
        u_p_opt = sol.root
        u_m_opt = (1.0 - self.R_p * u_p_opt) / self.R_m
        return np.array([u_m_opt, u_p_opt])

    def v_opt(self) -> np.ndarray:
        U = self.u_opt()
        v_m = 0.0
        v_p = 0.0
        if U[0] > 0.0:
            w_m = self._left_coordinate.w_from_p(1.0 / U[0])
            v_m = self._left_coordinate.v_from_w(w_m)
        if U[1] > 0.0:
            w_p = self._right_coordinate.w_from_p(1.0 / U[1])
            v_p = self._right_coordinate.v_from_w(w_p)
        return np.array([v_m, v_p])


class TurboAlphaSolver:
    def __init__(self, interp: SmoothConvexInterpolator):
        self._interpolator = interp

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        dv = self._interpolator.real_nodes.finite_velocity_increments()
        n_inner_intervals = len(self._interpolator.real_nodes.times) - 3
        n_inner_nodes = n_inner_intervals + 1
        r = self._interpolator.r_func()
        split_list = [
            IntervalSubdivision(R_m=0.0, R_p=0.0, c_m=0.0, c_p=0.0, r=r)
            for i in range(n_inner_nodes)
        ]
        could_be_split_list = [False] * n_inner_nodes
        for i in range(n_inner_nodes):
            same_sign = True
            if i > 0:
                if dv[i] == 0:
                    could_be_split_list[i] = False
                    continue
                R_m = dv[i - 1] / dv[i]
                if R_m < 0.0 and not math.isclose(R_m, 0.0):
                    same_sign = False
                if i > 1:
                    R_m += max(0.0, -dv[i - 2] / dv[i])
                split_list[i].R_m = R_m
            if i < n_inner_nodes - 1:
                R_p = dv[i + 1] / dv[i]
                if R_p < 0.0 and not math.isclose(R_p, 0.0):
                    same_sign = False
                if i < n_inner_nodes - 2:
                    R_p += max(0.0, -dv[i + 2] / dv[i])
                split_list[i].R_p = R_p
            if same_sign:
                could_be_split_list[i] = True

        alpha_m = 0.0 * 1e-10 * np.ones((n_inner_intervals))
        alpha_p = 0.0 * 1e-10 * np.ones((n_inner_intervals))

        for i in range(n_inner_nodes):
            if not could_be_split_list[i]:
                continue
            if split_list[i].is_convexity_condition_met():
                continue
            v_opt_vec = split_list[i].v_opt()
            if v_opt_vec[0] < 1.0:
                alpha_p[i - 1] = 1.0 - v_opt_vec[0]
            if v_opt_vec[1] < 1.0:
                alpha_m[i] = 1.0 - v_opt_vec[1]

        L_matrix = self._interpolator.L_matrix_func(alpha_p, alpha_m)
        no_split_row = r * np.ones(2)
        L_matrix = np.vstack([L_matrix, no_split_row])
        L_matrix = np.vstack([no_split_row, L_matrix])
        c_factor = r / (1.0 + r) ** 2
        dt = np.diff(self._interpolator.real_nodes.times)
        for i in range(n_inner_nodes):
            if not could_be_split_list[i]:
                continue
            if i > 0:
                c_m = c_factor * (1.0 + r - L_matrix[i - 1, 1]) * dt[i] / dt[i + 1]
                split_list[i].set_c_m(c_m)
            if i < n_inner_nodes - 1:
                c_p = c_factor * (1.0 + r - L_matrix[i + 2, 0]) * dt[i + 2] / dt[i + 1]
                split_list[i].set_c_p(c_p)

        for i in range(n_inner_nodes):
            if not could_be_split_list[i]:
                continue
            if split_list[i].is_convexity_condition_met():
                continue
            v_opt_vec = split_list[i].v_opt()
            if v_opt_vec[0] < 1.0:
                alpha_p[i - 1] = 1.0 - v_opt_vec[0]
            if v_opt_vec[1] < 1.0:
                alpha_m[i] = 1.0 - v_opt_vec[1]

        return alpha_m, alpha_p
