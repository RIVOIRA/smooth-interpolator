import unittest

import numpy as np
from src.sminterp import ConvexSmooth, LocalSmoothInterpolator

numerical_tol = 1e-10


def are_almost_equal(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.max(abs(v1 - v2)) < numerical_tol


class TestLocalSmoothInterpolatorOverUnitInterval(unittest.TestCase):
    def test_constantness(self):
        interpolator = LocalSmoothInterpolator.LocalSmoothInterp()
        x_s = np.array([1.0, 1.0])
        dx_s = np.zeros_like(x_s)
        d2x_s = np.zeros_like(x_s)
        t_i = np.linspace(0.0, 1.0)

        diff_nodes = LocalSmoothInterpolator.LocalSmoothInterp.UnitIntervalDifferentialNodes(
            positions=x_s, velocities=dx_s, accelerations=d2x_s
        )
        y_i = interpolator.unit_layer_interp(
            differential_node=diff_nodes, interpolated_times=t_i
        )
        flat_one = np.ones_like(t_i)
        is_flat = are_almost_equal(y_i, flat_one)
        self.assertTrue(is_flat)

    def test_core_function_is_worth_1half_at_1half(self):
        interpolator = LocalSmoothInterpolator.LocalSmoothInterp()
        x_s = np.array([0.0, 1.0])
        dx_s = np.zeros_like(x_s)
        d2x_s = np.zeros_like(x_s)
        t_i = 0.5
        diff_nodes = LocalSmoothInterpolator.LocalSmoothInterp.UnitIntervalDifferentialNodes(
            positions=x_s, velocities=dx_s, accelerations=d2x_s
        )
        y_i = interpolator.unit_layer_interp(
            differential_node=diff_nodes, interpolated_times=t_i
        )
        is_worth_1half = np.max(abs(y_i - 0.5)) < numerical_tol
        self.assertTrue(is_worth_1half)

    def test_identity(self):
        interpolator = LocalSmoothInterpolator.LocalSmoothInterp()
        x_s = np.array([0.0, 1.0])
        dx_s = np.ones_like(x_s)
        d2x_s = np.zeros_like(x_s)
        t_i = np.linspace(0.0, 1.0)

        diff_nodes = LocalSmoothInterpolator.LocalSmoothInterp.UnitIntervalDifferentialNodes(
            positions=x_s, velocities=dx_s, accelerations=d2x_s
        )
        y_i = interpolator.unit_layer_interp(
            differential_node=diff_nodes, interpolated_times=t_i
        )
        is_identity = are_almost_equal(y_i, t_i)
        self.assertTrue(is_identity)

    def test_parabola(self):
        interpolator = LocalSmoothInterpolator.LocalSmoothInterp()
        x_s = np.array([0.0, 1.0])
        dx_s = 2.0 * x_s
        d2x_s = 2.0 * np.ones_like(x_s)
        t_i = np.linspace(0.0, 1.0)

        diff_nodes = LocalSmoothInterpolator.LocalSmoothInterp.UnitIntervalDifferentialNodes(
            positions=x_s, velocities=dx_s, accelerations=d2x_s
        )
        y_i = interpolator.unit_layer_interp(
            differential_node=diff_nodes, interpolated_times=t_i
        )
        is_parabolic = are_almost_equal(y_i, t_i ** 2)
        self.assertTrue(is_parabolic)


class TestLocalSmoothInterpolatorOverTwoIntervals(unittest.TestCase):
    def test_constantness(self):
        interpolator = LocalSmoothInterpolator.LocalSmoothInterp()
        t_s = np.array([0.0, 1.0, 3.0])
        x_s = np.array([1.0, 1.0, 1.0])
        dx_s = np.zeros_like(x_s)
        d2x_s = np.zeros_like(x_s)
        t_i = np.linspace(0.0, 3.0)

        diff_nodes = LocalSmoothInterpolator.LocalSmoothInterp.SamplingDifferentialNodes(
            times=t_s, positions=x_s, velocities=dx_s, accelerations=d2x_s
        )
        y_i = interpolator.layered_interp(
            differential_node=diff_nodes, interpolated_times=t_i
        )
        flat_one = np.ones_like(t_i)
        is_flat = are_almost_equal(y_i, flat_one)
        self.assertTrue(is_flat)

    def test_identity(self):
        interpolator = LocalSmoothInterpolator.LocalSmoothInterp()
        t_s = np.array([0.0, 1.0, 3.0])
        x_s = t_s
        dx_s = np.ones_like(x_s)
        d2x_s = np.zeros_like(x_s)
        t_i = np.linspace(0.0, 3.0)

        diff_nodes = LocalSmoothInterpolator.LocalSmoothInterp.SamplingDifferentialNodes(
            times=t_s, positions=x_s, velocities=dx_s, accelerations=d2x_s
        )
        y_i = interpolator.layered_interp(
            differential_node=diff_nodes, interpolated_times=t_i
        )
        is_identity = are_almost_equal(y_i, t_i)
        self.assertTrue(is_identity)

    def test_parabola(self):
        interpolator = LocalSmoothInterpolator.LocalSmoothInterp()
        t_s = np.array([0.0, 1.0, 3.0])
        x_s = t_s ** 2
        dx_s = 2.0 * t_s
        d2x_s = 2.0 * np.ones_like(t_s)
        t_i = np.linspace(0.0, 3.0)

        diff_nodes = LocalSmoothInterpolator.LocalSmoothInterp.SamplingDifferentialNodes(
            times=t_s, positions=x_s, velocities=dx_s, accelerations=d2x_s
        )
        y_i = interpolator.layered_interp(
            differential_node=diff_nodes, interpolated_times=t_i
        )
        is_parabola = are_almost_equal(y_i, t_i ** 2)
        self.assertTrue(is_parabola)


class TestSamplingDifferentialNodes(unittest.TestCase):
    def test_constantness(self):
        t_s = np.array([0.0, 1.0, 3.0, 5.0, 5.3])
        x_s = np.ones_like(t_s)

        samples = LocalSmoothInterpolator.LocalSmoothInterp.SamplingNodes(
            times=t_s, positions=x_s
        )
        v, a = LocalSmoothInterpolator.LocalSmoothInterp.SamplingDifferentialNodes.compute_taylor_derivatives(samples)
        is_zero_v = are_almost_equal(v, np.zeros_like(v))
        is_zero_a = are_almost_equal(a, np.zeros_like(a))
        self.assertTrue(is_zero_v and is_zero_a)

    def test_identity(self):
        t_s = np.array([0.0, 1.0, 3.0, 5.0, 5.3])
        x_s = t_s
        samples = LocalSmoothInterpolator.LocalSmoothInterp.SamplingNodes(
            times=t_s, positions=x_s
        )
        v, a = LocalSmoothInterpolator.LocalSmoothInterp.SamplingDifferentialNodes.compute_taylor_derivatives(samples)
        is_unit_v = are_almost_equal(v, np.ones_like(v))
        is_zero_a = are_almost_equal(a, np.zeros_like(a))
        self.assertTrue(is_unit_v and is_zero_a)

    def test_parabola(self):
        t_s = np.array([0.0, 1.0, 3.0, 5.0, 5.3])
        x_s = t_s**2
        samples = LocalSmoothInterpolator.LocalSmoothInterp.SamplingNodes(
            times=t_s, positions=x_s
        )
        v, a = LocalSmoothInterpolator.LocalSmoothInterp.SamplingDifferentialNodes.compute_taylor_derivatives(samples)
        is_linear_v = are_almost_equal(v, 2.*t_s)
        is_unit_a = are_almost_equal(a, 2.*np.ones_like(a))
        self.assertTrue(is_linear_v and is_unit_a)


class TestLocalSmoothInterpolatorWithTaylorOverTwoIntervals(unittest.TestCase):
    def test_constantness(self):
        interpolator = LocalSmoothInterpolator.LocalSmoothInterp()
        t_s = np.array([0.0, 1.0, 3.0])
        x_s = np.array([1.0, 1.0, 1.0])
        t_i = np.linspace(0.0, 3.0)

        samples = LocalSmoothInterpolator.LocalSmoothInterp.SamplingNodes(
            times=t_s, positions=x_s
        )
        y_i = interpolator.layered_interp_taylor(
            samples= samples, interpolated_times=t_i
        )
        flat_one = np.ones_like(t_i)
        is_flat = are_almost_equal(y_i, flat_one)
        self.assertTrue(is_flat)

    def test_identity(self):
        interpolator = LocalSmoothInterpolator.LocalSmoothInterp()
        t_s = np.array([0.0, 1.0, 3.0])
        x_s = t_s
        t_i = np.linspace(0.0, 3.0)

        samples = LocalSmoothInterpolator.LocalSmoothInterp.SamplingNodes(
            times=t_s, positions=x_s
        )
        y_i = interpolator.layered_interp_taylor(
            samples=samples, interpolated_times=t_i
        )
        is_identity = are_almost_equal(y_i, t_i)
        self.assertTrue(is_identity)

    def test_parabola(self):
        interpolator = LocalSmoothInterpolator.LocalSmoothInterp()
        t_s = np.array([0.0, 1.0, 3.0])
        x_s = t_s ** 2
        t_i = np.linspace(0.0, 3.0)

        samples = LocalSmoothInterpolator.LocalSmoothInterp.SamplingNodes(
            times=t_s, positions=x_s
        )
        y_i = interpolator.layered_interp_taylor(
            samples=samples, interpolated_times=t_i
        )
        is_parabola = are_almost_equal(y_i, t_i ** 2)
        self.assertTrue(is_parabola)


class TestSmoothConvexOver3Intervals(unittest.TestCase):
    def test_parabola(self):
        interpolator = ConvexSmooth.SmoothConvexInterpolator()
        t_s = np.array([0.0, 1.0, 3.0, 5.0, 5.3])
        x_s = t_s ** 2
        t_i = np.linspace(0.0, 5.3)

        samples = LocalSmoothInterpolator.LocalSmoothInterp.SamplingNodes(
            times=t_s, positions=x_s
        )
        y_i = interpolator.convex_interp_opt(samples=samples, interpolated_times=t_i)
        is_parabola = are_almost_equal(y_i, t_i ** 2)
        self.assertTrue(is_parabola)

    def test_identity(self):
        interpolator = ConvexSmooth.SmoothConvexInterpolator()
        t_s = np.array([0.0, 1.0, 3.0, 5.0, 5.3])
        x_s = t_s
        t_i = np.linspace(0.0, 5.3)

        samples = LocalSmoothInterpolator.LocalSmoothInterp.SamplingNodes(
            times=t_s, positions=x_s
        )
        y_i = interpolator.convex_interp_opt(samples=samples, interpolated_times=t_i)
        is_identity = are_almost_equal(y_i, t_i)
        self.assertTrue(is_identity)

    def test_constantness(self):
        interpolator = ConvexSmooth.SmoothConvexInterpolator()
        t_s = np.array([0.0, 1.0, 3.0, 5.0, 5.3])
        x_s = np.ones_like(t_s)
        t_i = np.linspace(0.0, 5.3)

        samples = LocalSmoothInterpolator.LocalSmoothInterp.SamplingNodes(
            times=t_s, positions=x_s
        )
        y_i = interpolator.convex_interp_opt(samples=samples, interpolated_times=t_i)
        flat_on = np.ones_like(t_i)
        is_flat = are_almost_equal(y_i, flat_on)
        self.assertTrue(is_flat)


if __name__ == "__main__":
    unittest.main()
