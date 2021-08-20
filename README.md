# Smooth Interpolator Package

The objective of this package is to provide an implementation of the smooth interpolator class.

##Base class: Local smooth interpolator
The objets of the base class are both smooth but their constructor requires the knowledge of the sampled positions together with the sampled of velocities and accelerations.

Example of usage:

```
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
```
##Derived class: Convex smooth interpolator
The objects of that class are smooth and they preserve locally the convexity property of the sampled positions. Their constructor only requires the knowledge of those sample positions, not of that of their derivatives. 

Example of usage:

```
interpolator = ConvexSmooth.SmoothConvexInterpolator()
t_s = np.array([0.0, 1.0, 3.0, 5.0, 5.3])
x_s = t_s ** 2
t_i = np.linspace(0.0, 5.3)

samples = LocalSmoothInterpolator.LocalSmoothInterp.SamplingNodes(
    times=t_s, positions=x_s
)
y_i = interpolator.convex_interp_opt(samples=samples, interpolated_times=t_i)
is_parabola = are_almost_equal(y_i, t_i ** 2)
```

