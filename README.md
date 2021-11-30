# Smooth Interpolator Package

The objective of this package is to provide an implementation of the smooth interpolator class.

##Base class: Local smooth interpolator
The objets of the base class are both smooth but their constructor requires:
- The knowledge of the sampled positions
- The sampled velocities and accelerations.
Tip: If dy = None then it is calculated from d2y

Example of usage:

```
t_s = np.array([0.0, 1.0, 3.0])
x_s = t_s ** 2
dx_s = 2.0 * t_s
d2x_s = 2.0 * np.ones_like(t_s)

interpolator = LocalSmoothInterpolator.LocalSmoothInterp(
    x=t_s, y=x_s, d2y=d2x_s, dy=dx_s
)

t_i = np.linspace(0.0, 3.0)

y_i = interpolator(interpolated_times=t_i)
```

Another one with the calculation of the derivatives:

```
t_s = np.array([0.0, 1.0, 3.0])
x_s = t_s ** 2
dx_s = 2.0 * t_s
d2x_s = 2.0 * np.ones_like(t_s)

interpolator = LocalSmoothInterpolator.LocalSmoothInterp(
    x=t_s, y=x_s, d2y=d2x_s, dy=dx_s
)

t_i = np.linspace(0.0, 3.0)

y_i, dy_i, d2y_i = interpolator(interpolated_times=t_i, with_derivatives=True)
```


##Derived class: Taylor Local Smooth Interp
This derived class allows to interpolate from time and position samples only

Example of usage:

```
t_s = np.array([0.0, 1.0, 3.0])
x_s = t_s ** 2
interpolator = LocalSmoothInterpolator.TaylorLocalSmoothInterp(x=t_s, y=x_s)
t_i = np.linspace(0.0, 3.0)

y_i = interpolator(interpolated_times=t_i)
```

##Derived class: Convex smooth interpolator
The objects of that class are smooth, and they preserve locally the convexity property of the sampled positions.
Their constructor only requires the knowledge of those sample positions, not of that of their derivatives. 
Tip: Use fast_method = False might increase severely the computing time
Example of usage:

```
t_s = np.array([0.0, 1.0, 3.0, 5.0, 5.3])
x_s = t_s ** 2
interpolator = ConvexSmooth.SmoothConvexInterpolator(x=t_s, y=x_s)
t_i = np.linspace(0.0, 5.3)

y_i = interpolator(interpolated_times=t_i)
```

