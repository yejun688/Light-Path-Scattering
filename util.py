from .imports import *


def coordinate_system(z):
    sign = jnp.copysign(1.0, z[2])
    a = -1.0 / (sign + z[2])
    b = z[0] * z[1] * a
    x = jnp.array([1.0 + sign * z[0]**2 * a, sign * b, -sign * z[0]])
    y = jnp.array([b, sign + z[1]**2 * a, -z[1]])
    return x, y


def spherical_to_cartesian(cos_theta, phi, z):
    direction = cos_theta * z
    x, y = coordinate_system(z)
    sin_theta = jnp.sqrt(1.0 - cos_theta**2)
    direction += jnp.cos(phi) * sin_theta * x
    direction += jnp.sin(phi) * sin_theta * y
    return direction


def normalized(v):
    return v / jnp.linalg.norm(v)


def sample_unit_sphere(xi):
    z = 1.0 - 2.0 * xi[0]
    r = jnp.sqrt(1.0 - z * z)
    phi = 2.0 * jnp.pi * xi[1]
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), z])


def fract(x):
    return x - jnp.floor(x)


def fib_unit_sphere(n):
    golden = 5**0.5 * 0.5 + 0.5
    idx = jnp.arange(0, n)
    phi = 2.0 * jnp.pi * fract(idx / golden)
    cos_theta = 1.0 - (2.0 * idx + 1.0) / n
    r = jnp.sqrt(1.0 - cos_theta**2)
    return jnp.array([r * jnp.cos(phi), r * jnp.sin(phi), cos_theta])
