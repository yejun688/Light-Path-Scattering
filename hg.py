from .imports import *
from .util import *

"""
Henyey-Greenstein phase function
"""


def eval(pf: PhaseFunction, wi: ArrayLike, wo: ArrayLike) -> float:
    hg = pf.param
    return _hg(-jnp.dot(wi, wo), hg.g)


def sample_fwd(pf: PhaseFunction, wi: ArrayLike, xi: ArrayLike) -> tuple[Array, float]:
    g = pf.param.g
    y = jnp.sqrt(1.0 + g * g)
    z = 4.0 * g + 2.0 * xi[0]**2 * g - 3.0 * \
        (1.0 + y) - (-3.0 + g) * g * (-g + y)
    cos_theta = xi[0] * (1.0 + g*g) * (1.0 + y + g * (-2.0 + xi[0] + g - y))
    cos_theta /= 1.0 + y + g * (z - 2.0 * xi[0] * (-1.0 + g) * (1.0 - g + y))
    phi = 2.0 * jnp.pi * xi[1]
    wo = spherical_to_cartesian(cos_theta, phi, -wi)
    weight = 0.5 * (1.0 + g) / (g * y) * (-1.0 + g + y)
    return (wo, weight)


def sample_bwd(pf: PhaseFunction, wi: ArrayLike, xi: ArrayLike) -> tuple[Array, float]:
    g = pf.param.g
    y = jnp.sqrt(1.0 + g * g)
    cos_theta = (-1.0 + xi[0])*(1.0 + g**2) * \
        (-xi[0] + g**2 + xi[0] * y + g * y)
    cos_theta /= g * (2.0 * xi[0] - 2.0 * xi[0]**2 + g + 2.0 * xi[0] * g**2 + g **
                      3 + y - 2.0 * xi[0] * y + 2.0 * xi[0]**2 * y + 2.0 * xi[0] * g * y + g**2 * y)
    phi = 2.0 * jnp.pi * xi[1]
    wo = spherical_to_cartesian(cos_theta, phi, -wi)
    weight = 0.0  # TODO
    return (wo, weight)


def sample(pf: PhaseFunction, wi: ArrayLike, xi: ArrayLike) -> tuple[Array, float]:
    g = pf.param.g
    cos_theta_uniform = 1.0 - 2.0 * xi[0]
    cos_theta_hg = 0.5 / g * \
        (1.0 + g * g - ((1.0 - g * g) / (1.0 + g - 2.0 * g * xi[0]))**2)
    cos_theta = lax.select(jnp.abs(g) < 1e-3, cos_theta_uniform, cos_theta_hg)
    phi = 2.0 * jnp.pi * xi[1]
    wo = spherical_to_cartesian(cos_theta, phi, -wi)
    return (wo, 1.0)


def pdf_fwd(pf: PhaseFunction, wi: ArrayLike, wo: ArrayLike) -> float:
    cos_theta = -jnp.dot(wi, wo)
    g = pf.param.g
    y = jnp.sqrt(1.0 + g * g)
    f_hgf = -(-1.0 + g) * g * y
    f_hgf /= (-1.0 + g + y) * (1.0 + g*g - 2.0 * g * cos_theta)**1.5
    return lax.select(cos_theta >= 0.0, f_hgf / (2.0 * jnp.pi), 0.0)


def pdf_bwd(pf: PhaseFunction, wi: ArrayLike, wo: ArrayLike) -> float:
    return 0.0  # TODO


def pdf(pf: PhaseFunction, wi: ArrayLike, wo: ArrayLike) -> float:
    return eval(pf, wi, wo)


def _hg(cos_theta, g):
    denom = jnp.maximum(1.0 + g * g - 2.0 * g * cos_theta, 0.0)
    inv_4pi = 0.25 / jnp.pi
    return inv_4pi * (1.0 - g * g) / (denom * jnp.sqrt(denom))
