from .imports import *

# An Approximate Mie Scattering Function for Fog and Cloud Rendering, Jendersie and d'Eon 2023


class Mie(NamedTuple):
    d: float  # 5 < d < 50 (water droplet diameter in um)


class Draine(NamedTuple):
    alpha: float
    g: float


def eval(pf: PhaseFunction, wi: ArrayLike, wo: ArrayLike) -> float:
    raise NotImplementedError


def sample_fwd(pf: PhaseFunction, wi: ArrayLike, xi: ArrayLike) -> tuple[Array, float]:
    # TODO not implemented
    return (jnp.array([0.0, 0.0, 1.0]), 0.0)


def sample_bwd(pf: PhaseFunction, wi: ArrayLike, xi: ArrayLike) -> tuple[Array, float]:
    # TODO not implemented
    return (jnp.array([0.0, 0.0, 1.0]), 0.0)


def sample(pf: PhaseFunction, wi: ArrayLike, xi: ArrayLike) -> tuple[Array, float]:
    # TODO not implemented
    return (jnp.array([0.0, 0.0, 1.0]), 0.0)


def pdf_fwd(pf: PhaseFunction, wi: ArrayLike, wo: ArrayLike) -> float:
    return 0.0


def pdf_bwd(pf: PhaseFunction, wi: ArrayLike, wo: ArrayLike) -> float:
    return 0.0


def pdf(pf: PhaseFunction, wi: ArrayLike, wo: ArrayLike) -> float:
    return 0.0

# w_D(d)


def weight_draine(mie):
    return jnp.exp(-0.599085/(mie.d - 0.641583))


def alpha_draine(mie):
    return jnp.exp(3.62489 - 8.29288/(mie.d + 5.52825))


def g_draine(mie):
    return jnp.exp(-2.20679/(mie.d + 3.91029) - 0.428934)


def g_hg(mie):
    return jnp.exp(-0.0990567/(mie.d - 1.67154))


def hg_lobe(mie):
    return Hg(g_hg(mie))


def draine_lobe(mie):
    return Draine(alpha=alpha_draine(mie), g=g_draine(mie))


def eval(pf: PhaseFunction, wi: ArrayLike, wo: ArrayLike):
    return 0.0
    #w = weight_draine(mie)
    #return (1.0 - w) * hg_eval(hg_lobe(mie)) + w * draine_eval(draine_lobe(mie))


def draine_eval():
    g = 0.0  # todo
    cos_theta = 0.0  # todo
    alpha = 0.0  # todo
    phi = 0.25 / jnp.pi
    g2 = g * g
    phi *= 1 - g2
    phi /= (1.0 + g2 - 2.0 * g * cos_theta)**1.5
    phi *= 1.0 + alpha * cos_theta**2
    phi /= 1.0 + alpha * (1.0 + 2.0 * g2) / 3.0
    return phi
