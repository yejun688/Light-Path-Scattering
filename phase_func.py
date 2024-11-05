from .imports import *

from . import hg, mie

# convention for directions:
# * they point away from the interaction
# * wi is the incoming direction as sampled


def eval(pf: PhaseFunction, wi: ArrayLike, wo: ArrayLike) -> float:
    # does the int conversion, probably there's a more elgant solution
    family = pf.phase_family + 0
    pf_value = lax.switch(family, (hg.eval, mie.eval), pf, wi, wo)
    return lax.select(_eval_mask(pf, wi, wo), pf_value, 0.0)

# xi: 2 random numbers
# returns (sampled direction, sampling weight)


def sample(pf: PhaseFunction, wi: ArrayLike, xi: ArrayLike) -> tuple[Array, float]:
    return lax.switch(_switch_index(pf), [
        hg.sample_fwd,
        hg.sample_bwd,
        hg.sample,
        mie.sample_fwd,
        mie.sample_bwd,
        mie.sample,
    ], pf, wi, xi)


def pdf(pf: PhaseFunction, wi: ArrayLike, wo: ArrayLike) -> float:
    return lax.switch(_switch_index(pf), [
        hg.pdf_fwd,
        hg.pdf_bwd,
        hg.pdf,
        mie.pdf_fwd,
        mie.pdf_bwd,
        mie.pdf,
    ], pf, wi, wo)


def _eval_mask(pf: PhaseFunction, wi: ArrayLike, wo: ArrayLike) -> bool:
    is_fwd = -jnp.dot(wi, wo) >= 0.0
    is_bwd = jnp.logical_not(is_fwd)
    h = pf.scatter_hemi
    return (h == ScatterHemisphere.ALL) | ((h == ScatterHemisphere.FWD) & is_fwd) | ((h == ScatterHemisphere.BWD) & is_bwd)


def _switch_index(pf: PhaseFunction):
    return pf.phase_family * 3 + pf.scatter_hemi
