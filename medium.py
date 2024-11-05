from .imports import *

"""
A homogeneous medium.
"""


def eval_transmittance(medium: Medium, distance: float) -> float:
    tau = medium.mu_t * distance  # optical thickness
    return jnp.exp(-tau)


def pdf_free_path(medium: Medium, distance):
    return medium.mu_t * eval_transmittance(medium, distance)

# xi: 1 random number


def sample_free_path(medium: Medium, xi) -> float:
    return -jnp.log(1.0 - xi) / medium.mu_t


def sample_free_path_before(medium: Medium, dist_max, xi) -> tuple[float, float]:
    weight = 1.0 - jnp.exp(-medium.mu_t * dist_max)
    dist = sample_free_path(medium, xi * weight)
    return dist, weight


def pdf_free_path_before(medium: Medium, dist_max, dist) -> float:
    weight = 1.0 - jnp.exp(-medium.mu_t * dist_max)
    return pdf_free_path(medium, dist) / weight


def mean_free_path(medium: Medium) -> float:
    return 1.0 / medium.mu_t


def albedo(medium: Medium) -> float:
    return medium.mu_s / medium.mu_t
