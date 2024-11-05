from .imports import *
from .util import *

def sample_solid_angle(light: SphereLight, position: ArrayLike, xi: ArrayLike) -> tuple[Array, float]:
    '''xi: 2 random numbers'''
    to_center = light.sphere.position - position
    # cone angle subtended by sphere
    cos_theta_max = jnp.sqrt(
        1.0 - (light.sphere.radius / jnp.linalg.norm(to_center))**2)
    # sample direction in cone uniformly
    cos_theta = (1.0 - xi[0]) + xi[0] * cos_theta_max
    phi = xi[1] * 2.0 * jnp.pi
    sampled_dir = spherical_to_cartesian(cos_theta, phi, normalized(to_center))
    solid_angle = 2.0 * jnp.pi * (1.0 - cos_theta_max)
    pdf_solid_angle = 1.0 / solid_angle
    return sampled_dir, pdf_solid_angle


def sample_area(light: SphereLight, xi: ArrayLike) -> tuple[Array, float]:
    '''xi: 2 random numbers'''
    sphere = light.sphere
    pdf_area = 1.0 / (4.0 * jnp.pi * sphere.radius**2)
    position = sphere.position + sphere.radius * sample_unit_sphere(xi)
    return position, pdf_area
