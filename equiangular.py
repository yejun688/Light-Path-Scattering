from .imports import *
from .util import *
from . import medium, phase_func, nee

max_dist = 1e12


def connect(scene: Scene, path: PathState, wo: ArrayLike, key_outer: Array) -> float:
    """Since wo is passed in, this expects the phase function to already be included in the contribution"""
    t_near = 0
    ray = Ray(path.position, wo)
    light = scene.light
    t_far = max_dist
    key, subkey = random.split(key_outer)
    light_pos = light.position
    scatter_dist, dist_pdf = _sample(
        ray, light_pos, t_near, t_far, random.uniform(subkey))
    scatter_pos = path.position + scatter_dist * wo
    contribution = path.contribution
    contribution *= medium.eval_transmittance(
        scene.medium, scatter_dist) * scene.medium.mu_s / dist_pdf
    scattered_path = PathState(
        -wo, scatter_pos, contribution
    )
    return nee.connect(scene, scattered_path, key)


def connect_camera(scene: Scene, path: PathState, wo: ArrayLike, key: Array) -> tuple[float, float, float]:
    ray = Ray(path.position, wo)
    dist, pdf = _sample(ray, jnp.zeros(3), 0.0, max_dist, random.uniform(key))
    position = ray.origin + ray.direction * dist
    contribution = path.contribution
    contribution *= scene.medium.mu_s * \
        phase_func.eval(scene.phase_func, -wo, normalized(-position))
    contribution *= medium.eval_transmittance(
        scene.medium, jnp.linalg.norm(position) + dist)
    contribution *= 1.0/jnp.dot(position, position)
    contribution /= pdf
    return nee.splat(scene.cam, PathState(
        wi=-wo,
        position=position,
        contribution=contribution,
    ))


def _sample(ray: Ray, light_pos: Array, t_near: float, t_far: float, xi: float) -> tuple[float, float]:
    '''Returns (distance, pdf)'''
    delta = jnp.dot(light_pos - ray.origin, ray.direction)
    D = jnp.linalg.norm((ray.origin + ray.direction * delta) - light_pos)
    a = t_near - delta
    b = t_far - delta

    def regular():
        theta_a = jnp.arctan(a / D)
        theta_b = jnp.arctan(b / D)
        t = D * jnp.tan((1 - xi) * theta_a + xi * theta_b)
        pdf = D / (jnp.abs(theta_a - theta_b) * (D**2 + t**2))
        return (t, pdf)

    def singular():
        t = a * b / (b + (a - b) * xi)
        pdf = a * b / ((b - a) * t**2)
        return (t, pdf)
    k_eps = 1e-10
    t, pdf = lax.cond(D > k_eps, regular, singular)
    distance = delta + t
    return distance, pdf
