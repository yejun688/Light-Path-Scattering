
from .imports import *
from .util import *
from . import phase_func, medium, nee


def connect(scene: Scene, path: PathState, key: Array) -> float:
    origin = path.position
    key, subkey = random.split(key)
    target = scene.light.position

    key, subkey = random.split(key)
    direction, phase_weight = phase_func.sample(scene.phase_func, jnp.array(
        [0.0, 0.0, -1.0]), random.uniform(subkey, (2,)))
    theta = jnp.arccos(direction[2])
    key, subkey = random.split(key)
    s = jnp.linalg.norm(target - origin)
    x1 = _sample_x1(origin, target, theta, subkey)

    contribution = path.contribution
    contribution *= phase_weight
    wo_origin = normalized(x1 - origin)
    contribution *= phase_func.eval(scene.phase_func, path.wi, wo_origin)
    total_dist = jnp.linalg.norm(
        x1 - origin) + jnp.linalg.norm(target - x1)
    contribution *= medium.eval_transmittance(scene.medium, total_dist)
    contribution *= scene.medium.mu_s
    # contribution *= scene.light.radiance / target_pdf_area
    # light_normal = normalized(target - scene.light.sphere.position)
    # wn = normalized(target - x1)
    # contribution *= jnp.maximum(0.0, -jnp.dot(light_normal, wn))
    contribution *= scene.light.intensity
    remaining_pdf = s * jnp.sin(theta) / theta
    contribution /= remaining_pdf
    return contribution


def connect_camera(scene: Scene, path: PathState, key: Array, eval_phase: bool) -> tuple[float, float, float]:
    origin = path.position
    target = jnp.zeros(3)
    key, subkey = random.split(key)
    direction, phase_weight = phase_func.sample(scene.phase_func, jnp.array(
        [0.0, 0.0, -1.0]), random.uniform(subkey, (2,)))
    theta = jnp.arccos(direction[2])
    key, subkey = random.split(key)
    s = jnp.linalg.norm(target - origin)
    x1 = _sample_x1(origin, target, theta, subkey)

    contribution = path.contribution
    if eval_phase:
        contribution *= phase_func.eval(scene.phase_func,
                                        path.wi, normalized(x1 - origin))

    contribution *= phase_weight
    total_dist = jnp.linalg.norm(x1 - origin) + jnp.linalg.norm(target - x1)
    contribution *= medium.eval_transmittance(scene.medium, total_dist)
    contribution *= scene.medium.mu_s
    remaining_pdf = s * jnp.sin(theta) / theta
    contribution /= remaining_pdf
    return nee.splat(scene.cam, PathState(
        wi=None,
        position=x1,
        contribution=contribution
    ))


def _sample_x1(origin: Array, target: Array, theta: float, key: Array) -> Array:
    xi = random.uniform(key, (2,))
    t = jnp.cos(theta - xi[0] * theta) * \
        jnp.sin(xi[0] * theta) / jnp.sin(theta)  # eq 41
    R2 = 1.0 / (4.0 * jnp.sin(theta)**2)
    r = jnp.sqrt(R2 - (0.5 - t)**2) - jnp.sqrt(R2 - 0.25)  # eq 12
    phi = xi[1] * 2.0 * jnp.pi
    z = normalized(target - origin)
    x, y = coordinate_system(z)
    s = jnp.linalg.norm(target - origin)
    x1 = origin
    x1 += (s * t) * z
    x1 += (s * r * jnp.cos(phi)) * x
    x1 += (s * r * jnp.sin(phi)) * y
    return x1
