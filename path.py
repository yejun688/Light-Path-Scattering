from .imports import *
from . import phase_func, medium, util


def extend(scene: Scene, path: PathState, key: Array) -> PathState:
    xi = random.uniform(key, (3,))
    wo, phase_weight = phase_func.sample(scene.phase_func, path.wi, xi[0:2])
    dist = medium.sample_free_path(scene.medium, xi[2])
    new_contrib = path.contribution
    new_contrib *= phase_weight * medium.albedo(scene.medium)
    return PathState(
        wi=-wo,
        position=path.position + dist * wo,
        contribution=new_contrib
    )


class ExtendState(NamedTuple):
    key: Array
    path: PathState


def extend_n(scene: Scene, path: PathState, num_vertices: int, key: Array) -> PathState:
    def extend_split_key(_idx, state: ExtendState):
        next_key, subkey = random.split(state.key)
        return ExtendState(
            key=next_key,
            path=extend(scene, state.path, subkey)
        )
    init = ExtendState(
        key=key,
        path=path
    )
    return lax.fori_loop(0, num_vertices, extend_split_key, init).path


def init_camera(scene: Scene, origin: Array, direction: Array, key: Array) -> PathState:
    dist = medium.sample_free_path(scene.medium, random.uniform(key))
    return PathState(
        wi=-direction,
        position=origin + dist * direction,
        contribution=medium.albedo(scene.medium),
    )


def init_light(scene: Scene, key: Array) -> PathState:
    xi = random.uniform(key, (3,))
    direction = util.sample_unit_sphere(xi[0:2])
    dist = medium.sample_free_path(scene.medium, xi[2])
    return PathState(
        wi=-direction,
        contribution=scene.light.intensity * 4.0 *
        jnp.pi * medium.albedo(scene.medium),
        position=scene.light.position + dist * direction,
    )
