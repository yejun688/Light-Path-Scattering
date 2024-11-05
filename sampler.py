from .imports import *
from . import path as path_mod
from . import nee as nee_mod
from . import equiangular as equiangular_mod
from . import omnee as omnee_mod
from . import bridge as bridge_mod
from . import mvnee as mvnee_mod
from . import medium, phase_func, util
import os

"""
Path sampler collection

* nee pt/lt
* equiangular pt/lt
* bridge pt/lt
* omnee pt/lt
* mvnee pt/lt
"""


def nee(config: RenderConfig, scene: Scene, position: Array, direction: Array, key: Array) -> float:
    assert (config.path_length >= 3)
    key, subkey = random.split(key)
    path = path_mod.init_camera(scene, position, direction, subkey)
    key, subkey = random.split(key)
    path = path_mod.extend_n(scene, path, config.path_length - 3, subkey)
    return nee_mod.connect(scene, path, key)


def nee_lt(config: RenderConfig, scene: Scene, key: Array) -> tuple[float, float, float]:
    assert (config.path_length >= 3)
    key, subkey = random.split(key)
    path = path_mod.init_light(scene, subkey)
    key, subkey = random.split(key)
    path = path_mod.extend_n(scene, path, config.path_length - 3, subkey)
    return nee_mod.connect_camera(scene, path, key)


def equiangular(config: RenderConfig, scene: Scene, position: Array, direction: Array, key: Array) -> float:
    assert (config.path_length >= 3)
    if config.path_length == 3:
        path = PathState(
            wi=None,
            position=jnp.zeros(3),
            contribution=1.0,
        )
        wo = direction
    else:
        key, subkey = random.split(key)
        path = path_mod.init_camera(scene, position, direction, subkey)
        key, subkey = random.split(key)
        path = path_mod.extend_n(scene, path, config.path_length - 4, subkey)
        key, subkey = random.split(key)
        wo, pf_weight = phase_func.sample(
            scene.phase_func, path.wi, random.uniform(subkey, (2,)))
        path = PathState(
            wi=path.wi,
            position=path.position,
            contribution=path.contribution * pf_weight,
        )
    return equiangular_mod.connect(scene, path, wo, key)


def equiangular_lt(config: RenderConfig, scene: Scene, key: Array) -> tuple[float, float, float]:
    assert (config.path_length >= 3)
    if config.path_length == 3:
        key, subkey = random.split(key)
        wo = util.sample_unit_sphere(random.uniform(subkey, (2,)))
        path = PathState(
            wi=None,
            position=scene.light.position,
            contribution=scene.light.intensity * 4.0 * jnp.pi,
        )
    else:
        key, subkey = random.split(key)
        path = path_mod.init_light(scene, subkey)
        key, subkey = random.split(key)
        path = path_mod.extend_n(scene, path, config.path_length - 4, subkey)
        key, subkey = random.split(key)
        wo, pf_weight = phase_func.sample(
            scene.phase_func, path.wi, random.uniform(subkey, (2,)))
        path = PathState(
            wi=path.wi,
            position=path.position,
            contribution=path.contribution * pf_weight,
        )
    return equiangular_mod.connect_camera(scene, path, wo, key)


def omnee(config: RenderConfig, scene: Scene, position: Array, direction: Array, key: Array) -> float:
    assert (config.path_length >= 4)
    key, subkey = random.split(key)
    path = path_mod.init_camera(scene, position, direction, subkey)
    key, subkey = random.split(key)
    path = path_mod.extend_n(scene, path, config.path_length - 4, subkey)
    return omnee_mod.connect(scene, path, key)


def omnee_lt(config: RenderConfig, scene: Scene, key: Array) -> tuple[float, float, float]:
    assert (config.path_length >= 3)
    if config.path_length == 3:
        eval_phase = False
        path = PathState(
            wi=None,
            position=scene.light.position,
            contribution=scene.light.intensity,
        )
    else:
        eval_phase = True
        key, subkey = random.split(key)
        path = path_mod.init_light(scene, subkey)
        key, subkey = random.split(key)
        path = path_mod.extend_n(scene, path, config.path_length - 4, subkey)
    return omnee_mod.connect_camera(scene, path, key, eval_phase=eval_phase)


def bridge(config: RenderConfig, scene: Scene, position: Array, direction: Array, key: Array) -> float:
    assert (config.path_length >= 4)
    assert (config.num_scatter_verts >= 1)
    assert (config.path_length >= 3 + config.num_scatter_verts)
    assert (config.num_scatter_verts <= bridge_mod.max_num_vertices)
    key, subkey = random.split(key)
    path = path_mod.init_camera(scene, position, direction, subkey)
    key, subkey = random.split(key)
    path = path_mod.extend_n(
        scene, path, config.path_length - 3 - config.num_scatter_verts, subkey)
    return bridge_mod.connect(scene, path, config.num_scatter_verts, key)


def bridge_lt(config: RenderConfig, scene: Scene, key: Array) -> tuple[float, float, float]:
    assert (config.path_length >= 3)
    assert (config.num_scatter_verts >= 1)
    assert (config.path_length >= 2 + config.num_scatter_verts)
    assert (config.num_scatter_verts <= bridge_mod.max_num_vertices)
    if config.path_length == 2 + config.num_scatter_verts:
        eval_phase = False
        path = PathState(
            wi=None,
            position=scene.light.position,
            contribution=scene.light.intensity,
        )
    else:
        eval_phase = True
        key, subkey = random.split(key)
        path = path_mod.init_light(scene, subkey)
        key, subkey = random.split(key)
        path = path_mod.extend_n(
            scene, path, config.path_length - 3 - config.num_scatter_verts, subkey)
    return bridge_mod.connect_camera(scene, path, config.num_scatter_verts, key, eval_phase=eval_phase)


def load_table_m1():
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 'path-length.npz'))
    tab = np.load(path)
    return tab['mean_cosines'][0], tab['mean_cosines'][-1], jnp.array(tab['params'])


def load_table_m2():
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 'path-length-m2.npz'))
    tab = np.load(path)
    return tab['mean_cosines'][0], tab['mean_cosines'][-1], jnp.array(tab['params'])


def make_bridge_lt_sampled_len(table=None, control_scale=1.0):
    # if no table, use poisson
    def bridge_lt_sampled_len(config: RenderConfig, scene: Scene, key: Array) -> tuple[float, float, float]:
        # ignores config.num_scatter_verts (always uses the maximum)
        # assert(max_length >= 2)
        max_k = config.path_length - 2
        # assert(max_k <= max_num_vertices)
        key, subkey = random.split(key)
        dist = jnp.linalg.norm(scene.light.position)
        key, subkey = random.split(key)
        if table is None:
            num_scatter_verts, weight = bridge_mod.sample_num_scatter_vertices_poisson(
                scene, dist, control_scale, max_k, subkey)
        else:
            num_scatter_verts, weight = bridge_mod.sample_num_scatter_vertices_table(
                scene, dist, table, max_k, subkey)

        def contribution():
            path = PathState(
                wi=None,
                position=scene.light.position,
                contribution=scene.light.intensity,
            )
            x, y, contrib = bridge_mod.connect_camera(
                scene, path, num_scatter_verts, key, eval_phase=False)
            return x, y, contrib * weight
        return lax.cond(num_scatter_verts <= max_k, contribution, discard_splat)
    return bridge_lt_sampled_len


def mvnee(config: RenderConfig, scene: Scene, position: Array, direction: Array, key: Array) -> float:
    assert (config.path_length >= 4)
    assert (config.num_scatter_verts >= 1)
    assert (config.path_length >= 3 + config.num_scatter_verts)
    assert (config.num_scatter_verts <= mvnee_mod.max_num_vertices)
    key, subkey = random.split(key)
    path = path_mod.init_camera(scene, position, direction, subkey)
    key, subkey = random.split(key)
    path = path_mod.extend_n(
        scene, path, config.path_length - 3 - config.num_scatter_verts, subkey)
    return mvnee_mod.connect(scene, path, config.num_scatter_verts, key)


def mvnee_lt(config: RenderConfig, scene: Scene, key: Array) -> tuple[float, float, float]:
    assert (config.path_length >= 3)
    assert (config.num_scatter_verts >= 1)
    assert (config.path_length >= 2 + config.num_scatter_verts)
    assert (config.num_scatter_verts <= mvnee_mod.max_num_vertices)
    if config.path_length == 2 + config.num_scatter_verts:
        eval_phase = False
        path = PathState(
            wi=None,
            position=scene.light.position,
            contribution=scene.light.intensity,
        )
    else:
        eval_phase = True
        key, subkey = random.split(key)
        path = path_mod.init_light(scene, subkey)
        key, subkey = random.split(key)
        path = path_mod.extend_n(
            scene, path, config.path_length - 3 - config.num_scatter_verts, subkey)
    return mvnee_mod.connect_camera(scene, path, config.num_scatter_verts, key, eval_phase=eval_phase)


def discard_splat() -> tuple[float, float, float]:
    return (10.0, 10.0, 0.0)

def bridge_lt_2(config: RenderConfig, scene: Scene, key: Array) -> tuple[float, float, float]:
    assert (config.path_length == 3)
    assert (config.num_scatter_verts == 1)
    path = PathState(
        wi=None,
        position=scene.light.position,
        contribution=scene.light.intensity,
    )
    return bridge_mod.connect_camera_constrained2(scene, path, key, eval_phase=False)

def bridge_lt_3(config: RenderConfig, scene: Scene, key: Array) -> tuple[float, float, float]:
    assert (config.path_length == 4)
    assert (config.num_scatter_verts == 2)
    path = PathState(
        wi=None,
        position=scene.light.position,
        contribution=scene.light.intensity,
    )
    return bridge_mod.connect_camera_constrained3(scene, path, key, eval_phase=False)