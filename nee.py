from .imports import *
from . import phase_func, equiangular, medium
from .util import *


def connect(scene: Scene, path: PathState, key: Array) -> float:
    wo = normalized(scene.light.position - path.position)
    distance = jnp.linalg.norm(scene.light.position - path.position)
    contribution = path.contribution
    contribution *= phase_func.eval(scene.phase_func,
                                    path.wi, wo)
    contribution *= medium.eval_transmittance(scene.medium, distance)
    contribution *= scene.light.intensity
    contribution /= distance**2
    return contribution


def connect_camera(scene: Scene, path: PathState, key: Array) -> tuple[float, float, float]:
    contribution = path.contribution
    wo = normalized(-path.position)
    dist = jnp.linalg.norm(path.position)
    contribution *= phase_func.eval(scene.phase_func, path.wi, wo)
    contribution *= 1.0/dist**2
    contribution *= medium.eval_transmittance(scene.medium, dist)
    return splat(scene.cam, PathState(
        wi=path.wi,
        position=path.position,
        contribution=contribution,
    ))


# only do the very last step of light tracing (for techniques that sample the camera connection themselves)
def splat(cam: Camera, path: PathState) -> tuple[float, float, float]:
    # camera is assumed to be at (0, 0, 0) and looking into direction (0, 0, -1)
    screen_pos = path.position[0:2] / path.position[2]
    cam_dir = normalized(path.position)
    dot4 = cam_dir[2]**4
    A = 0.5135 * 0.5135 * cam.aspect
    contribution = path.contribution / (dot4 * A)
    return screen_pos[0], screen_pos[1], contribution
