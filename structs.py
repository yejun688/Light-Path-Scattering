from enum import IntEnum
from typing import Any
from typing import NamedTuple
from jax import Array


class Ray(NamedTuple):
    origin: Array
    direction: Array


class Sphere(NamedTuple):
    position: Array
    radius: float


class PathState(NamedTuple):
    wi: Array  # last direction on path, points away from pos
    position: Array
    contribution: float  # this will already contain the \mu_s on the last vertex


class SphereLight(NamedTuple):
    sphere: Sphere
    radiance: float


class PointLight(NamedTuple):
    position: Array
    intensity: float


class ScatterHemisphere(IntEnum):
    FWD = 0  # only forward scattering
    BWD = 1  # only backward scattering
    ALL = 2  # both forward and backward scattering


class PhaseFamily(IntEnum):
    HG = 0  # henyey - greenstein
    MIE = 1  # mie approximation


class HgParam(NamedTuple):
    g: float  # mean cosine


class PhaseFunction(NamedTuple):
    scatter_hemi: ScatterHemisphere
    phase_family: PhaseFamily
    param: Any  # different for each phase function family


class Medium(NamedTuple):
    mu_a: float  # absorbption coefficient
    mu_s: float  # scattering coefficient
    mu_t: float  # extinction coefficient: mu_a + mu_s


class Camera(NamedTuple):
    aspect: float  # width / height


class Scene(NamedTuple):
    medium: Medium
    # light: SphereLight
    light: PointLight
    phase_func: PhaseFunction
    cam: Camera


class RenderConfig(NamedTuple):
    width: int
    height: int
    filter_nan: bool
    path_length: int  # number of vertices (including light and camera)
    # number of extra inner vertices for samplers that connect via multiple vertices (e.g. mvnee, bridge)
    num_scatter_verts: int


def make_medium(albedo: float, mu_t: float) -> Medium:
    return Medium(
        mu_a=(1.0 - albedo) * mu_t,
        mu_s=albedo * mu_t,
        mu_t=mu_t
    )


def make_camera(render_config: RenderConfig) -> Camera:
    return Camera(
        aspect=render_config.width / render_config.height
    )


def make_hg(mean_cos: float, scatter_hemi: ScatterHemisphere) -> PhaseFunction:
    return PhaseFunction(
        scatter_hemi=scatter_hemi,
        phase_family=PhaseFamily.HG,
        param=HgParam(g=mean_cos)
    )


def make_config(width, height, path_length) -> RenderConfig:
    return RenderConfig(
        width=width,
        height=height,
        filter_nan=True,
        path_length=path_length,
        num_scatter_verts=path_length - 2,
    )
