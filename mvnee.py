from .imports import *
from .util import *
from . import medium, phase_func, nee


def sample_ggx_2d(alpha, xi):
    phi = jnp.pi * 2.0 * xi[1]
    r = alpha * jnp.sqrt(xi[0] / (1.0 - xi[0]))
    return r * jnp.array([jnp.cos(phi), jnp.sin(phi)])


def pdf_ggx_2d(alpha, x):
    alpha2 = alpha**2
    r2 = jnp.dot(x, x)
    return alpha2 / (jnp.pi * (alpha2 + r2)**2)


max_num_vertices = 32  # max number of inner (scatter) vertices

def connect(scene: Scene, path: PathState, num_scatter_vertices: int, key: Array) -> float:
    origin = path.position
    target = scene.light.position

    total_dist = jnp.linalg.norm(target - origin)
    key, subkey = random.split(key)
    dists, pdf_dist = _sample_dists(
        scene, total_dist, num_scatter_vertices, subkey)
    g = scene.phase_func.param.g
    sigma_phi = 1.08036502 + g * \
        (-1.20611787 + g * (-0.12860913 + g * 0.2640833))  # close fit to table 1
    key, subkey = random.split(key)
    verts, pdf_vert = _sample_verts(origin, target,
                                    sigma_phi, dists, num_scatter_vertices, subkey)

    contribution = path.contribution
    contribution *= _measurement_contribution(scene,
                                              origin, target, verts, num_scatter_vertices)
    contribution /= pdf_dist
    contribution /= pdf_vert
    wo = normalized(verts[0] - origin)
    contribution *= phase_func.eval(scene.phase_func, path.wi, wo)
    contribution *= scene.light.intensity
    return contribution

def connect_camera(scene: Scene, path: PathState, num_scatter_vertices: int, key: Array, eval_phase: bool) -> tuple[float, float, float]:
    origin = path.position
    target = jnp.zeros(3)
    total_dist = jnp.linalg.norm(target - origin)

    key, subkey = random.split(key)
    dists, pdf_dist = _sample_dists(
        scene, total_dist, num_scatter_vertices, subkey)
    g = scene.phase_func.param.g
    sigma_phi = 1.08036502 + g * \
        (-1.20611787 + g * (-0.12860913 + g * 0.2640833))  # close fit to table 1
    key, subkey = random.split(key)
    verts, pdf_vert = _sample_verts(origin, target,
                                    sigma_phi, dists, num_scatter_vertices, subkey)

    contribution = path.contribution
    contribution *= _measurement_contribution(scene,
                                              origin, target, verts, num_scatter_vertices)
    contribution /= pdf_dist
    contribution /= pdf_vert
    if eval_phase:
        wo = normalized(verts[0] - origin)
        contribution *= phase_func.eval(scene.phase_func, path.wi, wo)

    return nee.splat(scene.cam, PathState(
        wi=None,
        position=verts[num_scatter_vertices - 1],
        contribution=contribution
    ))


class DistState(NamedTuple):
    dists: Array
    remaining_dist: float
    pdf: float


def _sample_dists(scene: Scene, total_dist, num_scatter_vertices, key) -> tuple[Array, float]:
    max_num_dists = max_num_vertices + 1
    num_dists = num_scatter_vertices + 1
    xi = random.uniform(key, (max_num_dists,))

    def sample_next_dist(idx, state: DistState):
        dist, _w = medium.sample_free_path_before(
            scene.medium, state.remaining_dist, xi[idx])
        pdf = medium.pdf_free_path_before(
            scene.medium, state.remaining_dist, dist)
        dists = state.dists.at[idx].set(dist)
        return DistState(
            dists=dists,
            remaining_dist=jnp.maximum(0.0, state.remaining_dist - dist),
            pdf=state.pdf * pdf,
        )
    init = DistState(
        dists=jnp.zeros(max_num_dists),
        remaining_dist=total_dist,
        pdf=1.0,
    )
    result = lax.fori_loop(0, num_scatter_vertices, sample_next_dist, init)
    dists = result.dists.at[num_dists - 1].set(result.remaining_dist)
    return dists, result.pdf


class VertsState(NamedTuple):
    dist2_l: float
    dist2_r: float
    dist: float
    pdf: float
    verts: Array


def _sample_verts(origin, target, sigma_phi, dists, num_scatter_vertices, key) -> tuple[Array, float]:
    z = normalized(target - origin)
    x, y = coordinate_system(z)

    xi = random.uniform(key, (max_num_vertices, 2))
    dists2 = dists**2

    def sample_next_vert(idx, state: VertsState):
        dist2_l = state.dist2_l + dists2[idx]
        dist2_r = jnp.maximum(0.0, state.dist2_r - dists2[idx])
        sigma2_l = sigma_phi**2 * dist2_l
        sigma2_r = sigma_phi**2 * dist2_r
        sigma = jnp.sqrt(1.0 / (1.0/sigma2_l + 1.0/sigma2_r))
        cGGX = 1.637618734
        alpha = cGGX * sigma
        perturb_xy = sample_ggx_2d(alpha, xi[idx])
        perturb = perturb_xy[0] * x + perturb_xy[1] * y
        dist = state.dist + dists[idx]
        v = origin + z * dist + perturb
        pdf = pdf_ggx_2d(alpha, perturb_xy)
        verts = state.verts.at[idx].set(v)
        return VertsState(
            dist2_l=dist2_l,
            dist2_r=dist2_r,
            dist=dist,
            pdf=state.pdf * pdf,
            verts=verts,
        )
    init = VertsState(
        dist2_l=0,
        dist2_r=jnp.sum(dists2),
        dist=0,
        pdf=1.0,
        verts=jnp.zeros((max_num_vertices, 3))
    )
    result = lax.fori_loop(0, num_scatter_vertices, sample_next_vert, init)
    return result.verts, result.pdf


class McfState(NamedTuple):
    contribution: float
    total_dist: float


def _measurement_contribution(scene: Scene, origin, target, vertices, num_scatter_vertices) -> float:
    def eval_next_segment(idx, state: McfState):
        prev = lax.select(idx == 0, origin, vertices[idx - 1])
        curr = vertices[idx]
        next = lax.select(idx + 1 == num_scatter_vertices,
                          target, vertices[idx + 1])
        phase = phase_func.eval(scene.phase_func, normalized(
            prev - curr), normalized(next - curr))
        dist = jnp.linalg.norm(curr - prev)
        geo = 1.0 / dist**2
        return McfState(
            contribution=state.contribution * phase * geo,
            total_dist=state.total_dist + dist,
        )
    init = McfState(
        contribution=1.0,
        total_dist=0.0,
    )
    result = lax.fori_loop(0, num_scatter_vertices, eval_next_segment, init)
    contribution = result.contribution
    contribution *= scene.medium.mu_s**num_scatter_vertices
    last_vert = vertices[num_scatter_vertices - 1]
    total_dist = result.total_dist + jnp.linalg.norm(last_vert - target)
    contribution *= medium.eval_transmittance(scene.medium, total_dist)
    contribution *= 1.0 / jnp.dot(last_vert - target, last_vert - target)
    return contribution
