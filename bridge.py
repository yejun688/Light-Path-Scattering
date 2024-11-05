import functools
from .imports import *

from . import phase_func, medium, nee
from .util import *

max_num_vertices = 32  # max number of inner (scatter) vertices


def match(sampled: ArrayLike, target: ArrayLike) -> tuple[Array, float]:
    '''
    This computes a matrix (rotation + scale) to transform `sampled` to `target`. Inputs need to be
    centered on the same origin.

    Returns: (transformation matrix, scale factor)
    '''
    norm_sampled = jnp.linalg.norm(sampled)
    norm_target = jnp.linalg.norm(target)
    scale = norm_target / norm_sampled
    v_s = sampled / norm_sampled
    v_t = target / norm_target
    axis = jnp.linalg.cross(v_s, v_t)  # rotation axis * sin(theta)
    cos_theta = jnp.dot(v_s, v_t)
    K = jnp.cross(jnp.eye(3), axis)  # cross product matrix
    R = jnp.eye(3) + K + (K @ K) / (1.0 + cos_theta)
    return scale * R, scale


class BridgeState(NamedTuple):
    vertices: Array
    weight: float
    wi: Array


class BridgeState2(NamedTuple):
    vertices: Array
    weight: float
    wi: Array
    total_dist: float


def connect(scene: Scene, path: PathState, num_scatter_vertices: int, key: Array) -> float:
    '''
    Samples a point on the light source and connects the path to this point over `num_scatter_vertices`
    additional vertices (must be at least 1, at most max_num_vertices).
    '''

    origin = path.position
    target = scene.light.position
    num_bridge_vertices = num_scatter_vertices + 1
    wi = normalized(origin - target)
    bridge_vertices, bridge_weight = _sample_bridge(
        scene, wi, num_bridge_vertices, key)
    M, _scale = match(bridge_vertices[num_scatter_vertices], target - origin)
    vertices = origin + bridge_vertices @ M.T

    contribution = path.contribution
    contribution *= bridge_weight
    wo_origin = normalized(vertices[0] - origin)
    contribution *= phase_func.eval(scene.phase_func, path.wi, wo_origin)

    total_dist = _segment_length_sum(origin, vertices, num_bridge_vertices)

    contribution *= scene.light.intensity
    contribution /= jnp.linalg.norm(target - origin)**3
    logterm = -jax.scipy.special.gammaln(num_scatter_vertices + 1)
    logterm += num_bridge_vertices * jnp.log(total_dist)
    logterm -= scene.medium.mu_t * total_dist
    logterm += num_scatter_vertices * jnp.log(scene.medium.mu_s)
    contribution *= jnp.exp(logterm)


def connect_camera(scene: Scene, path: PathState, num_scatter_vertices: int, key: Array, eval_phase: bool) -> tuple[float, float, float]:
    origin = path.position
    target = jnp.zeros(3)
    num_bridge_vertices = num_scatter_vertices + 1
    wi = normalized(origin - target)
    bridge_vertices, bridge_weight, bridge_dist = _sample_bridge_with_dist(
        scene, wi, num_bridge_vertices, key)
    M, scale = match(bridge_vertices[num_scatter_vertices], target - origin)
    # vertices = origin + bridge_vertices @ M.T
    contribution = path.contribution * bridge_weight

    if eval_phase:
        wo = normalized(M @ bridge_vertices[0])
        contribution *= phase_func.eval(scene.phase_func, path.wi, wo)

    # total_dist = _segment_length_sum(origin, vertices, num_bridge_vertices)
    total_dist = bridge_dist * scale

    contribution /= jnp.linalg.norm(target - origin)**3
    logterm = -jax.scipy.special.gammaln(num_scatter_vertices + 1)
    logterm += num_bridge_vertices * jnp.log(total_dist)
    logterm -= scene.medium.mu_t * total_dist
    logterm += num_scatter_vertices * jnp.log(scene.medium.mu_s)
    contribution *= jnp.exp(logterm)

    return nee.splat(scene.cam, PathState(
        wi=None,
        position=origin + M @ bridge_vertices[num_scatter_vertices - 1],
        contribution=contribution
    ))


def sample_num_scatter_vertices_poisson(scene: Scene, dist: float, control_scale: float, max_num: int, key: Array) -> tuple[int, float]:
    lambda_ = scene.medium.mu_t * dist * control_scale
    num_verts, prob = poisson_sample_min_1(lambda_, max_num, key)
    return num_verts, (1.0 / prob)


def _curve_eval(param, x):
    num = 5
    first, center, last = param[0], param[1], param[2]

    def outside():
        linear = (x / first) * param[3]
        return lax.select(x <= first, linear, 0.0)

    def interpolate():
        def left():
            step = (center - first) / (num - 1)
            subidx = ((x - first) / step).astype(int)
            x0 = first + subidx * step
            p0_idx = 3 + 2 * subidx
            return x0, p0_idx, step

        def right():
            step = (last - center) / (num - 1)
            subidx = ((x - center) / step).astype(int)
            x0 = center + subidx * step
            p0_idx = 3 + 2 * (num - 1 + subidx)
            return x0, p0_idx, step
        x0, p0_idx, step = lax.cond(x < center, left, right)
        x1 = x0 + step
        y0 = param[p0_idx]
        dydx0 = param[p0_idx + 1]
        y1 = param[p0_idx + 2]
        dydx1 = param[p0_idx + 3]
        t = jnp.clip((x - x0) / (x1 - x0), 0.0, 1.0)
        t2 = t*t
        t3 = t2*t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        return h00 * y0 + h10 * (x1 - x0) * dydx0 + h01 * y1 + h11 * (x1 - x0) * dydx1

    return lax.cond((x > first) & (x < last), interpolate, outside)


def sample_num_scatter_vertices_table(scene: Scene, dist: float, table, max_num: int, key: Array) -> tuple[int, float]:
    g_min, g_max, params = table
    g = scene.phase_func.param.g
    last_g_idx = params.shape[0] - 1
    g_idx_real = (g - g_min) / (g_max - g_min) * params.shape[0]
    g_idx = jnp.clip(g_idx_real.astype(int), 0, last_g_idx)
    t_g = g_idx_real - g_idx
    g_row0 = params[g_idx]
    g_row1 = params[jnp.minimum(g_idx, last_g_idx)]
    x = dist * scene.medium.mu_t

    def fill_prob(idx: int, probs):
        idx = idx.astype(int)
        val0 = _curve_eval(g_row0[idx], x)
        val1 = _curve_eval(g_row1[idx], x)
        val = jnp.maximum(0.0, (1.0 - t_g) * val0 + t_g * val1)
        k = idx + 1
        val *= medium.albedo(scene.medium)**k
        return probs.at[idx].set(val)
    probs = lax.fori_loop(0, max_num, fill_prob, jnp.zeros(max_num_vertices))
    probs /= jnp.sum(probs)
    k = 1 + random.choice(key, len(probs), p=probs)
    return k, 1.0/probs[k - 1]


def poisson_probs_table(lambda_, max_k):
    def fill_prob(idx, probs):
        k = idx + 1.0
        logprob = k * jnp.log(lambda_) - lambda_ - \
            jax.scipy.special.gammaln(k + 1.0)
        return probs.at[idx].set(jnp.exp(logprob))
    probs = lax.fori_loop(0, max_k, fill_prob, jnp.zeros(max_num_vertices))
    probs /= jnp.sum(probs)
    return probs


def poisson_sample_min_1(lambda_, max_k, key):
    """Sample 1 <= k <= max_k from a Poisson distribution. Returns max_k + 1 if we sampled beyond the allowed range"""
    probs = poisson_probs_table(lambda_, max_k)
    k = 1 + random.choice(key, len(probs), p=probs)
    return k, probs[k - 1]


def _sample_bridge(scene, wi, num_bridge_vertices, key):
    # first vertex is x_1, last will be x_n (n = num_bridge_vertices)
    init_vertices = jnp.zeros((max_num_vertices + 1, 3))
    rand = random.uniform(key, (max_num_vertices + 1, 3))

    def extend_bridge(idx: int, bridge: BridgeState):
        wo_phase, weight_phase = phase_func.sample(
            scene.phase_func, bridge.wi, rand[idx, 0:2])
        prev = lax.select(idx == 0, jnp.zeros(3), bridge.vertices[idx - 1])
        wo = lax.select(idx == 0, -wi, wo_phase)
        weight = lax.select(idx == 0, 1.0, weight_phase)
        distance = -jnp.log(1.0 - rand[idx, 2])
        vertices = bridge.vertices.at[idx].set(prev + distance * wo)
        return BridgeState(
            vertices=vertices,
            weight=weight * bridge.weight,
            wi=-wo,
        )
    bridge = lax.fori_loop(0, num_bridge_vertices, extend_bridge, BridgeState(
        init_vertices, 1.0, jnp.zeros(3)))
    return bridge.vertices, bridge.weight


def _sample_bridge_with_dist(scene, wi, num_bridge_vertices, key):
    # first vertex is x_1, last will be x_n (n = num_bridge_vertices)
    init_vertices = jnp.zeros((max_num_vertices + 1, 3))
    rand = random.uniform(key, (max_num_vertices + 1, 3))

    def extend_bridge(idx: int, bridge: BridgeState2):
        wo_phase, weight_phase = phase_func.sample(
            scene.phase_func, bridge.wi, rand[idx, 0:2])
        prev = lax.select(idx == 0, jnp.zeros(3), bridge.vertices[idx - 1])
        wo = lax.select(idx == 0, -wi, wo_phase)
        weight = lax.select(idx == 0, 1.0, weight_phase)
        distance = -jnp.log(1.0 - rand[idx, 2])
        vertices = bridge.vertices.at[idx].set(prev + distance * wo)
        return BridgeState2(
            vertices=vertices,
            weight=weight * bridge.weight,
            wi=-wo,
            total_dist=bridge.total_dist + distance,
        )

    bridge = lax.fori_loop(0, num_bridge_vertices,
                           extend_bridge, BridgeState2(init_vertices, 1.0, jnp.zeros(3), 0.0))
    return bridge.vertices, bridge.weight, bridge.total_dist


def _segment_length_sum(origin, vertices, num_bridge_vertices):
    def segment_dist(idx: int, prefix_sum: float):
        vert_from = lax.select(idx == 0, origin, vertices[idx - 1])
        vert_to = vertices[idx]
        this_dist = jnp.linalg.norm(vert_to - vert_from)
        return prefix_sum + this_dist
    return lax.fori_loop(0, num_bridge_vertices, segment_dist, 0.0)


def slerp(p0, p1, t):
    theta = jnp.arccos(jnp.dot(p0, p1))
    sin_theta = jnp.sin(theta)

    def regular():
        return (jnp.sin((1.0 - t) * theta) * p0 + jnp.sin(t * theta) * p1) / sin_theta

    def fallback():
        p = (1.0 - t) * p0 + t * p1
        return p / jnp.linalg.norm(p)
    return lax.cond(sin_theta >= 1e-4, regular, fallback), 1.0 / theta


def connect_camera_constrained2(scene: Scene, path: PathState, key: Array, eval_phase: bool) -> tuple[float, float, float]:
    origin = path.position
    target = jnp.zeros(3)
    w1 = normalized(target - origin)
    xi = random.uniform(key, (3,))
    w2, weight_phase = phase_func.sample(scene.phase_func, -w1, xi[:2])
    O = jnp.column_stack((w1, w2))
    # actually we do know the analytical solution
    _U, S, Vh = jnp.linalg.svd(O)

    def contribution():
        n0 = normalized(Vh.T[0] / S)
        n1 = normalized(Vh.T[1] / S)
        rot90 = jnp.array([[0.0, -1.0], [1.0, 0.0]])
        l0 = rot90 @ n0
        l0 = lax.select(jnp.dot(l0, n1) >= 0.0, l0, -l0)
        l1 = rot90 @ n1
        l1 = lax.select(jnp.dot(l1, n0) >= 0.0, l1, -l1)
        t = xi[2]
        s = jnp.linalg.norm(target - origin)
        x, slerp_pdf = slerp(l0, l1, t)
        ds = Vh.T @ (s * x / S)
        M, _scale = match(O @ ds, target - origin)

        contribution = path.contribution * weight_phase
        if eval_phase:
            wo = M @ w1
            contribution *= phase_func.eval(scene.phase_func, path.wi, wo)
        total_dist = jnp.sum(ds)
        contribution *= medium.eval_transmittance(
            scene.medium, total_dist) * scene.medium.mu_s
        # would need to divide slerp_pdf by s, cancels with s^2 term from pdf
        contribution /= s * (S[0] * S[1]) * slerp_pdf
        v1 = origin + ds[0] * (M @ w1)
        return nee.splat(scene.cam, PathState(
            wi=None,
            position=v1,
            contribution=contribution
        ))
    return lax.cond(S[0] * S[1] > 1e-6, contribution, lambda: (10.0, 10.0, 0.0))


def sample_solid_angle_triangle(v0, v1, v2, xi0, xi1):
    G0 = jnp.abs(jnp.linalg.det(jnp.array([v0, v1, v2])))
    G1 = jnp.dot(v0, v2) + jnp.dot(v1, v2)
    G2 = 1.0 + jnp.dot(v0, v1)
    A_tri = 2.0 * jnp.arctan2(G0, G2 + G1)
    A = xi0 * A_tri
    r = (G0 * jnp.cos(A * 0.5) - G1 * jnp.sin(A * 0.5)) * \
        v0 + G2 * jnp.sin(A * 0.5) * v2
    v2_prime = -v0 + 2.0 * jnp.dot(v0, r) / jnp.dot(r, r) * r
    s2_prime = jnp.dot(v1, v2_prime)
    s = (1.0 - xi1) + xi1 * s2_prime
    t_prime = jnp.sqrt((1.0 - s**2) / (1.0 - s2_prime**2))
    omega_i = (s - t_prime * s2_prime) * v1 + t_prime * v2_prime
    return omega_i, 1.0 / A_tri


def connect_camera_constrained3(scene: Scene, path: PathState, key: Array, eval_phase: bool) -> tuple[float, float, float]:
    origin = path.position
    target = jnp.zeros(3)
    w1 = normalized(target - origin)
    xi = random.uniform(key, (6,))
    w2, weight_phase = phase_func.sample(scene.phase_func, -w1, xi[:2])
    w3, weight_phase2 = phase_func.sample(scene.phase_func, -w2, xi[2:4])
    O = jnp.column_stack((w1, w2, w3))
    _U, S, Vh = jnp.linalg.svd(O)

    def contribution():
        n0 = normalized(Vh.T[0] / S)
        n1 = normalized(Vh.T[1] / S)
        n2 = normalized(Vh.T[2] / S)
        l0 = normalized(jnp.cross(n0, n1))
        l0 = lax.select(jnp.dot(l0, n2) >= 0.0, l0, -l0)
        l1 = normalized(jnp.cross(n1, n2))
        l1 = lax.select(jnp.dot(l1, n0) >= 0.0, l1, -l1)
        l2 = normalized(jnp.cross(n0, n2))
        l2 = lax.select(jnp.dot(l2, n1) >= 0.0, l2, -l2)

        x, tri_pdf = sample_solid_angle_triangle(l0, l1, l2, xi[4], xi[5])
        s = jnp.linalg.norm(target - origin)
        ds = Vh.T @ (s * x / S)
        M, _scale = match(O @ ds, target - origin)

        contribution = path.contribution * weight_phase * weight_phase2
        if eval_phase:
            wo = M @ w1
            contribution *= phase_func.eval(scene.phase_func, path.wi, wo)
        total_dist = jnp.sum(ds)
        contribution *= medium.eval_transmittance(scene.medium, total_dist)
        contribution *= scene.medium.mu_s**2
        # s^2 term cancels with tri pdf scaling
        contribution /= (S[0] * S[1] * S[2]) * tri_pdf
        v1 = origin + ds[0] * (M @ w1) + ds[1] * (M @ w2)
        return nee.splat(scene.cam, PathState(
            wi=None,
            position=v1,
            contribution=contribution
        ))
    return lax.cond((S[0] * S[1] * S[2]) > 1e-6, contribution, lambda: (10.0, 10.0, 0.0))
