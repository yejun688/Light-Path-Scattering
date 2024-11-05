from jax_tqdm import loop_tqdm
from .util import *


def render_frame(sample_func):
    def inner(config: RenderConfig, scene: Scene, key: Array) -> Array:
        def render_pixel(idx, key):
            x = idx % config.width
            y = idx // config.width
            key, subkey = random.split(key)
            off = random.uniform(subkey, (2,))
            pos, dir = _camera_ray(
                config, scene, x + off[0], y + off[1])
            return sample_func(config, scene, pos, dir, key)
        num = config.width * config.height
        keys = random.split(key, (num,))
        return jax.vmap(render_pixel, in_axes=(0, 0))(jnp.arange(0, num), keys).reshape((config.height, config.width))
    return jax.jit(inner, static_argnames=['config'])


def _camera_ray(config: RenderConfig, scene: Scene, x: int, y: int):
    s = x / config.width
    t = y / config.height
    aspect = config.width / config.height
    sensor = 0.5135
    pos = jnp.zeros(3)
    dir = normalized(jnp.array([
        sensor * aspect * (-0.5 * (1.0 - s) + 0.5 * s),
        sensor * (-0.5 * (1.0 - t) + 0.5 * t),
        -1.0
    ]))
    return pos, dir


def _camera_splat(config: RenderConfig, x, y, weight):
    aspect = config.width / config.height
    sensor = 0.5135
    camx = sensor * aspect * 0.5
    camy = sensor * 0.5
    hist, _xedges, _yedges = jnp.histogram2d(y, x, bins=(
        config.height, config.width), range=((-camy, camy), (-camx, camx)), weights=weight)
    return hist


def render_frame_lt(sample_func):
    def inner(config: RenderConfig, scene: Scene, key: Array) -> Array:
        def render_pixel(key):
            return sample_func(config, scene, key)
        num = config.width * config.height
        keys = random.split(key, (num,))
        x, y, weight = jax.vmap(render_pixel)(keys)
        return _camera_splat(config, x, y, weight)
    return jax.jit(inner, static_argnames=['config'])


def render_frames(sample_func):
    def inner(config: RenderConfig, scene: Scene, key: Array, num_samples: int) -> Array:
        rf = render_frame(sample_func)

        @loop_tqdm(num_samples)
        def loop(_idx, state):
            acc, key = state
            key, subkey = random.split(key)
            this_frame = rf(config, scene, subkey)
            if config.filter_nan:
                newacc = acc + \
                    jnp.nan_to_num(this_frame, posinf=0.0)
            else:
                newacc = acc + this_frame
            return (newacc, key)
        z = jnp.zeros((config.height, config.width))
        acc, _ = lax.fori_loop(0, num_samples, loop, (z, key))
        return acc / num_samples
    return jax.jit(inner, static_argnames=['config', 'num_samples'])


def render_frames_lt(sample_func):
    def inner(config: RenderConfig, scene: Scene, key: Array, num_samples: int) -> Array:
        rf = render_frame_lt(sample_func)

        @loop_tqdm(num_samples)
        def loop(_idx, state):
            acc, key = state
            key, subkey = random.split(key)
            this_frame = rf(config, scene, subkey)
            if config.filter_nan:
                newacc = acc + \
                    jnp.nan_to_num(this_frame, posinf=0.0)
            else:
                newacc = acc + this_frame
            return (newacc, key)
        z = jnp.zeros((config.height, config.width))
        acc, _ = lax.fori_loop(0, num_samples, loop, (z, key))
        return acc / num_samples
    return jax.jit(inner, static_argnames=['config', 'num_samples'])
