#!/usr/bin/env python

from typing import Callable, Optional, Tuple, Union, Final

from numba import njit, prange
import numpy as np
from opensimplex import OpenSimplex
# HACK
from opensimplex.internals import _noise2, _noise3, _noise4
import pyfastnoisesimd as fns

from utils.numeric import rescale, rescale_in_place, max_abs


PI: Final = np.pi
TAU: Final = 2.0*np.pi

USE_FNS_DEFAULT = True


@njit(cache=True, parallel=True)
def _noise2v(x: np.ndarray, y: np.ndarray, perm) -> np.ndarray:
	ret = np.empty(x.shape, dtype=np.double)
	for iy in prange(x.shape[0]):
		for ix in prange(x.shape[1]):
			x_coord = x[iy, ix]
			y_coord = y[iy, ix]
			ret[iy, ix] = _noise2(x_coord, y_coord, perm)
	return ret


@njit(cache=True, parallel=True)
def _noise3v(x: np.ndarray, y: np.ndarray, z: np.ndarray, perm, perm_grad_index3) -> np.ndarray:
	ret = np.empty(x.shape, dtype=np.double)
	for iy in prange(x.shape[0]):
		for ix in prange(x.shape[1]):
			x_coord = x[iy, ix]
			y_coord = y[iy, ix]
			z_coord = z[iy, ix]
			ret[iy, ix] = _noise3(x_coord, y_coord, z_coord, perm, perm_grad_index3)
	return ret


@njit(cache=True, parallel=True)
def _noise4v(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray, perm) -> np.ndarray:
	ret = np.empty(x.shape, dtype=np.double)
	for iy in prange(x.shape[0]):
		for ix in prange(x.shape[1]):
			x_coord = x[iy, ix]
			y_coord = y[iy, ix]
			z_coord = z[iy, ix]
			w_coord = w[iy, ix]
			ret[iy, ix] = _noise4(x_coord, y_coord, z_coord, w_coord, perm)
	return ret


def xy_grid(width: int, height: int):
	max_dim = max(width, height)
	x_max = width / max_dim
	y_max = height / max_dim
	x = np.linspace(0.0, x_max, num=width, endpoint=False)
	y = np.linspace(0.0, y_max, num=height, endpoint=False)
	return np.meshgrid(x, y)


# def sphere_coord(height: int, width: Optional[int]=None, radius=1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
def sphere_coord(height: int, width: Optional[int]=None, radius=1/TAU) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

	# TODO: should default radius be something else, equivalent to cylinder or double_wrap? like 1/PI or 1/TAU or 1/4pi
	# Unlike those, resulting coordinates will all be same scale, so it probably doesn't matter

	if width is None:
		width = 2*height

	latitude = np.linspace(-PI/2, PI/2, num=height, endpoint=False)
	longitude = np.linspace(-PI, PI, num=width, endpoint=False)
	longitude, latitude = np.meshgrid(longitude, latitude)

	x = radius * np.cos(latitude) * np.cos(longitude)
	y = radius * np.cos(latitude) * np.sin(longitude)
	z = radius * np.sin(latitude)

	return x, y, z


def cylinder_coord(width: int, height: int):
	# https://www.redblobgames.com/maps/terrain-from-noise/

	nx, ny = xy_grid(width=width, height=height)

	angle_x = TAU * nx
	# In "noise parameter space", we need nx and ny to travel the
	# same distance. The circle created from nx needs to have
	# circumference=1 to match the length=1 line created from ny,
	# which means the circle's radius is 1/2pi
	xy_radius = 1 / TAU
	x = xy_radius * np.cos(angle_x)
	y = xy_radius * np.sin(angle_x)
	z = ny
	return x, y, z


def double_wrap_coord(width: int, height: int, radius=1/TAU):
	# https://www.redblobgames.com/maps/terrain-from-noise/
	nx, ny = xy_grid(width=width, height=height)
	angle_x = TAU * nx
	angle_y = TAU * ny
	x = radius * np.cos(angle_x)
	y = radius * np.sin(angle_x)
	z = radius * np.cos(angle_y)
	w = radius * np.sin(angle_y)
	return x, y, z, w


class NoiseCoords:

	@classmethod
	def xy_grid(cls, width: int, height: int):
		return cls(*xy_grid(width=width, height=height))

	@classmethod
	def sphere_coord(cls, height: int, width: Optional[int]=None):
		return cls(*sphere_coord(height=height, width=width))

	@classmethod
	def cylinder_coord(cls, width: int, height: int):
		return cls(*cylinder_coord(width=width, height=height))

	@classmethod
	def double_wrap_coord(cls, width: int, height: int):
		return cls(*double_wrap_coord(width=width, height=height))

	def __init__(self, *coord: np.ndarray):
		self._coord: tuple[np.ndarray, ...] = coord
		self._fns_coord: Optional[np.ndarray] = None

		if not coord:
			raise ValueError('Must provide at least 1 dimension')

		if not all(c.shape == coord[0].shape for c in coord[1:]):
			raise ValueError('Shapes do not match')

		if len(coord) <= 3:
			self._fns_coord = fns.empty_coords(coord[0].size)
			self._fns_coord[0,:] = coord[0].reshape(-1)
			self._fns_coord[1,:] = coord[1].reshape(-1) if len(coord) >= 2 else 0
			self._fns_coord[2,:] = coord[2].reshape(-1) if len(coord) >= 3 else 0

	@property
	def dimension(self) -> int:
		return len(self._coord)

	@property
	def x(self) -> np.ndarray:
		return self._coord[0]

	@property
	def y(self) -> Optional[np.ndarray]:
		return self._coord[1] if len(self._coord) >= 2 else None

	@property
	def z(self) -> Optional[np.ndarray]:
		return self._coord[2] if len(self._coord) >= 3 else None

	@property
	def w(self) -> Optional[np.ndarray]:
		return self._coord[3] if len(self._coord) >= 4 else None

	@property
	def shape(self) -> tuple[int, ...]:
		return self._coord[0].shape

	@property
	def num_points(self) -> int:
		return self._coord[0].size

	@property
	def fns_coord(self) -> Optional[np.ndarray]:
		return self._fns_coord


class FractalNoise:
	def __init__(
			self,
			# Seed
			seed: int,
			# Generation params
			octaves: int,
			gain: float = 0.5,
			lacunarity: float = 2.0,
			base_frequency: float = 1.0,
			# TODO: more parameters for noise type and such (e.g. diff, turbulence, etc)
			use_fns: bool = USE_FNS_DEFAULT,
			):

		if octaves <= 0:
			raise ValueError(f'{octaves=}')

		self.seed = seed
		self.octaves = octaves
		self.gain = gain
		self.lacunarity = lacunarity
		self.base_frequency = base_frequency
		self.use_fns = use_fns

		self.open_simplex_objs: list[OpenSimplex] = []
		self.fns: Optional[fns.Noise] = None
		self.init_amplitude: Optional[float] = None

		if self.use_fns:
			self._init_fns()
		else:
			self._init_opensimplex()

	def _init_fns(self):
		if self.fns is not None:
			return

		# TODO: seed seems it gets randomized per Python run
		self.fns = fns.Noise(seed=np.int64(self.seed))

		self.fns.frequency = self.base_frequency
		self.fns.fractal.octaves = self.octaves
		self.fns.fractal.lacunarity = self.lacunarity
		self.fns.fractal.gain = self.gain
		self.fns.perturb.perturbType = fns.PerturbType.NoPerturb

		# TODO?
		# self.fns.noiseType = fns.NoiseType.SimplexFractal

	def _init_opensimplex(self):
		if self.open_simplex_objs:
			return

		# TODO: instead of separate objects, try using different location in space
		self.open_simplex_objs = [
			OpenSimplex(hash(hash(self.seed) + idx))  # OpenSimplex doesn't like seed 0, so hash self.seed too
			for idx in range(self.octaves)
		]

		# TODO: can probably calculate this non-iteratively
		amplitude_sum = 0.0
		amplitude = 1.0
		for _ in range(self.octaves):
			amplitude_sum += amplitude
			amplitude *= self.gain
		self.init_amplitude = 1.0 / amplitude_sum

	def _gen_fns(self, nc: NoiseCoords, /) -> np.ndarray:
		self._init_fns()
		assert self.fns is not None
		assert nc.fns_coord is not None
		noise = self.fns.genFromCoords(nc.fns_coord)
		return noise[:nc.num_points].reshape(nc.shape)

	def _single_noise_opensimplex(self, x: float, y: float = 0.0, z: Optional[float]=None, w: Optional[float]=None, /) -> float:
		self._init_opensimplex()
		assert self.open_simplex_objs

		ret = 0.0
		frequency = self.base_frequency
		amplitude = self.init_amplitude
		for open_simplex_obj in self.open_simplex_objs:
			if z is None:
				ret += open_simplex_obj.noise2(x * frequency, y * frequency) * amplitude
			elif w is None:
				ret += open_simplex_obj.noise3(x * frequency, y * frequency, z * frequency) * amplitude
			else:
				ret += open_simplex_obj.noise4(x * frequency, y * frequency, z * frequency, w * frequency) * amplitude
			frequency *= self.lacunarity
			amplitude *= self.gain
		return ret

	def _vectorized_noise_opensimplex(self, x: np.ndarray, y: Optional[np.ndarray]=None, z: Optional[np.ndarray]=None, w: Optional[np.ndarray]=None, /) -> np.ndarray:

		if not all(val is None or val.shape == x.shape for val in [y, z, w]):
			raise ValueError('Shapes do not match!')

		self._init_opensimplex()
		assert self.open_simplex_objs

		ret = np.zeros_like(x)
		frequency = self.base_frequency
		amplitude = self.init_amplitude
		for open_simplex_obj in self.open_simplex_objs:
			if y is None:
				ret += amplitude * _noise2v(
					x * frequency,
					np.zeros_like(x),
					open_simplex_obj._perm)
			elif z is None:
				ret += amplitude * _noise2v(
					x * frequency,
					y * frequency,
					open_simplex_obj._perm)
			elif w is None:
				ret += amplitude * _noise3v(
					x * frequency,
					y * frequency,
					z * frequency,
					open_simplex_obj._perm,
					open_simplex_obj._perm_grad_index3)
			else:
				ret += amplitude * _noise4v(
					x * frequency,
					y * frequency,
					z * frequency,
					w * frequency,
					open_simplex_obj._perm)
			frequency *= self.lacunarity
			amplitude *= self.gain
		return ret

	def noise(
			self, nc: NoiseCoords, /, *,
			normalize=False, normalize_amplitude=False,
			) -> np.ndarray:
		if self.use_fns and nc.dimension <= 3:
			ret = self._gen_fns(nc)
		else:
			ret = self._vectorized_noise_opensimplex(nc.x, nc.y, nc.z, nc.w)

		# TODO: smarter amplitudes, make normalization less necessary
		if normalize:
			rescale_in_place(ret, range_out=(-1, 1))
		elif normalize_amplitude:
			ret /= max_abs(ret)

		return ret

	# TODO
	# def valley_noise(self, x: float, y: float) -> float:
	# 	ret = 0.0
	# 	frequency = self.base_frequency
	# 	amplitude = self.init_amplitude
	# 	for open_simplex_obj in self.open_simplex_objs:
	# 		ret += np.abs(open_simplex_obj.noise2(x * frequency, y * frequency)) * amplitude
	# 		frequency *= self.lacunarity
	# 		amplitude *= self.gain
	# 	return ret

	def valley_noise_grid(
			self,
			x: np.ndarray,
			y: np.ndarray,
			z: Optional[np.ndarray]=None,
			w: Optional[np.ndarray]=None,
			/) -> np.ndarray:

		# TODO: use FNS
		self._init_opensimplex()

		ret = None
		frequency = self.base_frequency
		amplitude = self.init_amplitude
		for open_simplex_obj in self.open_simplex_objs:

			if z is None:
				layer = open_simplex_obj.noise2array(x * frequency, y * frequency)
			elif w is None:
				layer = open_simplex_obj.noise3array(x * frequency, y * frequency, z * frequency)
			else:
				layer = open_simplex_obj.noise4array(x * frequency, y * frequency, z * frequency, w * frequency)
			layer = np.abs(layer)
			layer *= amplitude
			if ret is None:
				ret = layer
			else:
				ret += layer

			frequency *= self.lacunarity
			amplitude *= self.gain

		assert ret is not None
		return ret

	# TODO: this doesn't need to be a special function, can just do (1.0 - valley)
	def ridge_noise_grid(
			self,
			x: np.ndarray,
			y: np.ndarray,
			z: Optional[np.ndarray]=None,
			w: Optional[np.ndarray]=None,
			/) -> np.ndarray:

		ret = None
		frequency = self.base_frequency
		amplitude = self.init_amplitude
		for open_simplex_obj in self.open_simplex_objs:

			if z is None:
				layer = open_simplex_obj.noise2array(x * frequency, y * frequency)
			elif w is None:
				layer = open_simplex_obj.noise3array(x * frequency, y * frequency, z * frequency)
			else:
				layer = open_simplex_obj.noise4array(x * frequency, y * frequency, z * frequency, w * frequency)
			layer = 1.0 - np.abs(layer)
			layer *= amplitude
			if ret is None:
				ret = layer
			else:
				ret += layer

			frequency *= self.lacunarity
			amplitude *= self.gain

		assert ret is not None
		return ret



# TODO: make these all member functions of FractalNoise


def fbm(
		seed: int,

		coord: Optional[NoiseCoords] = None,
		width: Optional[int] = None,
		height: Optional[int] = None,

		octaves: int = 8,
		gain: float = 0.5,
		lacunarity: float = 2.0,
		base_frequency: float = 1.0,
		normalize: bool = False,
		normalize_amplitude: bool = False,
		use_fns: bool = USE_FNS_DEFAULT,
		) -> np.ndarray:

	if coord is None:
		if width is None or height is None:
			raise ValueError('Must provide width & height if not providing coord')
		coord = NoiseCoords.xy_grid(width=width, height=height)

	fractal_noise = FractalNoise(
		seed=seed, octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)

	return fractal_noise.noise(coord, normalize=normalize, normalize_amplitude=normalize_amplitude)


def diff_fbm(
		seed: int,
		diff_steps: int,
		fbm_func: Optional[Callable[..., np.ndarray]] = None,
		normalize: bool = False,
		**kwargs,
		) -> np.ndarray:

	# coord = NoiseCoords.xy_grid(width=width, height=height)  # TODO: use this

	if diff_steps < 1:
		raise ValueError('diff_steps must be at least 1')

	if fbm_func is None:
		fbm_func = fbm

	img = fbm_func(seed=seed, **kwargs)

	for step_idx in range(diff_steps):
		new = fbm_func(seed=(seed + step_idx + 1), **kwargs)
		img = np.abs(img - new) - 0.5

	if normalize:
		rescale_in_place(img, range_out=(-1, 1))
	else:
		img *= 1.0 / np.log2(1 + diff_steps)
		img += 0.5

	return img


def domain_warped_fbm(
		seed: int,
		width: int,
		height: int,
		warp_steps: int,
		warp_amount: float = 1.0,
		octaves: int = 8,
		gain: float = 0.5,
		lacunarity: float = 2.0,
		base_frequency: float = 1.0,
		normalize: bool = False,
		normalize_amplitude: bool = False,
		use_fns: bool = USE_FNS_DEFAULT,
		) -> np.ndarray:

	# TODO: use FNS built-in domain warping

	x, y = xy_grid(width, height)

	for idx in range(warp_steps):

		fbm_x = FractalNoise(
			seed=(seed + 1 + 2*idx), octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)
		fbm_y = FractalNoise(
			seed=(seed + 2 + 2*idx), octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)

		# TODO: normalize these?

		nc = NoiseCoords(x, y)
		x_warp = fbm_x.noise(nc)
		y_warp = fbm_y.noise(nc)

		x += (x_warp * warp_amount)
		y += (y_warp * warp_amount)

	nc = NoiseCoords(x, y)
	fractal_noise = FractalNoise(
		seed=seed, octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency)
	img = fractal_noise.noise(nc)

	if normalize:
		rescale_in_place(img, range_out=(-1, 1))
	elif normalize_amplitude:
		img /= max_abs(img)

	return img


def wrapped_fbm(
		seed: int,
		width: int,
		height: int,
		wrap_x: bool = True,
		wrap_y: bool = False,
		octaves: int = 8,
		gain: float = 0.5,
		lacunarity: float = 2.0,
		base_frequency: float = 1.0,
		normalize: bool = False,
		normalize_amplitude: bool = False,
		use_fns: bool = USE_FNS_DEFAULT,
		) -> np.ndarray:

	transpose = False
	if wrap_x and wrap_y:
		nc = NoiseCoords.double_wrap_coord(width=width, height=height)
	elif wrap_x:
		nc = NoiseCoords.cylinder_coord(width=width, height=height)
	elif wrap_y:
		# TODO: make transpose unnecessary - calculate the right cylinder coordinates in the first place
		transpose = True
		nc = NoiseCoords.cylinder_coord(width=width, height=height)
	else:
		nc = NoiseCoords.xy_grid(width=width, height=height)

	fractal_noise = FractalNoise(
		seed=seed, octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)

	img = fractal_noise.noise(nc)

	if transpose:
		img = img.transpose()

	if normalize:
		rescale_in_place(img, range_out=(-1, 1))
	elif normalize_amplitude:
		img /= max_abs(img)

	return img


def sphere_fbm(
		seed: int,
		height: int,
		width: Optional[int] = None,
		octaves: int = 8,
		gain: float = 0.5,
		lacunarity: float = 2.0,
		base_frequency: float = 1.0,
		normalize: bool = False,
		normalize_amplitude: bool = False,
		use_fns: bool = USE_FNS_DEFAULT,
		) -> np.ndarray:

	coord = NoiseCoords.sphere_coord(height=height, width=width)

	fractal_noise = FractalNoise(
		seed=seed, octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)

	img = fractal_noise.noise(coord)

	if normalize:
		rescale_in_place(img, range_out=(-1, 1))
	elif normalize_amplitude:
		img /= max_abs(img)

	return img


def sphere_domain_warped_fbm(
		seed: int,
		width: int,
		height: int,
		warp_steps: int,
		warp_amount: float = 1.0,
		octaves: int = 8,
		gain: float = 0.5,
		lacunarity: float = 2.0,
		base_frequency: float = 1.0,
		normalize: bool = False,
		normalize_amplitude: bool = False,
		use_fns: bool = USE_FNS_DEFAULT,
		) -> np.ndarray:

	# TODO: use FNS built-in domain warping

	x, y, z = sphere_coord(height, width=width)

	for idx in range(warp_steps):

		fbm_x = FractalNoise(
			seed=(seed + 1 + 3*idx), octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)
		fbm_y = FractalNoise(
			seed=(seed + 2 + 3*idx), octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)
		fbm_z = FractalNoise(
			seed=(seed + 3 + 3*idx), octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)

		# TODO: normalize?

		nc = NoiseCoords(x, y, z)
		x_warp = fbm_x.noise(nc)
		y_warp = fbm_y.noise(nc)
		z_warp = fbm_z.noise(nc)

		# TODO: support math operations with NoiseCoord, instead of needing to keep track of separate x/y/z
		x += (x_warp * warp_amount)
		y += (y_warp * warp_amount)
		z += (z_warp * warp_amount)

	nc = NoiseCoords(x, y, z)

	fractal_noise = FractalNoise(
		seed=seed, octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency)
	img = fractal_noise.noise(nc)

	if normalize:
		rescale_in_place(img, range_out=(-1, 1))
	elif normalize_amplitude:
		img /= np.amax(np.abs(img))

	return img


def valley_fbm(
		seed: int,
		width: int,
		height: int,
		octaves: int = 8,
		gain: float = 0.5,
		lacunarity: float = 2.0,
		base_frequency: float = 1.0,
		normalize: bool = False,
		use_fns: bool = USE_FNS_DEFAULT,
		) -> np.ndarray:

	# coord = NoiseCoords.xy_grid(width=width, height=height)  # TODO: use this

	fractal_noise = FractalNoise(
		seed=seed, octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)

	# TODO: factor in aspect ratio
	x_max = 1.0
	y_max = 1.0
	x = np.linspace(0.0, x_max, num=width, endpoint=False)
	y = np.linspace(0.0, y_max, num=height, endpoint=False)
	img = fractal_noise.valley_noise_grid(x, y)

	if normalize:
		img /= np.amax(img)

	return img


def ridge_fbm(
		seed: int,
		width: int,
		height: int,
		octaves: int = 8,
		gain: float = 0.5,
		lacunarity: float = 2.0,
		base_frequency: float = 1.0,
		normalize: bool = False,
		use_fns: bool = USE_FNS_DEFAULT,
		) -> np.ndarray:

	# coord = NoiseCoords.xy_grid(width=width, height=height)  # TODO: use this

	fractal_noise = FractalNoise(
		seed=seed, octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)

	# TODO: factor in aspect ratio
	x_max = 1.0
	y_max = 1.0
	x = np.linspace(0.0, x_max, num=width, endpoint=False)
	y = np.linspace(0.0, y_max, num=height, endpoint=False)
	img = fractal_noise.ridge_noise_grid(x, y)

	if normalize:
		rescale_in_place(img)

	return img
