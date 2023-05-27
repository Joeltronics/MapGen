#!/usr/bin/env python

from enum import Enum, unique, auto
from typing import Callable, Optional, Tuple, Union, Final
import warnings

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


def _noisev(
		open_simplex_obj: OpenSimplex,
		x: np.ndarray,
		y: Optional[np.ndarray] = None,
		z: Optional[np.ndarray] = None,
		w: Optional[np.ndarray] = None,
		/,
		frequency: float = 1.0,
		) -> np.ndarray:

	if y is None:
		return _noise2v(
			x * frequency,
			np.zeros_like(x),
			open_simplex_obj._perm)
	elif z is None:
		return _noise2v(
			x * frequency,
			y * frequency,
			open_simplex_obj._perm)
	elif w is None:
		return _noise3v(
			x * frequency,
			y * frequency,
			z * frequency,
			open_simplex_obj._perm,
			open_simplex_obj._perm_grad_index3)
	else:
		return _noise4v(
			x * frequency,
			y * frequency,
			z * frequency,
			w * frequency,
			open_simplex_obj._perm)


def _xy_grid(width: int, height: int):
	max_dim = max(width, height)
	x_max = width / max_dim
	y_max = height / max_dim
	x = np.linspace(0.0, x_max, num=width, endpoint=False)
	y = np.linspace(0.0, y_max, num=height, endpoint=False)
	return np.meshgrid(x, y)


def _sphere_coord(height: int, width: Optional[int]=None, radius=1/TAU) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

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


def _cylinder_coord(width: int, height: int):
	# https://www.redblobgames.com/maps/terrain-from-noise/

	nx, ny = _xy_grid(width=width, height=height)

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


def _torus_4d_coord(width: int, height: int, radius=1/TAU):
	# https://www.redblobgames.com/maps/terrain-from-noise/
	nx, ny = _xy_grid(width=width, height=height)
	angle_x = TAU * nx
	angle_y = TAU * ny
	x = radius * np.cos(angle_x)
	y = radius * np.sin(angle_x)
	z = radius * np.cos(angle_y)
	w = radius * np.sin(angle_y)
	return x, y, z, w


class NoiseCoords:

	@classmethod
	def make_xy_grid(cls, width: int, height: int):
		return cls(*_xy_grid(width=width, height=height))

	@classmethod
	def make_sphere(cls, height: int, width: Optional[int]=None):
		return cls(*_sphere_coord(height=height, width=width))

	@classmethod
	def make_cylinder(cls, width: int, height: int):
		return cls(*_cylinder_coord(width=width, height=height))

	@classmethod
	def make_double_wrap(cls, width: int, height: int):
		return cls(*_torus_4d_coord(width=width, height=height))

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


@unique
class Normalization(Enum):
	none = auto()
	amplitude = auto()
	full = auto()


@unique
class FbmType(Enum):
	standard = auto()
	valley = auto()


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
			noise_type: FbmType = FbmType.standard,

			# TODO: add diff_steps

			# Engine
			use_fns: bool = USE_FNS_DEFAULT,
			):

		if octaves <= 0:
			raise ValueError(f'{octaves=}')

		self.seed = seed
		self.octaves = octaves
		self.gain = gain
		self.lacunarity = lacunarity
		self.base_frequency = base_frequency
		self.noise_type = noise_type
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

		seed = np.uint64(self.seed % (2 ** 64))
		self.fns = fns.Noise(seed=seed)

		self.fns.frequency = self.base_frequency

		# TODO: if octaves == 1, can just use Simplex; may be slightly better optimized
		self.fns.noiseType = fns.NoiseType.SimplexFractal

		if self.noise_type == FbmType.valley:
			self.fns.fractal.fractalType = fns.FractalType.Billow
		else:
			self.fns.fractal.fractalType = fns.FractalType.FBM

		self.fns.fractal.octaves = self.octaves
		self.fns.fractal.lacunarity = self.lacunarity
		self.fns.fractal.gain = self.gain

		self.fns.perturb.perturbType = fns.PerturbType.NoPerturb

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

	def _gen_fns(self, nc: NoiseCoords) -> np.ndarray:
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
			ret += amplitude * _noisev(open_simplex_obj, x, y, z, w, frequency=frequency)
			frequency *= self.lacunarity
			amplitude *= self.gain
		return ret

	def generate(
			self, nc: NoiseCoords, /, *,
			normalize: Union[Normalization, bool, None] = Normalization.none,
			) -> np.ndarray:

		if (normalize is None) or (normalize is False):
			normalize = Normalization.none
		elif normalize is True:
			normalize = Normalization.full

		if self.noise_type == FbmType.standard:
			return self._basic_noise(nc, normalize=normalize)

		elif self.noise_type == FbmType.valley:
			return self._valley_noise(nc, normalize=(normalize != Normalization.none))

		else:
			raise AssertionError(f'Enum value not handled: {self.noise_type!r}')

	def _basic_noise(
			self, nc: NoiseCoords, /, *,
			normalize: Normalization = Normalization.none,
			) -> np.ndarray:
		if self.use_fns and nc.dimension <= 3:
			ret = self._gen_fns(nc)
		else:
			ret = self._vectorized_noise_opensimplex(nc.x, nc.y, nc.z, nc.w)

		# TODO: Be smarter about amplitudes, so normalization is less necessary
		if normalize == Normalization.full:
			rescale(ret, range_out=(-1, 1), in_place=True)
		elif normalize == Normalization.amplitude:
			ret /= max_abs(ret)

		return ret

	def _valley_noise(self, nc: NoiseCoords, /, *, normalize=False):

		if self.use_fns and nc.dimension <= 3:
			self._init_fns()
			assert self.fns.fractal.fractalType == fns.FractalType.Billow
			ret = self._gen_fns(nc)
			range_in = (-1., np.amax(ret)) if normalize else (-1., 1.)
			rescale(ret, range_in, (0., 1.), in_place=True)

		else:
			ret = self._valley_noise_opensimplex(nc, normalize=normalize)
			if normalize:
				ret /= np.amax(ret)

		return ret

	def _valley_noise_opensimplex(self, nc: NoiseCoords, normalize: bool):

		self._init_opensimplex()
		assert self.open_simplex_objs

		ret = None
		frequency = self.base_frequency
		amplitude = self.init_amplitude
		for open_simplex_obj in self.open_simplex_objs:
			layer = _noisev(open_simplex_obj, nc.x, nc.y, nc.z, nc.w, frequency=frequency)
			np.abs(layer, out=layer)
			layer *= amplitude
			if ret is None:
				ret = layer
			else:
				ret += layer

			frequency *= self.lacunarity
			amplitude *= self.gain

		return ret


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
		coord = NoiseCoords.make_xy_grid(width=width, height=height)

	# TODO: take Normalization as argument to this function
	if normalize:
		normalize = Normalization.full
	elif normalize_amplitude:
		normalize = Normalization.amplitude
	else:
		normalize = Normalization.none

	fractal_noise = FractalNoise(
		seed=seed, octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)

	return fractal_noise.generate(coord, normalize=normalize)


def diff_fbm(
		seed: int,
		diff_steps: int,
		fbm_func: Optional[Callable[..., np.ndarray]] = None,
		normalize: bool = False,
		**kwargs,
		) -> np.ndarray:

	# TODO: make this a member function of FractalNoise

	# coord = NoiseCoords.make_xy_grid(width=width, height=height)  # TODO: use this

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

	# TODO: make this a member function of FractalNoise
	# TODO: use FNS built-in domain warping

	# TODO: take Normalization as argument to this function
	if normalize:
		normalize = Normalization.full
	elif normalize_amplitude:
		normalize = Normalization.amplitude
	else:
		normalize = Normalization.none

	x, y = _xy_grid(width, height)

	for idx in range(warp_steps):

		fbm_x = FractalNoise(
			seed=(seed + 1 + 2*idx), octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)
		fbm_y = FractalNoise(
			seed=(seed + 2 + 2*idx), octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)

		# TODO: normalize these?

		nc = NoiseCoords(x, y)
		x_warp = fbm_x.generate(nc)
		y_warp = fbm_y.generate(nc)

		x += (x_warp * warp_amount)
		y += (y_warp * warp_amount)

	nc = NoiseCoords(x, y)
	fractal_noise = FractalNoise(
		seed=seed, octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency)
	return fractal_noise.generate(nc, normalize=normalize)


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

	# TODO: take Normalization as argument to this function
	if normalize:
		normalize = Normalization.full
	elif normalize_amplitude:
		normalize = Normalization.amplitude
	else:
		normalize = Normalization.none

	transpose = False
	if wrap_x and wrap_y:
		nc = NoiseCoords.make_double_wrap(width=width, height=height)
	elif wrap_x:
		nc = NoiseCoords.make_cylinder(width=width, height=height)
	elif wrap_y:
		# TODO: make transpose unnecessary - calculate the right cylinder coordinates in the first place
		transpose = True
		nc = NoiseCoords.make_cylinder(width=width, height=height)
	else:
		nc = NoiseCoords.make_xy_grid(width=width, height=height)

	fractal_noise = FractalNoise(
		seed=seed, octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)

	img = fractal_noise.generate(nc, normalize=normalize)

	if transpose:
		img = img.transpose()

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

	# TODO: take Normalization as argument to this function
	if normalize:
		normalize = Normalization.full
	elif normalize_amplitude:
		normalize = Normalization.amplitude
	else:
		normalize = Normalization.none

	coord = NoiseCoords.make_sphere(height=height, width=width)

	fractal_noise = FractalNoise(
		seed=seed, octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)

	return fractal_noise.generate(coord, normalize=normalize)


def sphere_domain_warped_fbm(
		seed: int,
		height: int,
		warp_steps: int,
		width: Optional[int] = None,
		warp_amount: float = 1.0,
		octaves: int = 8,
		gain: float = 0.5,
		lacunarity: float = 2.0,
		base_frequency: float = 1.0,
		normalize: bool = False,
		normalize_amplitude: bool = False,
		use_fns: bool = USE_FNS_DEFAULT,
		) -> np.ndarray:

	# TODO: put domain warping inside FractalNoise
	# TODO: use FNS built-in domain warping

	# TODO: take Normalization as argument to this function
	if normalize:
		normalize = Normalization.full
	elif normalize_amplitude:
		normalize = Normalization.amplitude
	else:
		normalize = Normalization.none

	x, y, z = _sphere_coord(height, width=width)

	for idx in range(warp_steps):

		fbm_x = FractalNoise(
			seed=(seed + 1 + 3*idx), octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)
		fbm_y = FractalNoise(
			seed=(seed + 2 + 3*idx), octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)
		fbm_z = FractalNoise(
			seed=(seed + 3 + 3*idx), octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency, use_fns=use_fns)

		# TODO: normalize?

		nc = NoiseCoords(x, y, z)
		x_warp = fbm_x.generate(nc)
		y_warp = fbm_y.generate(nc)
		z_warp = fbm_z.generate(nc)

		# TODO: support math operations with NoiseCoord, instead of needing to keep track of separate x/y/z
		x += (x_warp * warp_amount)
		y += (y_warp * warp_amount)
		z += (z_warp * warp_amount)

	nc = NoiseCoords(x, y, z)

	fractal_noise = FractalNoise(
		seed=seed, octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency)
	return fractal_noise.generate(nc, normalize=normalize)


def valley_fbm(
		seed: int,
		coord: Optional[NoiseCoords] = None,
		width: Optional[int] = None,
		height: Optional[int] = None,
		octaves: int = 8,
		gain: float = 0.5,
		lacunarity: float = 2.0,
		base_frequency: float = 1.0,
		normalize: bool = False,
		use_fns: bool = USE_FNS_DEFAULT,
		) -> np.ndarray:

	normalize = Normalization.full if normalize else Normalization.amplitude

	if coord is None:
		if width is None or height is None:
			raise ValueError('Must provide width & height if not providing coord')
		coord = NoiseCoords.make_xy_grid(width=width, height=height)

	fractal_noise = FractalNoise(
		seed=seed,
		octaves=octaves, gain=gain, lacunarity=lacunarity, base_frequency=base_frequency,
		noise_type=FbmType.valley,
		use_fns=use_fns,
	)

	return fractal_noise.generate(coord, normalize=normalize)


def ridge_fbm(seed: int, **kwargs) -> np.ndarray:
	"""
	Generate ridge noise, i.e. 1 - valley_fbm()
	:param kwargs: Args to forward to valley_fbm()
	"""
	return 1.0 - valley_fbm(seed=seed, **kwargs)
