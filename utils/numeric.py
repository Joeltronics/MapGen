#!/usr/bin/env python

from typing import Final, Optional, Tuple, TypeVar, Union

import numpy as np
import scipy.signal

from utils.consts import PI, TWOPI, EARTH_POLAR_CIRCUMFERENCE_M, EARTH_EQUATORIAL_CIRCUMFERENCE_M


T = TypeVar('T')
FloatOrArrayT = TypeVar('FloatOrArrayT', float, np.ndarray)


def linspace_midpoint(start, stop, num, **kwargs):
	vals, step = np.linspace(start, stop, num, retstep=True, endpoint=False, **kwargs)
	vals += 0.5 * step
	return vals


def require_same_shape(*args: np.ndarray) -> None:
	if not args:
		return
	if not all(arr.shape == args[0].shape for arr in args[1:]):
		raise ValueError('Arrays must have the same shape: ' + ', '.join(str(arr.shape) for arr in args))


def max_abs(x: np.ndarray, /) -> float:
	return np.amax(np.abs(x))


def lerp(x: FloatOrArrayT, /, range: Tuple[float, float]) -> FloatOrArrayT:
	return range[0]*(1-x) + range[1]*x


def reverse_lerp(y: FloatOrArrayT, /, range: Tuple[float, float]) -> FloatOrArrayT:
	if range[1] == range[0]:
		return np.zeros_like(y)
	return (y - range[0]) / (range[1] - range[0])


def rescale(
		val: FloatOrArrayT,
		/,
		range_in: Optional[Tuple[float, float]] = None,
		range_out: Optional[Tuple[float, float]] = None,
		*,
		bipolar = False,
		clip = False,
		in_place = False,
		) -> FloatOrArrayT:
	"""
	:param val: value to be rescaled. If not an array, range_in must be provided
	:param range_in: if None, defaults to range of input data, or +/- max_abs(val) if bipolar
	:param range_out: if None, defaults to (0, 1), or (-1, 1) if bipolar
	:param clip: if True, will clip output to range_out
	"""

	scalar = np.isscalar(val)

	if scalar and in_place:
		raise ValueError('Cannot use in_place with scalar')

	if range_out is None:
		range_out = (-1., 1.) if bipolar else (0., 1.)

	if range_in is None:
		# TODO: optimize this like with in-place case
		if scalar:
			raise ValueError('Must provide range_in with scalar')

		if bipolar:
			range_in = max_abs(val)
			range_in = (-range_in, range_in)
		else:
			range_in = data_range(val)

	elif range_in[0] == range_in[1]:
		ret = 0.5*(range_out[0] + range_out[1])
		if scalar:
			return ret
		elif in_place:
			val.fill(ret)
			return val
		else:
			return np.full_like(val, ret)

	if (not scalar) and (not in_place):
		val = np.copy(val)

	# These optimizations might seem pointless, but this function is mainly designed for use with large arrays

	if range_in[0] != 0:
		val -= range_in[0]

	if range_in[1] - range_in[0] != 1:
		val /= (range_in[1] - range_in[0])

	if clip and scalar:
		val = np.clip(val, 0., 1.)
	elif clip:
		np.clip(val, 0., 1., out=val)

	if range_out[1] - range_out[0] != 1:
		val *= (range_out[1] - range_out[0])

	if range_out[0] != 0:
		val += range_out[0]

	return val


def rescale_in_place(
		val: np.ndarray,
		/,
		range_in: Optional[Tuple[float, float]]=None,
		range_out: Tuple[float, float]=(0., 1.),
		*,
		clip=False
		) -> None:
	"""
	DEPRECATED - use rescale(in_place=True) instead

	:param val: value to be rescaled
	:param range_in: if None, will use range of input data (val must be array)
	:param range_out:
	:param clip: if True, will clip output to range_out
	"""

	if range_in is None:
		val -= np.amin(val)
		val /= np.amax(val)
	elif range_in[0] == range_in[1]:
		val[...] = 0.5*(range_out[0] + range_out[1])
		return
	elif range_in != (0., 1.):
		val -= range_in[0]
		val /= (range_in[1] - range_in[0])

	if clip:
		np.clip(val, 0., 1., out=val)

	if range_out != (0., 1.):
		val *= (range_out[1] - range_out[0])

	if range_out[0]:
		val += range_out[0]


def data_range(x: np.ndarray, /, ignore_nan=True) -> Tuple[float, float]:
	if ignore_nan:
		x = x[np.bitwise_not(np.isnan(x))]
	return np.amin(x), np.amax(x)


def magnitude(x: np.ndarray, y: np.ndarray) -> np.ndarray:
	require_same_shape(x, y)
	return np.sqrt(np.square(x) + np.square(y))


def clip_max_vector_magnitude(x: np.ndarray, y: np.ndarray, /, max_magnitude, *, in_place=False) -> tuple[np.ndarray, np.ndarray]:

	mag = magnitude(x, y)

	if True:
		direction = np.arctan2(y, x)
		mag = np.minimum(mag, max_magnitude)

		if in_place:
			np.cos(direction, out=x)
			np.sin(direction, out=y)
			x *= mag
			y *= mag
		else:
			x = mag * np.cos(direction)
			y = mag * np.sin(direction)

	else:
		mask = (mag > max_magnitude)

		if not in_place:
			x = x.copy()
			y = y.copy()

		x[mask] /= mag[mask]
		y[mask] /= mag[mask]

	return x, y
