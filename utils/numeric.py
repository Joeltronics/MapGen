#!/usr/bin/env python

from typing import Optional, Tuple, TypeVar, Union

import numpy as np


T = TypeVar('T')
FloatOrArrayT = TypeVar('FloatOrArrayT', float, np.ndarray)


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
		range_in: Optional[Tuple[float, float]]=None,
		range_out: Tuple[float, float]=(0., 1.),
		*,
		clip=False,
		) -> FloatOrArrayT:
	"""
	:param val: value to be rescaled
	:param range_in: if None, will use range of input data (val must be array)
	:param range_out:
	:param clip: if True, will clip output to range_out
	"""

	scalar = np.isscalar(val)

	if range_in is None:
		# TODO: optimize this like with in-place case
		if scalar:
			raise ValueError('Must provice range_in with scalar')
		range_in = data_range(val)

	if range_in[0] == range_in[1]:
		ret = 0.5*(range_out[0] + range_out[1])
		return ret if scalar else np.full_like(val, ret)

	if not scalar:
		val = np.copy(val)

	if range_in != (0., 1.):
		val -= range_in[0]
		val /= (range_in[1] - range_in[0])

	if clip and scalar:
		val = np.clip(val, 0., 1.)
	elif clip:
		np.clip(val, 0., 1., out=val)

	if range_out != (0., 1.):
		val *= (range_out[1] - range_out[0])
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
		val += range_out[0]


def data_range(x: np.ndarray, /, ignore_nan=True) -> Tuple[float, float]:
	if ignore_nan:
		x = x[np.bitwise_not(np.isnan(x))]
	return np.amin(x), np.amax(x)
