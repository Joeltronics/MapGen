#!/usr/bin/env python3

import numpy as np

from utils.numeric import linspace_midpoint


def make_latitude_map(
		height: int,
		width: int = 1,
		latitude_range: tuple[float, float] = (-90., 90.),
		radians: bool = False,
		endpoint: bool = False,
		) -> np.ndarray:
	"""
	Make latitude map

	:param height: Image height
	:param width: Image width; if 0, return 1D array; if 1, return column vector; if > 1, return 2D array (repeated column vector)
	:param latitude_range: Range of latitudes spanned. Order can be either direction.
	:param radians: if True, latitude will be in radians; otherwise will be in degrees
	:param endpoint: if True, includes exact endpoints of latitude range; if False, returns midpoint of each pixel
	"""

	if width < 0:
		raise ValueError('Width must be >= 0')

	if height <= 1:
		raise ValueError('height must be > 1')

	if len(latitude_range) != 2:
		raise TypeError(f'latitude_range must have 2 elements: {latitude_range}')

	south, north = sorted(latitude_range)

	if endpoint:
		latitude = np.linspace(north, south, height, endpoint=True)
	else:
		latitude = linspace_midpoint(north, south, height)

	if radians:
		latitude = np.radians(latitude)

	if width == 0:
		return latitude

	latitude = latitude[:, np.newaxis]

	if width == 1:
		return latitude

	return latitude_vector_to_map(latitude=latitude, width=width)


def latitude_vector_to_map(latitude: np.ndarray, width: int) -> np.ndarray:
	"""
	Takes 1D latitude vector and converts to 2D map
	:param latitude: Either 1D, or column vector (2D with shape [N, 1])
	"""

	if len(latitude.shape) == 1:
		latitude = latitude[: np.newaxis]
	elif len(latitude.shape) != 2 or latitude.shape[1] != 1:
		raise ValueError(f'Invalid shape: {latitude.shape}')

	return np.repeat(latitude, repeats=width, axis=1)
