#!/usr/bin/env python3

from typing import Final, Optional

import numpy as np

from data.data import get_topography, get_mask

from utils.numeric import rescale
from utils.image import resize_array
from utils.utils import tprint


def get_earth_topography(
		width: int,
		height: int,
		) -> np.ndarray:
	"""
	:returns: (topography in meters, topography in in range [-1, 1])
	"""

	topography_m = get_topography()

	needs_resize = (topography_m.shape != (height, width))

	ELEVATION_DATA_RANGE: Final = (-8000., 6400.)

	# Current generation model just uses elevation for ocean, so it can't handle land below sea level
	# Set land minimum to slightly above sea level, ocean maximum to slightly below
	min_land_elevation_m = 0.1

	land_mask = get_mask(land=True, ocean=False, lakes=True)
	ocean_mask = np.logical_not(land_mask)
	topography_m[land_mask] = np.maximum(topography_m[land_mask], min_land_elevation_m)
	topography_m[ocean_mask] = np.minimum(topography_m[ocean_mask], -min_land_elevation_m)

	if needs_resize:
		topography_m = resize_array(topography_m, (width, height), data_range=ELEVATION_DATA_RANGE)

	topography_norm = rescale(topography_m, (-8000., 8000.), (-1., 1.))

	return topography_m, topography_norm


def scale_topography_for_water_level(topography_norm: np.ndarray, water_amount=0.5, power_scale=False) -> np.ndarray:
	"""
	:param topography_norm: Topography in range [-1, 1]
	:param water_amount: Amount of water - does not correspond to actual percentage, but may behave similarly
	:returns: Topography in in range [-1, 1]
	"""

	if water_amount == 0.5:
		topography_norm = np.copy(topography_norm)

	elif power_scale:
		power = 2.0 ** rescale(water_amount, (0.0, 1.0), (-4.0, 4.0))
		topography_norm = topography_norm * 0.5 + 0.5
		topography_norm = np.power(topography_norm, power)
		topography_norm = topography_norm * 2.0 - 1.0

	else:
		water_level = water_amount * 2.0 - 1.0

		topography_norm = topography_norm - water_level
		max_elevation = 1.0 - water_level
		min_elevation = -1.0 - water_level

		topography_norm[topography_norm >= 0] = rescale(topography_norm[topography_norm >= 0], (0.0, max_elevation), (0.0, 1.0))
		topography_norm[topography_norm < 0] = rescale(topography_norm[topography_norm < 0], (0.0, min_elevation), (0.0, -1.0))

	# For flat areas by shores, continental shelves, etc
	# TODO: steeper continental shelf dropoff
	topography_norm[topography_norm >= 0] = np.square(topography_norm[topography_norm >= 0])
	topography_norm[topography_norm < 0] = -np.square(topography_norm[topography_norm < 0])

	return topography_norm


def _erode_mountain_cells(elevation, mountain_cells, erosion_amount=0.5):

	# TODO: scale erosion by rainfall - more rain means more erosion

	assert np.amin(mountain_cells) >= 0
	assert np.amax(mountain_cells) <= 1

	erosion = 1.0 - mountain_cells

	erosion_amount = rescale(elevation, (0.1, 1.0), (0.0, erosion_amount), clip=True)
	return elevation - erosion_amount * erosion


def erode(
		topography_norm: np.ndarray,
		valley_noise: np.ndarray,
		amount,
		):
	"""
	:param topography_norm: Topography in range [0, 1]
	:param valley_noise: FBM valley noise
	:returns: Topography in in range [-1, 1]
	"""

	assert np.amin(valley_noise) >= 0
	assert np.amax(valley_noise) <= 1

	erosion = 1.0 - valley_noise

	amount = rescale(topography_norm, (0.1, 1.0), (0.0, amount), clip=True)
	return topography_norm - amount * erosion


def generate_topography(
		topography_noise: np.ndarray,
		valley_noise: Optional[np.ndarray],
		water_amount: float,
		erosion_amount: float,
		) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

	topography_norm = scale_topography_for_water_level(topography_noise, water_amount=water_amount)

	if erosion_amount > 0 and valley_noise is not None:
		topography_before_erosion = topography_norm
		topography_norm = erode(topography_norm, valley_noise=valley_noise, amount=erosion_amount)
		erosion = (topography_norm - topography_before_erosion) / topography_norm
	else:
		erosion = np.zeros_like(topography_norm)

	topography_m = rescale(topography_norm, (-1., 1.), (-8000., 8000.))

	return topography_m, topography_norm, erosion
