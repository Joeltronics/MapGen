#!/usr/bin/env python3

from functools import cached_property
from math import isclose
from typing import Final, Optional

import numpy as np

from data.data import get_topography, get_mask

from .map_properties import MapProperties

from utils.numeric import rescale, max_abs, magnitude
from utils.image import resize_array, map_gradient, gaussian_blur_map
from utils.utils import tprint


class Terrain:
	def __init__(
			self,
			map_properties: MapProperties,
			terrain_m: Optional[np.ndarray] = None,
			terrain_norm: Optional[np.ndarray] = None,
			erosion: Optional[np.ndarray] = None,
			):

		if terrain_m is None:
			if terrain_norm is None:
				raise ValueError('Must provide either terrain_m or terrain_norm')
			terrain_m = rescale(terrain_norm, (-1., 1.), (-8000., 8000.))
		elif terrain_norm is None:
			max_val = max(8000, max_abs(terrain_m))
			terrain_norm = rescale(terrain_m, (-max_val, max_val), (-1, 1))

		self._map_properties = map_properties

		self._terrain_m = terrain_m
		self._terrain_norm = terrain_norm

		self._elevation_m = np.maximum(terrain_m, 0.)

		self._erosion = erosion

	# Simple getter properties

	@property
	def terrain_m(self) -> np.ndarray:
		return self._terrain_m
	
	@property
	def terrain_norm(self) -> np.ndarray:
		return self._terrain_norm

	@property
	def elevation_m(self) -> np.ndarray:
		return self._elevation_m

	@property
	def erosion(self) -> Optional[np.ndarray]:
		return self._erosion

	# Cached properties

	@cached_property
	def land_mask(self) -> np.ndarray:
		return self._terrain_m >= 0

	@cached_property
	def elevation_100km(self) -> np.ndarray:
		return gaussian_blur_map(self.elevation_m, sigma_km=100.0, flat_map=self._map_properties.flat, latitude_span=self._map_properties.latitude_span)

	@cached_property
	def elevation_1000km(self) -> np.ndarray:
		return gaussian_blur_map(self.elevation_m, sigma_km=1000.0, flat_map=self._map_properties.flat, latitude_span=self._map_properties.latitude_span)

	@cached_property
	def gradient(self) -> tuple[np.ndarray, np.ndarray]:
		return map_gradient(self.elevation_m, flat_map=self._map_properties.flat, magnitude=False, latitude_span=self._map_properties.latitude_span)

	@cached_property
	def gradient_100km(self) -> tuple[np.ndarray, np.ndarray]:
		# TODO optimization: can use elevation_100km for this
		return self.gradient_at_scale(scale_km=100, _allow_cached_properties=False)

	@cached_property
	def gradient_1000km(self) -> tuple[np.ndarray, np.ndarray]:
		# TODO optimization: can use elevation_1000km for this
		return self.gradient_at_scale(scale_km=1000, _allow_cached_properties=False)

	@cached_property
	def gradient_magnitude(self) -> np.ndarray:
		return magnitude(*self.gradient)

	@cached_property
	def gradient_magnitude_100km(self) -> np.ndarray:
		return magnitude(*self.gradient_100km)

	@cached_property
	def gradient_magnitude_1000km(self) -> np.ndarray:
		return magnitude(*self.gradient_100km)

	# Calculation methods

	def gradient_at_scale(self, scale_km, _allow_cached_properties=True) -> tuple[np.ndarray, np.ndarray]:

		# _allow_cached_properties to prevent recursion loop
		# TODO: instead of needing _allow_cached_properties argument, check if property is already cached
		if _allow_cached_properties:
			if scale_km == 100:
				return self.gradient_100km
			elif scale_km == 1000:
				return self.gradient_1000km

		return map_gradient(
			self.elevation_m,
			sigma_km=scale_km,
			flat_map=self._map_properties.flat,
			latitude_span=self._map_properties.latitude_span,
			magnitude=False,
		)

	def gradient_magnitude_at_scale(self, scale_km, _allow_cached_properties=True) -> np.ndarray:

		# TODO: as above, check cached_property
		if _allow_cached_properties:
			if scale_km == 100:
				return self.gradient_magnitude_100km
			elif scale_km == 1000:
				return self.gradient_magnitude_1000km

		return map_gradient(
			self.elevation_m,
			sigma_km=scale_km,
			flat_map=self._map_properties.flat,
			latitude_span=self._map_properties.latitude_span,
			magnitude=True,
		)


def get_earth_topography(map_properties: MapProperties) -> Terrain:

	if not all([
			isclose(map_properties.latitude_range[0], -90),
			isclose(map_properties.latitude_range[1], 90),
			isclose(map_properties.longitude_range[0], -180),
			isclose(map_properties.longitude_range[1], 180)]):
		raise NotImplementedError(
			'latitude & longitude ranges must be full globe for get_earth_topography; ' +
			f'lat {map_properties.latitude_range}; lon {map_properties.longitude_range}')

	width = map_properties.width
	height = map_properties.height

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

	return Terrain(map_properties=map_properties, terrain_m=topography_m, terrain_norm=topography_norm)


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

		land = topography_norm >= 0
		ocean = np.logical_not(land)

	else:
		water_level = water_amount * 2.0 - 1.0

		topography_norm = topography_norm - water_level
		max_elevation = 1.0 - water_level
		min_elevation = -1.0 - water_level

		land = topography_norm >= 0
		ocean = np.logical_not(land)

		topography_norm[land] = rescale(topography_norm[land], (0.0, max_elevation), (0.0, 1.0))
		topography_norm[ocean] = rescale(topography_norm[ocean], (0.0, min_elevation), (0.0, -1.0))

	# For flat areas by shores, and to make the highest mountains relatively rare
	topography_norm[land] = np.power(topography_norm[land], 3.0)

	# topography_norm[ocean] = -np.square(topography_norm[ocean])

	# Abyssal plain
	# y = 0.5 * (2*x + 1)^3 - 0.5
	topography_norm[ocean] = 0.5 * (2 * topography_norm[ocean] + 1) ** 3.0 - 0.5

	# Continental shelf
	# shelf = np.logical_and(ocean, topography_norm > -0.1)
	shelf_scale = rescale(topography_norm, (-0.2, -0.1), (1., 0.25), clip=True)
	shelf_scale[land] = 1.
	topography_norm *= shelf_scale

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
	erosion = np.square(1.0 - valley_noise)

	# TODO: should there be some erosion right down to sea level too?

	amount = rescale(topography_norm, (0.05, 1.0), (0.0, amount), clip=True)
	return topography_norm - amount * erosion


def generate_topography(
		map_properties: MapProperties,
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

	return Terrain(map_properties=map_properties, terrain_m=topography_m, terrain_norm=topography_norm, erosion=erosion)
