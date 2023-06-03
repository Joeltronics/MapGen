#!/usr/bin/env python3

from functools import cached_property
from typing import Final

import numpy as np

from utils.numeric import rescale, require_same_shape

from .map_properties import MapProperties
from .topography import Terrain
from .winds import WindModel


DEFAULT_PRECIPITATION_RANGE_CM: Final = (0.5, 400)


# TODO: rename rainfall -> precipitation everywhere


def latitude_rainfall_fn(latitude_radians: np.ndarray) -> np.ndarray:
	# Roughly based on https://commons.wikimedia.org/wiki/File:Relationship_between_latitude_vs._temperature_and_precipitation.png
	# return (np.cos(2 * latitude_radians) * 0.5 + 0.5) * (np.cos(6 * latitude_radians) * 0.5 + 0.5)
	return 0.5*(np.cos(2*latitude_radians) * 0.5 + 0.5) + 0.5*(np.cos(6*latitude_radians) * 0.5 + 0.5)
	# return 0.4*(np.cos(2*latitude_radians) * 0.5 + 0.5) + 0.6*(np.cos(6*latitude_radians) * 0.5 + 0.5)


def _calculate_rainfall(
		noise: np.ndarray,
		latitude_deg: np.ndarray,
		noise_strength=0.25,
		precipitation_range_cm=DEFAULT_PRECIPITATION_RANGE_CM,
		) -> np.ndarray:

	require_same_shape(noise, latitude_deg)

	latitude = np.radians(latitude_deg)
	latitude_rainfall_map = latitude_rainfall_fn(latitude)

	# TODO: should this use domain warping instead of interpolation? or combination of both?
	rainfall_01 = noise * noise_strength + latitude_rainfall_map * (1.0 - noise_strength)

	rainfall_cm = rescale(rainfall_01, (0.0, 1.0), precipitation_range_cm)
	return rainfall_cm


class PrecipitationModel:
	def __init__(
			self,
			map_properties: MapProperties,
			terrain: Terrain,
			wind: WindModel,
			*,
			effective_latitude_deg: np.ndarray,
			noise: np.ndarray,
			noise_strength = 0.25,
			precipitation_range_cm = DEFAULT_PRECIPITATION_RANGE_CM,
			):
		self._properties = map_properties
		self._terrain = terrain
		self._wind = wind
		self._effective_latitude_deg = effective_latitude_deg
		self._noise = noise
		self._noise_strength = noise_strength
		self._precipitation_range_cm = precipitation_range_cm

	@cached_property
	def wind_dir_dot_gradient(self):
		wind_x, wind_y = self._wind.direction
		gradient_x, gradient_y = self._terrain.gradient_100km
		return wind_x * gradient_x + wind_y * gradient_y

	def clear_cache(self):
		del self.wind_dir_dot_gradient

	def process(self, keep_cache=False) -> np.ndarray:
		rain_cm = _calculate_rainfall(
			noise=self._noise,
			latitude_deg=self._latitude_deg,
			noise_strength=self._noise_strength,
			precipitation_range_cm=self._precipitation_range_cm,
		)

		# TODO: wind_dir_dot_gradient -> adiabatic rainfall

		# TODO: rain shadows

		if not keep_cache:
			self.clear_cache()

		return rain_cm


def calculate_rainfall(
		noise: np.ndarray,
		latitude_deg: np.ndarray,
		noise_strength=0.25,
		precipitation_range_cm=DEFAULT_PRECIPITATION_RANGE_CM,
		) -> np.ndarray:
	return _calculate_rainfall(
		noise=noise,
		latitude_deg=latitude_deg,
		noise_strength=noise_strength,
		precipitation_range_cm=precipitation_range_cm,
	)


def main(args=None):
	import argparse



if __name__ == "__main__":
	main()
