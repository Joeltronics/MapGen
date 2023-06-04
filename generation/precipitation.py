#!/usr/bin/env python3

from functools import cached_property
from typing import Final

import numpy as np

from utils.image import resize_array
from utils.numeric import rescale, require_same_shape, magnitude
from utils.utils import tprint

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

		# TODO: this is now base precipitation range, it should probably be lower now that we're adding orographic
		self._precipitation_range_cm = precipitation_range_cm

	@cached_property
	def base_rain_cm(self):
		# Older naive calculation
		return _calculate_rainfall(
			noise=self._noise,
			latitude_deg=self._latitude_deg,
			noise_strength=self._noise_strength,
			precipitation_range_cm=self._precipitation_range_cm,
		)

	@cached_property
	def wind_dir_dot_gradient(self):
		wind_x, wind_y = self._wind.direction
		gradient_x, gradient_y = self._terrain.gradient_100km
		return wind_x * gradient_x + wind_y * gradient_y

	@cached_property
	def orographic_precipitation_scale(self):
		# Orographic rainfall (where wind is going uphill)
		OROGRAPHIC_RAIN_SCALE = 10.
		# TODO: should the max be capped here?
		orographic_rain = np.maximum(0, -self.wind_dir_dot_gradient)
		# TODO: should this be a 1/x scale instead of linear?
		return 1 + OROGRAPHIC_RAIN_SCALE*orographic_rain

	def clear_cache(self):
		del self.base_rain_cm
		del self.wind_dir_dot_gradient

	def process(self, keep_cache=False) -> np.ndarray:

		# Older naive calculation

		base_rain_cm = self.base_rain_cm

		# Orographic rainfall (where wind is going uphill)

		rain_cm = base_rain_cm * self.orographic_precipitation_scale

		# Rain shadows

		# TODO

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

	from matplotlib import pyplot as plt
	from matplotlib.gridspec import GridSpec
	from matplotlib.axes import Axes

	from .test_datasets import get_test_datasets

	# TODO: de-duplicate this from winds.py

	parser = argparse.ArgumentParser()
	mx = parser.add_mutually_exclusive_group()
	mx.add_argument('--circle', dest='circle_only', action='store_true', help='Only run circle test')
	mx.add_argument('--fullres', action='store_true', help='Include full resolution simulation')
	args = parser.parse_args(args)

	MAX_ARROWS_HEIGHT_STANDARD: Final = 180 // 5
	MAX_ARROWS_HEIGHT_HIGH_RES: Final = 180 // 2

	datasets = get_test_datasets(
		full_res_earth = args.fullres,
		lower_res_earch = not args.circle_only,
		earth_flat = not args.circle_only,
		north_america = not args.circle_only,
		circle = True,
	)

	for dataset in datasets:
		dataset_title = dataset['title']

		print()
		tprint('Processing ' + dataset_title)

		source_data = dataset['source_data']
		resolution = dataset.get('resolution', None)
		if resolution is None:
			resolution = source_data.shape
		height, width = resolution

		high_res_arrows = dataset.get('high_res_arrows', False)

		map_properties = MapProperties(
			flat=dataset['flat_map'],
			height=height,
			width=width,
			latitude_range=dataset.get('latitude_range', (-90, 90)),
			longitude_range=dataset.get('longitude_range', None),
		)
		latitude_range = map_properties.latitude_range
		longitude_range = map_properties.longitude_range

		if resolution != source_data.shape:
			tprint(f'Resizing {source_data.shape[1]}x{source_data.shape[0]} -> {width}x{height}...')
			topography_m = resize_array(source_data, (width, height))
		else:
			topography_m = source_data

		assert topography_m.shape == (height, width)

		terrain = Terrain(map_properties=map_properties, terrain_m=topography_m)

		tprint('Calculating wind')

		wind_sim = WindModel(map_properties=map_properties, terrain=terrain)
		wind_x, wind_y = wind_sim.process()

		# TODO







if __name__ == "__main__":
	main()
