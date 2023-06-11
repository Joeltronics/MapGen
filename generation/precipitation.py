#!/usr/bin/env python3

from functools import cached_property
from typing import Final, Optional

import numpy as np

from utils.image import resize_array
from utils.numeric import data_range, linspace_midpoint, magnitude, rescale, require_same_shape
from utils.utils import tprint

from .map_properties import MapProperties
from .topography import Terrain
from .winds import WindModel


DEFAULT_PRECIPITATION_RANGE_CM: Final = (0.5, 400)


# OROGRAPHIC_PRECIP_SCALE = 10.
OROGRAPHIC_PRECIP_SCALE = 100.


# TODO: rename rainfall -> precipitation everywhere


def latitude_rainfall_fn(latitude_radians: np.ndarray) -> np.ndarray:
	# Roughly based on https://commons.wikimedia.org/wiki/File:Relationship_between_latitude_vs._temperature_and_precipitation.png
	# return (np.cos(2 * latitude_radians) * 0.5 + 0.5) * (np.cos(6 * latitude_radians) * 0.5 + 0.5)
	return 0.5*(np.cos(2*latitude_radians) * 0.5 + 0.5) + 0.5*(np.cos(6*latitude_radians) * 0.5 + 0.5)
	# return 0.4*(np.cos(2*latitude_radians) * 0.5 + 0.5) + 0.6*(np.cos(6*latitude_radians) * 0.5 + 0.5)


def _calculate_rainfall(
		noise: Optional[np.ndarray],
		latitude_deg: np.ndarray,
		noise_strength=0.25,
		precipitation_range_cm=DEFAULT_PRECIPITATION_RANGE_CM,
		) -> np.ndarray:

	if noise is not None:
		require_same_shape(noise, latitude_deg)

	latitude = np.radians(latitude_deg)
	latitude_rainfall_map = latitude_rainfall_fn(latitude)

	if noise_strength > 0:
		# TODO: should this use domain warping instead of interpolation? or combination of both?
		rainfall_01 = noise * noise_strength + latitude_rainfall_map * (1.0 - noise_strength)
	else:
		rainfall_01 = latitude_rainfall_map

	rainfall_cm = rescale(rainfall_01, (0.0, 1.0), precipitation_range_cm)
	return rainfall_cm


class PrecipitationModel:
	def __init__(
			self,
			map_properties: MapProperties,
			terrain: Terrain,
			wind: WindModel,
			*,
			effective_latitude_deg: Optional[np.ndarray],
			noise: np.ndarray,
			noise_strength = 0.25,
			precipitation_range_cm = DEFAULT_PRECIPITATION_RANGE_CM,
			):
		self._properties = map_properties
		self._terrain = terrain
		self._wind = wind
		# TODO optimization: use map_properties.latitude_column instead
		self._effective_latitude_deg = effective_latitude_deg if (effective_latitude_deg is not None) else map_properties.latitude_map
		self._noise = noise
		self._noise_strength = noise_strength

		# TODO: this is now *base* precipitation range, it should probably be lower now that orographic can add extra
		self._precipitation_range_cm = precipitation_range_cm

		self._precipitation_cm = None

	@property
	def precipitation_cm(self) -> np.ndarray:
		if self._precipitation_cm is None:
			self.process()
		assert self._precipitation_cm is not None
		return self._precipitation_cm

	@cached_property
	def base_precipitation_cm(self):
		# Older naive calculation
		return _calculate_rainfall(
			noise=self._noise,
			latitude_deg=self._effective_latitude_deg,
			noise_strength=self._noise_strength,
			precipitation_range_cm=self._precipitation_range_cm,
		)

	@cached_property
	def wind_dir_dot_gradient(self):
		wind_x, wind_y = self._wind.direction
		gradient_x, gradient_y = self._terrain.gradient_100km
		# TODO: blur this
		# Even though terrain gradient is already blurred, wind isn't and has weird convergence/divergence behavior
		return wind_x * gradient_x + wind_y * gradient_y

	@cached_property
	def orographic_precipitation_scale(self):
		# Orographic rainfall (where wind is going uphill)
		# TODO: should the max be capped here?
		# TODO: should this be a 1/x scale instead of linear?
		return 1 + OROGRAPHIC_PRECIP_SCALE*np.maximum(0, self.wind_dir_dot_gradient)

	def clear_cache(self):
		del self.base_precipitation_cm
		del self.wind_dir_dot_gradient

	def process(self, keep_cache=False) -> np.ndarray:

		# Older naive calculation

		base_precipitation_cm = self.base_precipitation_cm

		# TODO: probably makes more sense to do most stuff after this in log domain

		# Orographic rainfall (where wind is going uphill)

		rain_cm = base_precipitation_cm * self.orographic_precipitation_scale

		# Rain shadows

		# TODO

		self._precipitation_cm = rain_cm

		if not keep_cache:
			self.clear_cache()

		return self._precipitation_cm


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

	import matplotlib
	from matplotlib import pyplot as plt
	from matplotlib import colors
	from matplotlib.gridspec import GridSpec
	from matplotlib.axes import Axes

	from .test_datasets import get_test_datasets

	# TODO: de-duplicate this from winds.py

	parser = argparse.ArgumentParser()
	mx = parser.add_mutually_exclusive_group()
	mx.add_argument('--circle', dest='circle_only', action='store_true', help='Only run circle test')
	mx.add_argument('--multires', action='store_true', help='Run earth simulation at multiple resolutions')
	mx.add_argument('--fullres', action='store_true', help='Include full resolution simulation')
	args = parser.parse_args(args)

	MAX_ARROWS_HEIGHT_STANDARD: Final = 180 // 5
	MAX_ARROWS_HEIGHT_HIGH_RES: Final = 180 // 2

	standard_res_earth = not (args.circle_only or args.fullres)
	lower_res_earth = args.multires
	earth_regions = standard_res_earth
	circle = args.circle_only or not args.fullres

	datasets = get_test_datasets(
		earth_3600 = args.fullres,
		earth_1024 = standard_res_earth,
		earth_256 = lower_res_earth,
		earth_1024_flat = lower_res_earth,
		africa = earth_regions,
		north_america = earth_regions,
		south_america = earth_regions,
		pacific_northwest = earth_regions,
		circle = circle,
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

		tprint('Calculating rain')

		rain_sim = PrecipitationModel(
			map_properties=map_properties,
			terrain=terrain,
			wind=wind_sim,
			effective_latitude_deg=None,
			noise=None,
			noise_strength=0.0,
		)
		rain_sim.process(keep_cache=True)

		delta_precip = rain_sim.precipitation_cm / rain_sim.base_precipitation_cm

		tprint('Calculating wind arrows to plot')

		max_arrows_height = MAX_ARROWS_HEIGHT_HIGH_RES if high_res_arrows else MAX_ARROWS_HEIGHT_STANDARD
		if height >= max_arrows_height:
			arrows_height = max_arrows_height
			arrows_width = round(max_arrows_height * width / height)
		else:
			arrows_width = width
			arrows_height = height

		arrows_size = (arrows_width, arrows_height)
		arrows_x = resize_array(wind_x, new_size=arrows_size)
		arrows_y = resize_array(wind_y, new_size=arrows_size)

		arrow_mags = magnitude(arrows_x, arrows_y)
		assert np.amin(arrow_mags) > 0
		arrows_x_norm = arrows_x / arrow_mags
		arrows_y_norm = arrows_y / arrow_mags

		arrow_locs_x = linspace_midpoint(longitude_range[0], longitude_range[1], arrows_x.shape[1])
		arrow_locs_y = linspace_midpoint(latitude_range[1], latitude_range[0], arrows_x.shape[0])
		arrow_step = abs(arrow_locs_y[1] - arrow_locs_y[0])
		arrow_locs_x, arrow_locs_y = np.meshgrid(arrow_locs_x, arrow_locs_y)

		base_wind_x, base_wind_y = wind_sim.base_winds_mps
		base_dir_arrows_x = resize_array(base_wind_x, new_size=arrows_size)
		base_dir_arrows_y = resize_array(base_wind_y, new_size=arrows_size)
		mag = magnitude(base_dir_arrows_x, base_dir_arrows_y)
		base_dir_arrows_x /= mag
		base_dir_arrows_y /= mag

		quiver_kwargs = dict(
			pivot="middle", angles="xy",
			scale=1/arrow_step, scale_units="y",
			width=arrow_step/10, units="y",
		)

		tprint('Plotting')

		fig = plt.figure()
		gs = GridSpec(3, 4, figure=fig)

		def _plot(
				pos_or_axes,
				data: np.ndarray,
				title: str = '',
				colorbar = True,
				colorbar_loc = 'bottom',
				grid = False,
				add_range_to_title=True,
				cmap = 'viridis',
				**kwargs):

			if isinstance(pos_or_axes, Axes):
				ax = pos_or_axes
			else:
				ax = fig.add_subplot(pos_or_axes)

			im = ax.imshow(
				data,
				extent=(longitude_range[0], longitude_range[1], latitude_range[0], latitude_range[1]),
				cmap=cmap,
				**kwargs)

			if colorbar:
				fig.colorbar(im, ax=ax, location=colorbar_loc)

			if grid:
				ax.grid()

			if add_range_to_title:
				if title:
					title += ' '
				dr = data_range(data)
				title += f'[{dr[0]:.2g}, {dr[1]:.2g}]'

			if title:
				ax.set_title(title)

			return ax

		fig.suptitle(dataset_title)

		_plot(gs[0, 0], terrain.elevation_m, title='Elevation (m)', cmap='inferno')
		_plot(gs[0, 1], terrain.elevation_100km, title='Elevation blur', cmap='inferno')

		# TODO: center colormap around zero
		_plot(gs[0, 2], rain_sim.wind_dir_dot_gradient, title='Wind dir dot gradient', cmap='bwr')

		elevation_im = terrain.elevation_m.copy()
		elevation_im[elevation_im < 0] = -1000.

		ax_main = _plot(gs[1:3, 0:2], rain_sim.precipitation_cm, title='Precipitation/Wind/Elevation')
		# _plot(ax_main, terrain.elevation_m, cmap='gray', alpha=0.25, colorbar=False, add_range_to_title=False)
		_plot(ax_main, elevation_im, cmap='gray', alpha=0.25, colorbar=False, add_range_to_title=False)
		ax_main.quiver(
			arrow_locs_x, arrow_locs_y, arrows_x_norm, arrows_y_norm,
			color='white', alpha=0.25, **quiver_kwargs)

		_plot(gs[1, 2], rain_sim.base_precipitation_cm, title='Base')
		_plot(gs[1, 3], delta_precip, title='Relative to Base', cmap='Spectral',
			# norm=colors.LogNorm(vmin=1e-1, vmax=1e1),
		)

		# _plot(gs[0, 3], rain_sim.orographic_precipitation_scale, title='Orographic')

		# TODO: more

	print()
	tprint('Showing plots')
	plt.show()






if __name__ == "__main__":
	main()
