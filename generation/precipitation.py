#!/usr/bin/env python3

from functools import cached_property
from typing import Final, Optional

import numpy as np

from utils.image import resize_array, gaussian_blur_map
from utils.numeric import data_range, linspace_midpoint, magnitude, rescale, require_same_shape, max_abs
from utils.utils import tprint

from .map_properties import MapProperties
from .topography import Terrain
from .winds import WindModel


DEFAULT_PRECIPITATION_RANGE_CM: Final = (0.5, 400)
BASE_PRECIPITATION_RANGE_CM: Final = (0.5, 200)

# Max gradient in earth dataset at 100km scale is around 0.018
OROGRAPHIC_PRECIP_MAX_GRADIENT = 0.01
OROGRAPHIC_PRECIP_SCALE_TO_RAINFALL = 20.


# TODO: rename rainfall -> precipitation everywhere
# TODO: change everything to be in mm instead of cm


def latitude_rainfall_fn(latitude_radians: np.ndarray) -> np.ndarray:
	# Roughly based on https://commons.wikimedia.org/wiki/File:Relationship_between_latitude_vs._temperature_and_precipitation.png
	# return (np.cos(2 * latitude_radians) * 0.5 + 0.5) * (np.cos(6 * latitude_radians) * 0.5 + 0.5)
	return 0.5*(np.cos(2*latitude_radians) * 0.5 + 0.5) + 0.5*(np.cos(6*latitude_radians) * 0.5 + 0.5)
	# return 0.4*(np.cos(2*latitude_radians) * 0.5 + 0.5) + 0.6*(np.cos(6*latitude_radians) * 0.5 + 0.5)


def _calculate_rainfall_basic(
		noise: Optional[np.ndarray],
		latitude_deg: np.ndarray,
		noise_strength=0.25,
		precipitation_range_cm=DEFAULT_PRECIPITATION_RANGE_CM,
		) -> np.ndarray:

	# TODO: use rainfall model here

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
			):
		self._properties = map_properties
		self._terrain = terrain
		self._wind = wind
		# TODO optimization: use map_properties.latitude_column instead
		self._effective_latitude_deg = effective_latitude_deg if (effective_latitude_deg is not None) else map_properties.latitude_map
		self._noise = noise
		self._noise_strength = noise_strength

		self._precipitation_cm = None

	@property
	def precipitation_cm(self) -> np.ndarray:
		if self._precipitation_cm is None:
			self.process()
		assert self._precipitation_cm is not None
		return self._precipitation_cm

	@cached_property
	def base_precipitation_cm(self):
		return _calculate_rainfall_basic(
			noise=self._noise,
			latitude_deg=self._effective_latitude_deg,
			noise_strength=self._noise_strength,
			precipitation_range_cm=BASE_PRECIPITATION_RANGE_CM,
		)

	@cached_property
	def wind_dir_dot_gradient_100km(self):
		wind_x, wind_y = self._wind.direction
		gradient_x, gradient_y = self._terrain.gradient_100km
		return (wind_x * gradient_x) + (wind_y * gradient_y)

	def wind_dir_dot_gradient_at_scale(self, scale_km):
		wind_x, wind_y = self._wind.direction
		# gradient_x, gradient_y = self._terrain.gradient_1000km
		gradient_x, gradient_y = self._terrain.gradient_at_scale(scale_km=scale_km)
		return (wind_x * gradient_x) + (wind_y * gradient_y)

	# @cached_property
	# def wind_dir_dot_gradient_1000km(self):
	# 	wind_x, wind_y = self._wind.direction
	# 	# gradient_x, gradient_y = self._terrain.gradient_1000km
	# 	gradient_x, gradient_y = self._terrain.gradient_at_scale(1000)
	# 	return (wind_x * gradient_x) + (wind_y * gradient_y)

	@cached_property
	def orographic_precipitation_scale_log10(self):
		"""
		Orographic rainfall (where wind is going uphill)
		"""

		# TODO: May even want smaller scale than this? 10km?

		# Even though terrain gradient is already blurred, wind isn't and has weird convergence/divergence behavior
		# that can lead to sudden changes in dot product
		# TODO: should blur be after rescale?
		rain = gaussian_blur_map(
			self.wind_dir_dot_gradient_100km,
			sigma_km=100,
			flat_map=self._properties.flat, latitude_span=self._properties.latitude_span)

		rescale(
			rain,
			range_in=(0.0, OROGRAPHIC_PRECIP_MAX_GRADIENT),
			range_out=(0.0, np.log10(OROGRAPHIC_PRECIP_SCALE_TO_RAINFALL)),
			clip=True,
			in_place=True,
		)

		return rain

	@cached_property
	def rain_shadow_scale_simple_log10(self):
		"""
		Simple model for calculating rain shadows

		A proper model would add shadows behind orographic precipitation, but this requires iterative calculation
		which is very slow. (TODO: still want to try this!)

		Instead, we just based this on where wind is going downhill - essentially the inverse of the orographic model,
		except looking at multiple scales (generally larger)

		This isn't technically correct, and yet it ends up with fairly realistic looking results
		"""

		"""
		TODO: get MAX_SCALES working properly

		TODO: explain why we want this

		Enabling it leads to noticeable stepping in the results
		Smaller lacunarity helps, but only so much without going to very small
		
		Try:
		- Harmonic or geometric mean? (Might be equivalent to taking mean after taking inverse instead of before)
		- Some sort of smoothmax function?
		"""
		MAX_SCALES = False
		# MAX_SCALES = True

		# If using MAX_SCALES, there's noticeable stepping in the results; smaller lacunarity helps

		# SCALES = [100]
		# SCALES = [1000]

		# Approx lacunarity 2
		# SCALES = [100, 200, 500, 1000, 2000]
		SCALES = [100, 200, 500, 1000]

		# Lacunarity golden ratio (1.618)
		# Smaller lacunarity helps with stepping
		# SCALES = [100, 162, 262, 424, 685, 1109, 1794]

		# Approx lacunarity sqrt(2)
		# SCALES = [100, 141, 200, 316, 500, 1000, 1414]

		shadows = []
		for scale_km in SCALES:
			this_shadow = gaussian_blur_map(
				# self.wind_dir_dot_gradient_1000km,
				self.wind_dir_dot_gradient_at_scale(scale_km),
				sigma_km=scale_km,
				flat_map=self._properties.flat,
				latitude_span=self._properties.latitude_span,
			)
			rescale(
				this_shadow,
				range_in=(0.0, -OROGRAPHIC_PRECIP_MAX_GRADIENT * 100 / scale_km),
				range_out=(0.0, OROGRAPHIC_PRECIP_SCALE_TO_RAINFALL),
				clip=True,
				in_place=True,
			)
			shadows.append(this_shadow)

		assert len(shadows) == len(SCALES)

		if len(shadows) == 1:
			shadow = shadows[0]

		elif MAX_SCALES:
			shadow = shadows[0]
			for this_shadow in shadows[1:]:
				shadow = np.maximum(shadow, this_shadow)

		else:
			shadow = sum(shadows) / len(shadows)

		# TODO: do this in log domain in the first place

		# return 1 / (1 + shadow)
		ret_lin = 1 / (1 + shadow)
		ret_log = np.log10(ret_lin)
		return ret_log

	def clear_cache(self):
		del self.base_precipitation_cm
		del self.wind_dir_dot_gradient_100km
		# del self.wind_dir_dot_gradient_1000km
		del self.orographic_precipitation_scale_log10
		del self.rain_shadow_scale_simple_log10

	def process(self, keep_cache=False) -> np.ndarray:

		# Base precipitation from latitude

		base_precipitation_cm = self.base_precipitation_cm

		# Orographic precipitation & rain shadows

		# TODO: where there's overlap, shadow should "win"

		scale_log10 = self.orographic_precipitation_scale_log10 + self.rain_shadow_scale_simple_log10
		scale = np.power(10.0, scale_log10)

		rain_cm = base_precipitation_cm * scale

		# TODO: also factor in wind vector convergence
		# Some of this is already accounted for by latitude function, but rainfall should increase anywhere wind model
		# leads toward additional convergence beyond what is expected for latitude

		self._precipitation_cm = rain_cm

		if not keep_cache:
			self.clear_cache()

		return self._precipitation_cm


def calculate_rainfall(
		noise: np.ndarray,
		latitude_deg: np.ndarray,
		noise_strength=0.25,
		) -> np.ndarray:
	"""
	DEPRECATED
	"""

	# TODO: Remove this

	return _calculate_rainfall_basic(
		noise=noise,
		latitude_deg=latitude_deg,
		noise_strength=noise_strength,
		precipitation_range_cm=DEFAULT_PRECIPITATION_RANGE_CM,
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
	mx.add_argument('--test', dest='test_shapes_only', action='store_true', help='Only run simple test shapes')
	mx.add_argument('--multires', action='store_true', help='Run earth simulation at multiple resolutions')
	mx.add_argument('--fullres', action='store_true', help='Include full resolution simulation')
	args = parser.parse_args(args)

	MAX_ARROWS_HEIGHT_STANDARD: Final = 180 // 5
	MAX_ARROWS_HEIGHT_HIGH_RES: Final = 180 // 2

	standard_res_earth = not (args.test_shapes_only or args.fullres)
	lower_res_earth = args.multires
	earth_regions = standard_res_earth
	test_shapes = args.test_shapes_only or not args.fullres

	datasets = get_test_datasets(
		earth_3600 = args.fullres,
		earth_1024 = standard_res_earth,
		earth_256 = lower_res_earth,
		earth_1024_flat = lower_res_earth,
		africa = earth_regions,
		north_america = earth_regions,
		south_america = earth_regions,
		himalaya = earth_regions,
		pacific_northwest = earth_regions,
		circle = test_shapes,
		lines = test_shapes,
	)

	plt.figure()
	latitude_rads = np.linspace(-0.5*np.pi, 0.5*np.pi, num=361, endpoint=True)
	base_rainfall = rescale(latitude_rainfall_fn(latitude_rads), (0, 1), BASE_PRECIPITATION_RANGE_CM)
	plt.plot(np.degrees(latitude_rads), base_rainfall)
	plt.title('Base rainfall by latitude (cm)')
	plt.grid()
	plt.xticks(np.linspace(-90, 90, num=(180 // 15) + 1, endpoint=True))

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

			if title:
				if add_range_to_title:
					dr = data_range(data)
					title += f'\n[{dr[0]:.2g}, {dr[1]:.2g}]'
				ax.set_title(title)

			return ax

		fig.suptitle(dataset_title)

		# _plot(gs[0, 0], terrain.elevation_m, title='Elevation (m)', cmap='inferno')
		_plot(gs[0, 0], terrain.elevation_100km, title='Elevation 100km blur', cmap='inferno')
		_plot(gs[0, 1], terrain.gradient_magnitude_100km, title='100km Gradient mag', cmap='inferno')

		dot_1000km = rain_sim.wind_dir_dot_gradient_at_scale(1000)
		dot_max = max(max_abs(rain_sim.wind_dir_dot_gradient_100km), max_abs(dot_1000km) * 10)
		_plot(gs[0, 2], rain_sim.wind_dir_dot_gradient_100km, title='Wind dir dot 100k grad', cmap='bwr_r', vmin=-dot_max, vmax=dot_max)
		_plot(gs[0, 3], dot_1000km, title='Wind dir dot 1000k grad', cmap='bwr_r', vmin=-dot_max/10, vmax=dot_max/10)

		elevation_im = terrain.elevation_m.copy()

		max_elevation = np.amax(elevation_im)
		elevation_im[elevation_im < 0] = max_elevation / -3

		ax_main = _plot(gs[1:3, 0:2], elevation_im, cmap='gray', colorbar=False)
		_plot(
			ax_main, rain_sim.precipitation_cm,
			cmap='Spectral',
			alpha=0.75,
			title='Precipitation/Wind/Elevation')
		ax_main.quiver(
			arrow_locs_x, arrow_locs_y, arrows_x_norm, arrows_y_norm,
			color='white', alpha=0.25, **quiver_kwargs)

		_plot(gs[1, 2], rain_sim.base_precipitation_cm, title='Base')

		ax_relative = _plot(gs[1, 3], elevation_im, cmap='gray', colorbar=False)
		_plot(ax_relative, delta_precip, title='Relative to Base',
			alpha=0.75,
			cmap='bwr_r',
			norm=colors.LogNorm(vmin=0.1, vmax=10.),
		)

		_plot(gs[2, 2], rain_sim.orographic_precipitation_scale_log10, title='Orographic (log10)')
		_plot(gs[2, 3], rain_sim.rain_shadow_scale_simple_log10, title='Shadow (log10)')

	print()
	tprint('Showing plots')
	plt.show()


if __name__ == "__main__":
	main()
