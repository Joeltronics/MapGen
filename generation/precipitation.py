#!/usr/bin/env python3

from functools import cached_property, lru_cache
from typing import Final, Optional

import numpy as np
import scipy.interpolate

from utils.image import resize_array, gaussian_blur_map
from utils.numeric import data_range, linspace_midpoint, magnitude, rescale, require_same_shape, max_abs
from utils.utils import tprint

from .map_properties import MapProperties
from .topography import Terrain
from .winds import WindModel


BASE_PRECIPITATION_RANGE_MM: Final = (5, 4000)


# Max orographic rain is 5x the base amount, max rain shadow is 1/5 base
OROGRAPHIC_PRECIP_MAX_SCALE: Final = 5.
OROGRAPHIC_PRECIP_MAX_SCALE_LOG10: Final = np.log10(OROGRAPHIC_PRECIP_MAX_SCALE)
# Max orographic rain or rain shadow is hit when dot product = 0.005
# (Or it would be, except we use soft-clip)
# Max gradient in earth dataset at 100km scale is around 0.018
OROGRAPHIC_PRECIP_MAX_GRADIENT_DOT_PRODUCT: Final = 0.005


def latitude_precipitation_fn(latitude_radians: np.ndarray) -> np.ndarray:
	# Roughly based on https://commons.wikimedia.org/wiki/File:Relationship_between_latitude_vs._temperature_and_precipitation.png
	# return (np.cos(2 * latitude_radians) * 0.5 + 0.5) * (np.cos(6 * latitude_radians) * 0.5 + 0.5)
	return 0.5*(np.cos(2*latitude_radians) * 0.5 + 0.5) + 0.5*(np.cos(6*latitude_radians) * 0.5 + 0.5)
	# return 0.4*(np.cos(2*latitude_radians) * 0.5 + 0.5) + 0.6*(np.cos(6*latitude_radians) * 0.5 + 0.5)


# Orographic effects are weaker at tropics & arctic
LATITUDE_OROGRAPHIC_SCALE: Final = [
	(-0.01, 0.25),
	(7.5, 0.3),
	(15, 0.5),
	(22.5, 0.9),
	(30, 1.0),
	(45, 0.9),
	(60, 0.7),
	(75, 0.3),
	(90.01, 0.1),
]
_interp_latitude_orographic_scale: Final = scipy.interpolate.interp1d(
	x=np.radians([val[0] for val in LATITUDE_OROGRAPHIC_SCALE]),
	y=[val[1] for val in LATITUDE_OROGRAPHIC_SCALE],
)

def latitude_orographic_scale_fn(latitude_radians: np.ndarray) -> np.ndarray:
	return _interp_latitude_orographic_scale(np.abs(latitude_radians))

# latitude_orographic_scale_fn = None  # DEBUG


def _calculate_base_precipitation(
		noise: Optional[np.ndarray],
		latitude_deg: np.ndarray,
		noise_strength=0.25,
		precipitation_range_mm=BASE_PRECIPITATION_RANGE_MM,
		) -> np.ndarray:

	if noise is not None:
		require_same_shape(noise, latitude_deg)

	latitude = np.radians(latitude_deg)
	latitude_precip_map = latitude_precipitation_fn(latitude)

	if noise_strength > 0:
		# TODO: should this use domain warping instead of interpolation? or combination of both?
		precip_01 = noise * noise_strength + latitude_precip_map * (1.0 - noise_strength)
	else:
		precip_01 = latitude_precip_map

	precip_mm = rescale(precip_01, (0.0, 1.0), precipitation_range_mm)
	return precip_mm


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
			high_quality = False,
			):
		self._properties = map_properties
		self._terrain = terrain
		self._wind = wind
		# TODO optimization: use map_properties.latitude_column instead
		self._effective_latitude_deg = effective_latitude_deg if (effective_latitude_deg is not None) else map_properties.latitude_map
		self._noise = noise
		self._noise_strength = noise_strength
		self._high_quality = high_quality

		self._precipitation_mm = None

	@property
	def precipitation_mm(self) -> np.ndarray:
		if self._precipitation_mm is None:
			self.process()
		assert self._precipitation_mm is not None
		return self._precipitation_mm

	@cached_property
	def base_precipitation_mm(self):
		return _calculate_base_precipitation(
			noise=self._noise,
			latitude_deg=self._effective_latitude_deg,
			noise_strength=self._noise_strength,
			precipitation_range_mm=BASE_PRECIPITATION_RANGE_MM,
		)

	def _resize(self, arr: np.ndarray, force_copy=True) -> np.ndarray:
		return resize_array(arr, (self._properties.width, self._properties.height), force_copy=force_copy)

	@lru_cache
	def _wind_dir_dot_gradient_at_scale(self, scale_km, resize=True):
		"""
		:param resize: if True, can return a smaller image; if False, will always return the full size (though may still use internal resizing)
		"""

		blur_kwargs = dict(
			sigma_km=scale_km,
			flat_map=self._properties.flat,
			latitude_span=self._properties.latitude_span,
		)
		common_kwargs = dict(
			resize=True,
			truncate=(4 if self._high_quality else 2),
			resize_target_sigma_px=(8 if self._high_quality else 2),
		)

		wind_x, wind_y = self._wind.direction
		wind_x = gaussian_blur_map(wind_x, **blur_kwargs, **common_kwargs)
		wind_y = gaussian_blur_map(wind_y, **blur_kwargs, **common_kwargs)
		gradient_x, gradient_y = self._terrain.gradient_at_scale(scale_km=scale_km, **common_kwargs)

		assert wind_x.shape == wind_y.shape == gradient_x.shape == gradient_y.shape, \
			f"{wind_x.shape=}, {wind_y.shape=}, {gradient_x.shape=}, {gradient_y.shape=}"

		ret = (wind_x * gradient_x) + (wind_y * gradient_y)

		if not resize:
			ret = self._resize(ret, force_copy=False)

		return ret

	@cached_property
	def orographic_precipitation_scale_log10(self):
		"""
		Orographic precipitation (where wind is going uphill)
		"""
		# TODO: May even want smaller scale than this? 10km?

		# Rescale from (0.0, OROGRAPHIC_PRECIP_MAX_GRADIENT_DOT_PRODUCT) to (0.0, OROGRAPHIC_PRECIP_MAX_SCALE_LOG10),
		# but with soft-clip at top end
		ret = self._wind_dir_dot_gradient_at_scale(100) / OROGRAPHIC_PRECIP_MAX_GRADIENT_DOT_PRODUCT
		np.maximum(ret, 0.0, out=ret)
		np.tanh(ret, out=ret)
		ret *= OROGRAPHIC_PRECIP_MAX_SCALE_LOG10
		ret = self._resize(ret, force_copy=False)
		return ret

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

		Seems like it would give more realistic results - e.g. it would help with a small valley in a much bigger slope

		However, enabling it leads to noticeable stepping in the results
		Smaller lacunarity helps, but not enough (without going to very small)

		Try:
		- Harmonic or geometric mean? (Might be equivalent to taking mean in log domain or not)
		- Some sort of smoothmax function?
		"""
		MAX_SCALES = False

		# SCALES = [100]
		# SCALES = [1000]

		# Different lacunarities
		# SCALES = [100, 200, 500, 1000]  # approx 2
		SCALES = [100, 200, 400, 800, 1600]  # 2
		# SCALES = [100, 162, 262, 424, 685, 1109, 1794]  # golden ratio (1.618)
		# SCALES = [100, 141, 200, 316, 500, 1000, 1414]  # approx sqrt(2)

		"""
		If False:
			1. Calculate dot products at low resolution
			2. Scale each to full resolution
			3. Average & soft-clip
		If True:
			1. Calculate dot products at low resolution
			2. Scale each to the largest dot product resolution (i.e. smallest scale)
			3. Average & soft-clip
			4. Scale result to full resolution
		"""
		INTERNAL_RESOLUTION_USE_LOWEST_SCALE = True

		shadows = []
		resize_resolution = None
		for scale_km in SCALES:
			# Compensate for scale - i.e. 100 km gradient will have approx 10x range of 1000 km gradient
			mag_scale = (scale_km / 100) * (-1 / OROGRAPHIC_PRECIP_MAX_GRADIENT_DOT_PRODUCT)
			this_shadow = mag_scale * self._wind_dir_dot_gradient_at_scale(scale_km)
			# Clip to >= 0 only (will soft-clip to <= 1 later, after averaging/maxing)
			np.maximum(this_shadow, 0.0, out=this_shadow)

			if not INTERNAL_RESOLUTION_USE_LOWEST_SCALE:
				# print(f'Shadow scale {scale_km} km, resolution {(this_shadow.shape[1], this_shadow.shape[0])}'
				# 	f'scaling to final resolution {(self._properties.width, self._properties.height)}')  # DEBUG
				this_shadow = self._resize(this_shadow, force_copy=False)
			elif resize_resolution is None:
				resize_resolution = (this_shadow.shape[1], this_shadow.shape[0])
				# print(f'Shadow scale {scale_km} km, resolution {resize_resolution}')  # DEBUG
			else:
				assert resize_resolution[0] >= this_shadow.shape[1] and resize_resolution[1] >= this_shadow.shape[0], \
					f"{resize_resolution=}, {this_shadow.shape=}"
				# print(f'Shadow scale {scale_km} km, resolution {(this_shadow.shape[1], this_shadow.shape[0])}, '
				# 	f'scaling to {resize_resolution}')  # DEBUG
				this_shadow = resize_array(this_shadow, resize_resolution, force_copy=False)

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

		np.tanh(shadow, out=shadow)
		shadow *= -OROGRAPHIC_PRECIP_MAX_SCALE_LOG10

		if INTERNAL_RESOLUTION_USE_LOWEST_SCALE:
			# print(f'Scaling shadows to final resolution {(self._properties.width, self._properties.height)}')  # DEBUG
			shadow = self._resize(shadow, force_copy=False)

		return shadow


	def clear_cache(self):
		# Keep base_precipitation_mm, as the main generator still uses it
		self._wind_dir_dot_gradient_at_scale.cache_clear()
		del self.orographic_precipitation_scale_log10
		del self.rain_shadow_scale_simple_log10

	def process(self, keep_cache=False):

		# Base precipitation from latitude

		tprint("Calculating base precipitation")
		base_precipitation_mm = self.base_precipitation_mm

		# Orographic precipitation & rain shadows

		# TODO: where there's overlap, shadow should "win"

		tprint("Calculating orographic precipitation")
		scale_log10 = self.orographic_precipitation_scale_log10.copy()
		tprint("Calculating orographic shadows")
		scale_log10 += self.rain_shadow_scale_simple_log10
		tprint("Calculating final precipitation")

		if latitude_orographic_scale_fn is not None:
			scale_log10 *= latitude_orographic_scale_fn(self._properties.latitude_column_radians)

		scale = np.power(10.0, scale_log10)

		self._precipitation_mm = base_precipitation_mm * scale

		# TODO: also factor in wind vector convergence
		# Some of this is already accounted for by latitude function, but rainfall should increase anywhere wind model
		# leads toward additional convergence beyond what is expected for latitude

		if not keep_cache:
			self.clear_cache()


def main(args=None):
	import argparse

	import matplotlib
	from matplotlib import pyplot as plt
	from matplotlib import colors
	from matplotlib.gridspec import GridSpec
	from matplotlib.axes import Axes

	from .test_datasets import get_test_datasets

	# TODO: de-duplicate much this from winds.py

	MAX_ARROWS_HEIGHT_STANDARD: Final = 180 // 5
	MAX_ARROWS_HEIGHT_HIGH_RES: Final = 180 // 2

	parser = argparse.ArgumentParser()
	parser.add_argument('--test', dest='test_shapes', action='store_true', help='Only run simple test shapes')
	parser.add_argument('--multires', action='store_true', help='Run earth simulation at multiple resolutions')
	parser.add_argument('--fullres', action='store_true', help='Full-resolution Earth simulation')
	parser.add_argument('--hq', action='store_true', help='Compare low & high quality')
	args = parser.parse_args(args)

	earth_standard_res = False
	earth_lower_res = False
	earth_regions = False
	test_shapes = False
	debug_graph = False

	if args.multires:
		earth_standard_res = True
		earth_lower_res = True

	if args.test_shapes:
		test_shapes = True
		debug_graph = True

	if not any([args.fullres, args.multires, args.test_shapes]):
		earth_standard_res = True
		earth_regions = True
		test_shapes = True
		debug_graph = True

	datasets = get_test_datasets(
		earth_3600 = args.fullres,
		earth_1024 = earth_standard_res,
		earth_256 = earth_lower_res,
		earth_1024_flat = earth_lower_res,
		africa = earth_regions,
		north_america = earth_regions,
		south_america = earth_regions,
		himalaya = earth_regions,
		pacific_northwest = earth_regions,
		circle = test_shapes,
		lines = test_shapes,
	)

	if args.hq:
		datasets_out = []
		for dataset in datasets:
			dataset_lq = dict(dataset)
			dataset_hq = dict(dataset)
			dataset_lq['high_quality'] = False
			dataset_lq['title'] += ' (LQ)'
			dataset_hq['high_quality'] = True
			dataset_hq['title'] += ' (HQ)'
			datasets_out.append(dataset_lq)
			datasets_out.append(dataset_hq)
		datasets = datasets_out

	if debug_graph:
		fig, ax = plt.subplots(2, 1)
		latitude_rads = np.linspace(-0.5*np.pi, 0.5*np.pi, num=361, endpoint=True)
		base_precip = rescale(latitude_precipitation_fn(latitude_rads), (0, 1), BASE_PRECIPITATION_RANGE_MM)
		ticks = np.linspace(-90, 90, num=(180 // 15) + 1, endpoint=True)
		ax[0].plot(np.degrees(latitude_rads), base_precip)
		ax[0].set_title('Base precipitation by latitude (mm)')
		ax[0].grid()
		ax[0].set_xticks(ticks)
		ax[0].set_xlim([-90, 90])
		if latitude_orographic_scale_fn is not None:
			orographic_scale = latitude_orographic_scale_fn(latitude_rads)
		else:
			orographic_scale = np.ones_like(latitude_rads)
		ax[1].plot(np.degrees(latitude_rads), orographic_scale)
		ax[1].set_title('Strength of orographic effects by latitude (mm)')
		ax[1].grid()
		ax[1].set_xticks(ticks)
		ax[1].set_xlim([-90, 90])
		ax[1].set_ylim([0.0, 1.1])

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
			high_quality=dataset.get('high_quality', False)
		)
		rain_sim.process(keep_cache=True)

		delta_precip = rain_sim.precipitation_mm / rain_sim.base_precipitation_mm

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
					# HACK: Hard-coded for displaying large rainfall ranges (without switching to scientific)
					if 0 <= dr[0] < 1000 and 10 <= dr[1] < 10000:
						title += f'\n[{dr[0]:.2f}, {dr[1]:.1f}]'
					else:
						title += f'\n[{dr[0]:.2g}, {dr[1]:.2g}]'
				ax.set_title(title)

			return ax

		fig.suptitle(dataset_title)

		_plot(gs[0, 0], terrain.elevation_100km, title='Elevation 100km blur', cmap='inferno')
		_plot(gs[0, 1], terrain.gradient_magnitude_100km, title='100km Gradient mag', cmap='inferno')

		dot_100km = rain_sim._wind_dir_dot_gradient_at_scale(100)
		dot_1000km = rain_sim._wind_dir_dot_gradient_at_scale(1000)
		dot_max = max(max_abs(dot_100km), 10 * max_abs(dot_1000km))
		_plot(gs[0, 2], dot_100km, title='Wind dir dot 100k grad', cmap='bwr_r', vmin=-dot_max, vmax=dot_max)
		_plot(gs[0, 3], dot_1000km, title='Wind dir dot 1000k grad', cmap='bwr_r', vmin=-dot_max/10, vmax=dot_max/10)

		elevation_im = terrain.elevation_m.copy()

		max_elevation = np.amax(elevation_im)
		elevation_im[elevation_im < 0] = max_elevation / -3

		ax_main = _plot(gs[1:3, 0:2], elevation_im, cmap='gray', colorbar=False)
		_plot(
			ax_main, rain_sim.precipitation_mm,
			cmap='Spectral',
			alpha=0.75,
			title='Precipitation/Wind/Elevation',
			vmin=0, vmax=10000)
		ax_main.quiver(
			arrow_locs_x, arrow_locs_y, arrows_x_norm, arrows_y_norm,
			color='white', alpha=0.25, **quiver_kwargs)

		_plot(gs[1, 2], rain_sim.base_precipitation_mm, title='Base')

		ax_relative = _plot(gs[1, 3], elevation_im, cmap='gray', colorbar=False)
		_plot(ax_relative, delta_precip, title='Relative to Base',
			alpha=0.75,
			cmap='bwr_r',
			norm=colors.LogNorm(vmin=1/OROGRAPHIC_PRECIP_MAX_SCALE, vmax=OROGRAPHIC_PRECIP_MAX_SCALE),
		)

		_plot(gs[2, 2], rain_sim.orographic_precipitation_scale_log10, title='Orographic (log10)')
		_plot(gs[2, 3], rain_sim.rain_shadow_scale_simple_log10, title='Shadow (log10)')

	print()
	tprint('Showing plots')
	plt.show()


if __name__ == "__main__":
	main()
