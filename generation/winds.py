#!/usr/bin/env python

from functools import cached_property
from typing import Final, Optional

from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import colors
from matplotlib import pyplot as plt
from math import ceil
import numpy as np
import scipy.interpolate
import scipy.ndimage

from .map_properties import MapProperties
from .topography import Terrain

from utils.numeric import rescale, data_range, linspace_midpoint, magnitude, clip_max_vector_magnitude
from utils.image import \
	matplotlib_figure_canvas_to_image, resize_array, map_gradient, gaussian_blur_map, divergence, sphere_divergence
from utils.utils import tprint


"""
TODO:
- Calculate seasonally (even just 3 points: northern summer, equinox, southern summer)
- Use coriolis in _bend_direction_for_elevation()
- More advanced Ferrel Cell simulation
	- Summer: ocean is high pressure, land is low pressure (inverse in winter)
	- Northen hemisphere: clockwise around high pressure, anticlockwise around low pressure (inverse in Southern)

Optimization:
- Most steps need to be calculated at full resolution - anything that's blurred (which is almost everything) can be downscaled
- Latitude can be a 1D array that gets broadcasted when needed
"""


WIND_CMAP = plt.get_cmap('viridis')
IMPEDANCE_CMAP = plt.get_cmap('RdYlBu')
DIV_CMAP = plt.get_cmap('bwr')


# Direction is Cartesian angle, not screen-space (i.e. N = 90 = +Y)
# Direction is based on linear interpolation, so wrapping around won't work
BASE_WIND_STRENGTH_DIR_BY_LATITUDE: Final = [
	# Polar cells
	(90.001, 6.5, -90),  # S
	(60.5,  9.0, -135),  # SW

	# Polar jet stream
	(60.0, 9.0, 0),  # E

	# Ferrel cells
	(59.5,  9.0, 22.5),  # ENE
	(45.0,   8.0, 60),  # NE
	(31.0,  6.5, 90),  # N

	# Subtropical jet stream
	(30.0, 6.5, 0),  # E

	# Hadley cells
	(29.0,  6.5, -90),  # S
	# (15.0,   8.0, -135),  # SW
	# (-0.001, 4.5, -180),  # W
	(-0.001, 4.5, -135),  # W
]

_wind_strength_interpolator = scipy.interpolate.interp1d(
	[latitude for latitude, _, _ in BASE_WIND_STRENGTH_DIR_BY_LATITUDE],
	[strength for _, strength, _ in BASE_WIND_STRENGTH_DIR_BY_LATITUDE],
	kind='linear'
)
_wind_dir_interpolator = scipy.interpolate.interp1d(
	[latitude for latitude, _, _ in BASE_WIND_STRENGTH_DIR_BY_LATITUDE],
	[np.radians(dir) for _, _, dir in BASE_WIND_STRENGTH_DIR_BY_LATITUDE],
	kind='linear'
)


REDUCE_WIND_SPEED_ON_LAND: Final = 4.0
MAX_GRADIENT_DIRECTION_SHIFT: Final = 0.75
MAX_GRADIENT_DIRECTION_SHIFT_DOWNHILL: Final = 0.5
HILL_MAP_LAND_SCALE: Final = 2.0
HILL_MAP_ELEVATION_SCALE: Final = 1.0/6000.0
HILL_MAP_GRADIENT_SCALE: Final = 1e6


class WindModel:
	def __init__(
			self,
			map_properties: MapProperties,
			terrain: Terrain,
			effective_latitude_deg: Optional[np.ndarray] = None,
			):

		self._map_properties = map_properties
		self._terrain = terrain

		# TODO: use this
		self._effective_latitude_deg = effective_latitude_deg if (effective_latitude_deg is not None) else map_properties.latitude_column

		self._height = map_properties.height
		self._width = map_properties.width

		self._dtype = terrain.terrain_m.dtype

		# Cached properties that are not @cached_property
		self._base_magnitude_mps = None
		self._base_dir_unit_x = None
		self._base_dir_unit_y = None
		self._magnitude_mps = None
		self._dir_unit_x = None
		self._dir_unit_y = None

		# Main output
		self._prevailing_wind_x = None
		self._prevailing_wind_y = None

	# Main output

	# TODO: rename these to be more consistent

	@property
	def prevailing_wind_mps(self) -> tuple[np.ndarray, np.ndarray]:
		"""
		:returns: (wind X, wind Y) in meters per second
		"""
		if (self._prevailing_wind_x is None) or (self._prevailing_wind_y is None):
			self.process()
		assert (self._prevailing_wind_x is not None) and (self._prevailing_wind_y is not None)
		return self._prevailing_wind_x, self._prevailing_wind_y

	@property
	def magnitude_mps(self) -> np.ndarray:
		if self._magnitude_mps is None:
			self.process()
		assert self._magnitude_mps is not None
		return self._magnitude_mps

	@property
	def direction(self) -> tuple[np.ndarray, np.ndarray]:
		"""
		:returns (unit vector x, unit vector y)
		"""
		if (self._prevailing_wind_x is None) or (self._prevailing_wind_y is None):
			self.process()
		assert (self._dir_unit_x is not None) and (self._dir_unit_y is not None)
		return self._dir_unit_x, self._dir_unit_y

	# Other computed properties

	@property
	def base_magnitude_mps(self) -> np.ndarray:
		if self._base_magnitude_mps is None:
			self._make_base_winds()
		assert self._base_magnitude_mps is not None
		return self._base_magnitude_mps

	@property
	def base_direction_unit_vectors(self) -> tuple[np.ndarray, np.ndarray]:
		if (self._base_dir_unit_x is None) or (self._base_dir_unit_y is None):
			self._make_base_winds()
		assert (self._base_dir_unit_x is not None) and (self._base_dir_unit_y is not None)
		return self._base_dir_unit_x, self._base_dir_unit_y

	@property
	def base_winds_mps(self) -> tuple[np.ndarray, np.ndarray]:
		mag = self.base_magnitude_mps
		unit_x, unit_y = self.base_direction_unit_vectors
		return mag * unit_x, mag * unit_y

	@cached_property
	def land_blur(self) -> np.ndarray:
		land = self._terrain.land_mask.astype(self._dtype)
		return gaussian_blur_map(land, sigma_km=1000.0, flat_map=self._map_properties.flat, latitude_span=self._map_properties.latitude_span)

	@cached_property
	def land_blur_large(self) -> np.ndarray:
		land = self._terrain.land_mask.astype(self._dtype)
		return gaussian_blur_map(land, sigma_km=10000.0, flat_map=self._map_properties.flat, latitude_span=self._map_properties.latitude_span)

	@cached_property
	def hill_map(self) -> np.ndarray:
		return HILL_MAP_ELEVATION_SCALE * self._terrain.elevation_100km + HILL_MAP_LAND_SCALE * 0.5 * (self.land_blur + self.land_blur_large)

	@cached_property
	def hill_gradients(self) -> tuple[np.ndarray, np.ndarray]:
		return map_gradient(
			self.hill_map * HILL_MAP_GRADIENT_SCALE,
			flat_map=self._map_properties.flat,
			latitude_span=self._map_properties.latitude_span,
		)

	@cached_property
	def hill_gradients_clipped(self):
		hill_gradient_x, hill_gradient_y = self.hill_gradients
		return clip_max_vector_magnitude(hill_gradient_x, hill_gradient_y, 1.0)

	@cached_property
	def wind_dir_dot_clipped_hill_gradient(self):
		assert (self._base_dir_unit_x is not None) and (self._base_dir_unit_y is not None)
		hill_gradient_x, hill_gradient_y = self.hill_gradients_clipped
		return self._base_dir_unit_x * hill_gradient_x + self._base_dir_unit_y * hill_gradient_y

	# Freeing memory

	def clear_cache(self):
		self._base_magnitude_mps = None
		self._base_dir_unit_x = None
		self._base_dir_unit_y = None
		del self.land_blur
		del self.land_blur_large
		del self.hill_map
		del self.hill_gradients
		del self.hill_gradients_clipped
		del self.wind_dir_dot_clipped_hill_gradient

	# Processing

	def process(self, keep_cache=False) -> tuple[np.ndarray, np.ndarray]:
		"""
		:returns: (wind X, wind Y) in meters per second
		"""

		tprint("Making base winds")
		self._make_base_winds()
		tprint("Scaling wind magnitude for land")
		self._scale_magnitude_for_land()
		tprint("Bending wind direction for elevation")
		self._bend_direction_for_elevation()

		tprint("Calculating final wind")

		assert self._magnitude_mps is not None
		assert self._dir_unit_x is not None
		assert self._dir_unit_y is not None

		self._prevailing_wind_x = self._magnitude_mps * self._dir_unit_x
		self._prevailing_wind_y = self._magnitude_mps * self._dir_unit_y

		if not keep_cache:
			self.clear_cache()

		return self._prevailing_wind_x, self._prevailing_wind_y

	def _make_base_winds(self):

		assert all(val is None for val in [self._base_magnitude_mps, self._base_dir_unit_x, self._base_dir_unit_y])

		latitude_deg = self._map_properties.latitude_column

		southern = (latitude_deg < 0)
		abs_latitude_deg = np.abs(latitude_deg)

		magnitude_mps = _wind_strength_interpolator(abs_latitude_deg)
		wind_direction = _wind_dir_interpolator(abs_latitude_deg)

		dir_unit_x = np.cos(wind_direction)
		dir_unit_y = np.sin(wind_direction)

		dir_unit_y[southern] = -dir_unit_y[southern]

		magnitude_mps = np.repeat(magnitude_mps, repeats=self._width, axis=1)
		dir_unit_x = np.repeat(dir_unit_x, repeats=self._width, axis=1)
		dir_unit_y = np.repeat(dir_unit_y, repeats=self._width, axis=1)

		self._base_magnitude_mps = magnitude_mps
		self._base_dir_unit_x = dir_unit_x
		self._base_dir_unit_y = dir_unit_y

	def _scale_magnitude_for_land(self):
		assert self._magnitude_mps is None
		assert self._base_magnitude_mps is not None
		land_blur = self.land_blur
		self._magnitude_mps = self._base_magnitude_mps / rescale(land_blur, (0., 1.), (1., REDUCE_WIND_SPEED_ON_LAND))

	def _bend_direction_for_elevation(self):
		assert (self._base_dir_unit_x is not None) and (self._base_dir_unit_y is not None)
		assert (self._dir_unit_x is None) and (self._dir_unit_y is None)

		hill_gradient_x, hill_gradient_y = self.hill_gradients_clipped

		"""
		Amount we adjust wind depends on dot product:

			< 0: wind is going downhill
				- More on this later

			0: wind is perpendicular to slope
				- Don't scale

			> 0: Wind is going somewhat uphill
				- Bend it away from the hill
				- Bigger dot product = steeper slope and/or more directly uphill = more bend

			1: Wind is going pefectly uphill
				- In theory, bend the maximum amount
				- However, unit vectors cancel out, so no resulting change (assuming max_dir_shift < 1)

		Example:

			max_dir_shift = 0.5
			Input wind = 45° NE = (0.71, 0.71)
			Impedance gradient = N, steep = (0, 1)

			dot_product = 0 * 0.71 + 1 * 0.71 = 0.71

			Shifted:
				= wind - max_dir_shift * dot_product * impedance
				= (0.71, 0.71) - 0.5 * 0.71 * (0, 1)
				= (0.71, 0.71) - (0, 0.36)
				= (0.71, 0.36)

			Renormalize: (0.89, 0.45) = 26° NE

		What about negative?

		- Bend the wind *away* from downhill, not toward as you might suspect
		- But the effect is quite a bit weaker than uphill
		- This is more consistent with how aerodynamics behave. My non mechanincal engineering understanding of this:
			- The hill will have pushed away wind on the uphill side
			- This leads to lower air pressure on the downhill side, which sucks wind toward the hill
		"""

		# TODO: factor in Coriolis as well

		assert MAX_GRADIENT_DIRECTION_SHIFT_DOWNHILL >= 0

		if MAX_GRADIENT_DIRECTION_SHIFT_DOWNHILL == 0:
			wind_adjust = rescale(self.wind_dir_dot_clipped_hill_gradient, (0.0, 1.0), (0.0, MAX_GRADIENT_DIRECTION_SHIFT), clip=True)
		else:
			positive_mask = self.wind_dir_dot_clipped_hill_gradient >= 0
			negative_mask = np.logical_not(positive_mask)
			wind_adjust = self.wind_dir_dot_clipped_hill_gradient.copy()
			wind_adjust[positive_mask] *= MAX_GRADIENT_DIRECTION_SHIFT
			wind_adjust[negative_mask] *= MAX_GRADIENT_DIRECTION_SHIFT_DOWNHILL

		dir_unit_x = self._base_dir_unit_x - (wind_adjust * hill_gradient_x)
		dir_unit_y = self._base_dir_unit_y - (wind_adjust * hill_gradient_y)

		# Renormalize direction
		unit_mag = magnitude(dir_unit_x, dir_unit_y)
		assert np.amin(unit_mag) > 0
		dir_unit_x /= unit_mag
		dir_unit_y /= unit_mag

		self._dir_unit_x = dir_unit_x
		self._dir_unit_y = dir_unit_y


def make_prevailing_wind_imgs(
		# TODO: take in WindSimulation instead (or make this a member)
		prevailing_wind_mps: tuple[np.ndarray, np.ndarray],
		latitude_range: tuple[float, float],
		) -> list[np.ndarray]:

	SHOW_DIRECTION: Final = True
	SHOW_AXES: Final = False
	SCALE_MAX_WIND: Final = 12.0

	wind_x, wind_y = prevailing_wind_mps
	assert wind_x.shape == wind_y.shape
	assert len(wind_x.shape) == 2
	height, width = wind_x.shape

	longitude_span = abs(latitude_range[1] - latitude_range[0]) * width / height
	longitude_range = (-0.5*longitude_span, 0.5*longitude_span)

	scale_out = ceil(512 / height)
	assert scale_out >= 1
	width_out = round(width * scale_out)
	height_out = round(height * scale_out)

	# Determine vector resolution

	# Target cells of 10 pixels
	arrows_width = round(width_out / 10)
	arrows_height = round(height_out / 10)

	# TODO: also want max total resolution?

	# No point in arrow image having higher resolution than actual wind data
	arrows_width = min(width, arrows_width)
	arrows_height = min(height, arrows_height)

	# Magnitude

	mag = magnitude(wind_x, wind_y)
	assert np.amin(mag) > 0

	if not SHOW_DIRECTION:
		mag_img = WIND_CMAP(rescale(mag, (0., SCALE_MAX_WIND), clip=True))
		return [mag_img]

	# Direction vectors

	arrows_size = (arrows_width, arrows_height)
	arrows_x = resize_array(wind_x, new_size=arrows_size)
	arrows_y = resize_array(wind_y, new_size=arrows_size)

	# Normalized direction vectors
	# TODO: can matplotlib do this with quiver() args?

	arrow_mags = magnitude(arrows_x, arrows_y)
	assert np.amin(arrow_mags) > 0
	arrows_x_norm = arrows_x / arrow_mags
	arrows_y_norm = arrows_y / arrow_mags

	# Arrow locations

	arrow_locs_x = linspace_midpoint(longitude_range[0], longitude_range[1], arrows_x.shape[1])
	arrow_locs_y = linspace_midpoint(latitude_range[1], latitude_range[0], arrows_x.shape[0])
	arrow_step = abs(arrow_locs_y[1] - arrow_locs_y[0])
	arrow_locs_x, arrow_locs_y = np.meshgrid(arrow_locs_x, arrow_locs_y)

	# Plot with Matplotlib
	"""
	Plot 2 figures:
	
	- Magnitude as background, with normalized arrows overlaid
	- Un-normalized arrows (with no heads)

	(These show the same information, they're just different ways of displaying it)
	"""

	dpi = 100.0
	figsize = (width_out / dpi, height_out / dpi)

	# Figure 1

	fig = Figure(figsize=figsize, dpi=dpi)
	canvas = FigureCanvas(fig)

	if SHOW_AXES:
		ax = fig.gca()
		fig.tight_layout()
	else:
		ax = fig.add_axes([0., 0., 1., 1.])
		ax.set_axis_off()

	ax.imshow(mag, cmap='viridis', vmin=0., vmax=SCALE_MAX_WIND, extent=[longitude_range[0], longitude_range[1], latitude_range[0], latitude_range[1]])

	ax.quiver(
		arrow_locs_x, arrow_locs_y,
		arrows_x_norm, arrows_y_norm,
		color='white', alpha=0.75,
		pivot='middle',
		angles='xy',
		scale=1/arrow_step, scale_units='y',
		width=arrow_step/10, units='y',
	)

	fig1 = matplotlib_figure_canvas_to_image(figure=fig, canvas=canvas)

	# Figure 2

	fig = Figure(figsize=figsize, dpi=dpi)
	canvas = FigureCanvas(fig)

	if SHOW_AXES:
		ax = fig.gca()
		fig.tight_layout()
	else:
		ax = fig.add_axes([0., 0., 1., 1.])
		ax.set_axis_off()

	ax.quiver(
		arrow_locs_x, arrow_locs_y,
		arrows_x, arrows_y,
		color='black',
		pivot='tail',
		headwidth=0, headlength=0, headaxislength=0,
	)
	ax.set_xlim(longitude_range)
	ax.set_ylim(latitude_range)

	fig2 = matplotlib_figure_canvas_to_image(figure=fig, canvas=canvas)

	return [fig1, fig2]


def main(args=None):
	import argparse
	
	from matplotlib.gridspec import GridSpec
	from matplotlib.axes import Axes

	from .test_datasets import get_test_datasets

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
		north_america = earth_regions,
		south_america = earth_regions,
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

		sim = WindModel(map_properties=map_properties, terrain=terrain)
		wind_x, wind_y = sim.process(keep_cache=True)

		wind_mag = sim.magnitude_mps
		wind_x_norm, wind_y_norm = sim.direction

		tprint('Calculating divergence')

		div_func = divergence if map_properties.flat else sphere_divergence

		div = div_func(x=wind_x, y=wind_y)
		div_norm = div_func(x=wind_x_norm, y=wind_y_norm)

		tprint('Calculating arrows to plot')

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

		hill_gradient_x, hill_gradient_y = sim.hill_gradients
		hill_gradient_mag = magnitude(hill_gradient_x, hill_gradient_y)

		hill_gradient_arrows_x = resize_array(hill_gradient_x, new_size=arrows_size)
		hill_gradient_arrows_y = resize_array(hill_gradient_y, new_size=arrows_size)

		base_wind_x, base_wind_y = sim.base_winds_mps
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

		# fig, ax = plt.subplots(3, 4)

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
				cmap = 'inferno',
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

		_plot(gs[0, 0], terrain.elevation_m, title='Elevation (m)')
		_plot(gs[0, 1], sim.land_blur, title='Land blur')
		_plot(gs[0, 2], terrain.elevation_100km, title='Elevation blur')
		_plot(gs[0, 3], sim.hill_map, vmin=0.0, title='Hill map')

		ax_hill_gradient = _plot(gs[1, 2], hill_gradient_mag, title='Hill map gradient')
		_plot(gs[1, 3], sim.wind_dir_dot_clipped_hill_gradient, cmap='bwr', title='wind dot hill gradient (clipped)')

		_plot(gs[2, 2], div, cmap='bwr', title='Divergence', norm=colors.SymLogNorm(linthresh=1, vmin=-1e3, vmax=1e3))
		_plot(gs[2, 3], div_norm, cmap='bwr', title='Divergence of normalized', norm=colors.SymLogNorm(linthresh=1, vmin=-1e3, vmax=1e3))

		wind_elevation_im = terrain.elevation_m.copy()
		wind_elevation_im[topography_m < 0] = -1000.

		ax_wind = _plot(gs[1:3, 0:2], wind_mag, vmin=1.0, vmax=12.0, cmap='viridis', title='Wind')
		_plot(ax_wind, wind_elevation_im, cmap='gray', alpha=0.25, colorbar=False, add_range_to_title=False)

		# TODO: make the arrows less ugly
		ax_wind.quiver(
			arrow_locs_x, arrow_locs_y, base_dir_arrows_x, base_dir_arrows_y,
			color='red', alpha=0.75, **quiver_kwargs)

		ax_wind.quiver(
			arrow_locs_x, arrow_locs_y, arrows_x_norm, arrows_y_norm,
			color='white', alpha=0.75, **quiver_kwargs)

		ax_hill_gradient.quiver(
			arrow_locs_x, arrow_locs_y, hill_gradient_arrows_x, hill_gradient_arrows_y,
			color='white', alpha=0.5, **quiver_kwargs)

	print()
	tprint('Showing plots')
	plt.show()


if __name__ == "__main__":
	main()
