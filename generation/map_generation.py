#!/usr/bin/env python

import argparse
from dataclasses import dataclass, field
from enum import Enum, unique
from pathlib import Path
from typing import List, Optional, Tuple, Literal, Final

import gradio as gr
from matplotlib.backends.backend_agg import FigureCanvasAgg, FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, LightSource
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator

from generation.fbm import NoiseCoords, fbm, diff_fbm, sphere_fbm, wrapped_fbm, valley_fbm
from utils.image import float_to_uint8, gradient, sphere_gradient, linear_to_gamma, gamma_to_linear, remap
from utils.map_projection import make_projection_map
from utils.numeric import data_range, rescale, max_abs, require_same_shape
from utils.utils import Parameter, tprint


"""
TODO:
- prevailing winds, and with it adiabatic rainfall & rain shadows
- tectonic continents
"""


PI: Final = np.pi
DEFAULT_MAX_PRECIPITATION_CM = 400
DEFAULT_TEMPERATURE_RANGE_C = (-10, 30)
DEGREES_C_COLDER_PER_KM_ELEVATION = 7.5
SEAWATER_FREEZING_POINT_C = -1.8


@unique
class GeneratorType(Enum):
	flat_map = '2D flat map'
	planet_2d = '2D planet'
	planet_3d = '3D planet'


@dataclass
class TopographyParams:
	elevation_steps: int
	water_amount: float = 0.5
	continent_size: float = 1.0


@dataclass
class TemperatureParams:
	pole_C: float = float(DEFAULT_TEMPERATURE_RANGE_C[0])
	equator_C: float = float(DEFAULT_TEMPERATURE_RANGE_C[1])


@dataclass
class ErosionParams:
	amount: float = 0.5
	cell_size: float = 1.0/32.0


@dataclass
class GeneratorParams:
	generator: GeneratorType
	seed: int
	topography: TopographyParams
	temperature: TemperatureParams
	erosion: ErosionParams


LAND = (0, 100, 0)  # "darkgreen"
WATER = (0, 0, 139)  # "darkblue"

GIST_EARTH = plt.get_cmap('gist_earth')
GIST_EARTH_CMAP_ZERO_POINT = 1.0 / 3.0

ELEVATION_CMAP = plt.get_cmap('seismic')
EROSION_CMAP = plt.get_cmap('inferno')
TEMPERATURE_CMAP = plt.get_cmap('coolwarm')
RAINFALL_CMAP = plt.get_cmap('YlGn')

# Land biomes
# Colors from https://en.wikipedia.org/wiki/Biome#/media/File:Climate_influence_on_terrestrial_biome.svg
TROPICAL_RAINFOREST = (0, 82, 44)
SAVANNA = (152, 167, 34)
DESERT = (201, 114, 52)
TEMPERATE_RAINFOREST = (2, 83, 109)
TEMPERATE_SEASONAL_FOREST = (40, 138, 161)
WOODLAND = (180, 125, 0)
TEMPERATE_GRASSLAND = (147, 127, 44)
BOREAL_FOREST = (91, 144, 81)
TUNDRA = (148, 169, 173)

# Ocean biomes
CONTINENTAL_SHELF = (0, 0, 177)
OCEAN = (0, 0, 139)
TRENCH = (0, 0, 77)
ICE_CAP = (210, 210, 210)

BIOME_GRID = np.array([
	[TUNDRA, TEMPERATE_GRASSLAND, TEMPERATE_GRASSLAND, DESERT],
	[TUNDRA, WOODLAND, WOODLAND, SAVANNA],
	[TUNDRA, BOREAL_FOREST, TEMPERATE_SEASONAL_FOREST, SAVANNA],
	[TUNDRA, BOREAL_FOREST, TEMPERATE_RAINFOREST, TROPICAL_RAINFOREST],
], dtype=np.uint8)


COLORMAP_RESOLUTION = 256
COLORMAP_FILENAME = Path('generation') / 'colormap.png'
COLORMAP = Image.open(COLORMAP_FILENAME).convert('RGB')  # In case of RGBA
COLORMAP = COLORMAP.resize((COLORMAP_RESOLUTION, COLORMAP_RESOLUTION), resample=Image.BILINEAR)
COLORMAP = np.array(COLORMAP, dtype=np.uint8)


OCEAN_CMAP = plt.get_cmap('ocean')


def _gradient_shading(
		im: np.ndarray,
		gradient: Tuple[np.ndarray, np.ndarray],
		strength = 0.125,
		overlay = True,
		gamma_correct = True,
		) -> np.ndarray:

	gradient_x, gradient_y = gradient

	# From above
	gradient_shading = gradient_y / max_abs(gradient_y)

	# 45 degree angle
	# gradient_shading = (gradient_y - gradient_x) / max_abs(gradient_y - gradient_x)

	gradient_shading *= strength

	im = im.astype(float) / 255.

	if gamma_correct:
		im = gamma_to_linear(im)

	if overlay:
		a = gradient_shading + 0.5
		mask = a < 0.5
		im[..., 0][mask] = 2 * a[mask] * im[..., 0][mask]
		im[..., 1][mask] = 2 * a[mask] * im[..., 1][mask]
		im[..., 2][mask] = 2 * a[mask] * im[..., 2][mask]
		mask = np.logical_not(mask)
		im[..., 0][mask] = 1 - 2 * ((1 - a[mask]) * (1 - im[..., 0][mask]))
		im[..., 1][mask] = 1 - 2 * ((1 - a[mask]) * (1 - im[..., 1][mask]))
		im[..., 2][mask] = 1 - 2 * ((1 - a[mask]) * (1 - im[..., 2][mask]))
	else:
		im[..., 0] += gradient_shading
		im[..., 1] += gradient_shading
		im[..., 2] += gradient_shading

	if gamma_correct:
		im = linear_to_gamma(im)

	im = np.clip(im, 0., 1.)
	im = np.round(im * 255.).astype(np.uint8)
	return im


def to_image(
		elevation_11: np.ndarray,
		gradient: Tuple[np.ndarray, np.ndarray],
		temperature_C: np.ndarray,
		rainfall_cm: np.ndarray,
		) -> np.ndarray:

	# TODO: take elevation into account (besides just ocean)

	assert COLORMAP.shape == (COLORMAP_RESOLUTION, COLORMAP_RESOLUTION, 3)
	colormap_temperature_idx = rescale(temperature_C, (0., 30.), (0., COLORMAP_RESOLUTION-1.), clip=True)
	colormap_rainfall_idx = rescale(rainfall_cm, (0., DEFAULT_MAX_PRECIPITATION_CM), (0., COLORMAP_RESOLUTION-1.), clip=True)
	colormap_temperature_idx = np.floor(colormap_temperature_idx).astype(int)
	colormap_rainfall_idx = np.floor(colormap_rainfall_idx).astype(int)

	im = COLORMAP[colormap_temperature_idx, colormap_rainfall_idx]

	ocean_mask = elevation_11 < 0
	land_mask = np.logical_not(ocean_mask)

	ocean_color_data = rescale(elevation_11[ocean_mask], (-1., 0.), (0., 1.), clip=True)
	ocean_color_data = np.square(ocean_color_data)  # TODO: some sort of S-curve would be better
	ocean_color_data = rescale(ocean_color_data, (0., 1.), (0.3, 0.5), clip=True)
	ocean_color_data = OCEAN_CMAP(ocean_color_data)[:, :3]  # if cmap gives RGBA, reduce to just RGB
	ocean_color_data = float_to_uint8(ocean_color_data)
	im[ocean_mask] = ocean_color_data

	# TODO: fade over 1-2 degrees (especially on land)
	# TODO: sea ice cap has no elevation and thus no gradient shading, so add a little bit of noise or something
	im[np.logical_and(ocean_mask, temperature_C < SEAWATER_FREEZING_POINT_C)] = ICE_CAP
	im[np.logical_and(land_mask, temperature_C < 0)] = ICE_CAP

	im = _gradient_shading(im, gradient)

	return im


def biome_map(elevation_11: np.ndarray, temperature_C: np.ndarray, rainfall_cm: np.ndarray):

	if not (elevation_11.shape == temperature_C.shape == rainfall_cm.shape):
		raise ValueError(f'Arrays do not have the same shape: {elevation_11.shape}, {temperature_C.shape}, {rainfall_cm.shape}')

	temperature_01 = rescale(temperature_C, DEFAULT_TEMPERATURE_RANGE_C, (0., 1.), clip=True)
	rainfall_01 = rescale(rainfall_cm, (0.0, DEFAULT_MAX_PRECIPITATION_CM), (0., 1.0), clip=True)

	rainfall_quadrant = np.clip(np.floor(rainfall_01 * 4), 0, 3).astype(np.uint8)
	temperature_quadrant = np.clip(np.floor(temperature_01 * 4), 0, 3).astype(np.uint8)
	biomes = BIOME_GRID[rainfall_quadrant, temperature_quadrant]

	biomes[elevation_11 < 0] = CONTINENTAL_SHELF
	biomes[elevation_11 < -0.1] = OCEAN
	biomes[elevation_11 < -0.75] = TRENCH
	biomes[np.logical_and(elevation_11 < 0, temperature_C < SEAWATER_FREEZING_POINT_C)] = ICE_CAP

	return biomes


def make_polar_azimuthal(equirectangular_texture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

	# TODO: generate these directly instead of remapping equirectangular

	height = equirectangular_texture.shape[0] // 2

	def _make_view(lat0, lon0, orthographic):
		xmap, ymap = make_projection_map(height, lat0=lat0, lon0=lon0, orthographic=orthographic, input_shape=equirectangular_texture.shape)
		view = remap(equirectangular_texture, xmap, ymap, nan=0, x_bounds='wrap', y_bounds='fold')
		return view

	azim_north = _make_view(lat0=90, lon0=0, orthographic=False)
	azim_south = _make_view(lat0=-90, lon0=0, orthographic=False)
	return azim_north, azim_south


def make_views(equirectangular_texture: np.ndarray) -> np.ndarray:

	# TODO: generate these directly instead of remapping equirectangular
	# TODO: until then, generate at higher resolution and downscale (for basic anti-aliasing)
	# TODO: add shadows

	height = equirectangular_texture.shape[0] // 2

	def _make_view(lat0, lon0, orthographic):
		xmap, ymap = make_projection_map(height, lat0=lat0, lon0=lon0, orthographic=orthographic, input_shape=equirectangular_texture.shape)
		view = remap(equirectangular_texture, xmap, ymap, nan=0, x_bounds='wrap', y_bounds='fold')
		return view

	debug_many_views = False

	if debug_many_views:

		view_coords = [
			(90, 0), (90, 90), (90, 180), (90, 270),
			(60, 0), (60, 90), (60, 180), (60, 270),
			(0, 0), (0, 90), (0, 180), (0, 270),
			(-90, 0), (-90, 90), (-90, 180), (-90, 270)]
		views_ortho = [_make_view(lat0=lat0, lon0=lon0, orthographic=True) for lat0, lon0 in view_coords]
		views_azim = [_make_view(lat0=lat0, lon0=lon0, orthographic=False) for lat0, lon0 in view_coords]

		n = len(view_coords) // 4

		return np.concatenate([
			np.concatenate(views_ortho[:n], axis=1),
			np.concatenate(views_azim[:n], axis=1),
			np.concatenate(views_ortho[n:2*n], axis=1),
			np.concatenate(views_azim[n:2*n], axis=1),
			np.concatenate(views_ortho[2*n:3*n], axis=1),
			np.concatenate(views_azim[2*n:3*n], axis=1),
			np.concatenate(views_ortho[3*n:], axis=1),
			np.concatenate(views_azim[3*n:], axis=1),
		], axis=0)

	else:
		lons = [0, 90, 180, 270]
		ortho_views_north = [_make_view(lat0=45, lon0=lon0, orthographic=True) for lon0 in lons]
		ortho_views_south = [_make_view(lat0=-45, lon0=lon0, orthographic=True) for lon0 in lons]
		# return np.concatenate([
		# 	np.concatenate(ortho_views_north, axis=1),
		# 	np.concatenate(ortho_views_south, axis=1)
		# ], axis=0)
		return ortho_views_north + ortho_views_south


def make_gradient_imgs(gradient_x, gradient_y, gradient_mag):
	gradient_mag = rescale(gradient_mag, data_range(gradient_mag), (0., 1.))
	gradient_img_bw = float_to_uint8(gradient_mag, bipolar=False)

	# gradient_img_color = None
	# gradient_img_color = float_to_uint8(gradient_y, bipolar=True)
	max_gradient = np.amax(np.abs(gradient_y))
	gradient_img_color = float_to_uint8(gradient_y / max_gradient, bipolar=True)

	return gradient_img_bw, gradient_img_color

	# FIXME: doesn't work, just gives noise...
	gradient_direction = np.arctan2(gradient_y, gradient_x)
	gradient_direction = rescale(gradient_direction, (-PI, PI), (0., 1.), clip=True)
	gradient_img_hsv = np.stack((
		gradient_direction,
		gradient_mag,
		gradient_mag,
	), axis=-1)
	assert len(gradient_img_hsv.shape) == 3 and (gradient_img_hsv.shape[-1] == 3)
	gradient_img_color = float_to_uint8(hsv_to_rgb(gradient_img_hsv), bipolar=False)

	# DEBUG
	gradient_img_color = float_to_uint8(np.stack(
		gradient_direction,
		gradient_direction,
		gradient_direction,
	), axis=-1)

	return gradient_img_bw, gradient_img_color


@dataclass
class Planet:
	equirectangular: Optional[np.ndarray] = None
	polar_azimuthal: Optional[Tuple[np.ndarray, np.ndarray]] = None
	views: List[np.ndarray] = field(default_factory=list)

	# TODO: latitude_data (for flat map)

	elevation_data: Optional[np.ndarray] = None
	elevation_img: Optional[np.ndarray] = None
	gradient_data: Optional[np.ndarray] = None
	gradient_img_bw: Optional[np.ndarray] = None
	gradient_img_color: Optional[np.ndarray] = None
	erosion_img: Optional[np.ndarray] = None

	temperature_C: Optional[np.ndarray] = None
	temperature_img: Optional[np.ndarray] = None

	prevailing_wind_img: Optional[np.ndarray] = None

	rainfall_cm: Optional[np.ndarray] = None
	rainfall_img: Optional[np.ndarray] = None

	water_data: Optional[np.ndarray] = None
	land_water_img: Optional[np.ndarray] = None

	biomes_img: Optional[np.ndarray] = None

	graph_figure: Optional[np.ndarray] = None

	@classmethod
	def make(
			cls,
			latitude_deg: np.ndarray,
			elevation_m: np.ndarray,
			temperature_C: np.ndarray,
			prevailing_wind: np.ndarray,
			rainfall_cm: np.ndarray,
			flat: bool,
			graph_figure=None,
			erosion=None,
			) -> 'Planet':

		if not (latitude_deg.shape == elevation_m.shape == temperature_C.shape == rainfall_cm.shape):
			raise ValueError(f'Arrays do not have the same shape: {latitude_deg.shape}, {elevation_m.shape}, {temperature_C.shape}, {rainfall_cm.shape}')

		height = elevation_m.shape[0]
		width = elevation_m.shape[1]

		land_mask = elevation_m >= 0
		water_mask = np.invert(land_mask)

		elevation_above_sea_m = np.maximum(elevation_m, 0.0)

		max_abs_elevation = max_abs(elevation_m)
		elevation_11 = rescale(elevation_m, range_in=(-max_abs_elevation, max_abs_elevation), range_out=(-1., 1.))
		temperature_01 = rescale(temperature_C)
		rainfall_01 = rescale(rainfall_cm)

		tprint('Calculating gradient')
		if flat:
			gradient_x, gradient_y = gradient(elevation_above_sea_m)
		else:
			gradient_x, gradient_y = sphere_gradient(elevation_above_sea_m, latitude_adjust=True)
		gradient_mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
		tprint('Gradient range: X [%f, %f], Y [%f, %f], mag [%f, %f]' % (
			*data_range(gradient_x), *data_range(gradient_y), *data_range(gradient_mag)
		))
		gradient_data = np.stack((gradient_x, gradient_y), axis=-1)
		gradient_img_bw, gradient_img_color = make_gradient_imgs(gradient_x=gradient_x, gradient_y=gradient_y, gradient_mag=gradient_mag)

		prevailing_wind_img = None  # TODO

		tprint('Making image')
		equirectangular = to_image(elevation_11=elevation_11, gradient=(gradient_x, gradient_y), temperature_C=temperature_C, rainfall_cm=rainfall_cm)

		tprint('Making other data views')
		elevation_img = ELEVATION_CMAP(elevation_11 * 0.5 + 0.5)
		temperature_img = TEMPERATURE_CMAP(temperature_01)
		rainfall_img = RAINFALL_CMAP(rainfall_01)

		erosion_img = None
		if erosion is not None:
			erosion_img = EROSION_CMAP(rescale(-erosion))

		land_water_img = np.zeros((height, width, 3), dtype=np.uint8)
		land_water_img[land_mask, :] = LAND
		land_water_img[water_mask, :] = WATER

		biomes_img = biome_map(elevation_11=elevation_11, temperature_C=temperature_C, rainfall_cm=rainfall_cm)

		if not flat:
			tprint('Making map projections')
			views = make_views(equirectangular)
			polar_azimuthal = make_polar_azimuthal(equirectangular)
		else:
			views = None
			polar_azimuthal = None

		# TODO: possibly also projection views for other data, like biomes

		return cls(
			equirectangular=equirectangular,
			polar_azimuthal=polar_azimuthal,
			views=views,
			elevation_data=elevation_m,
			elevation_img=elevation_img,
			gradient_data=gradient_data,
			gradient_img_bw=gradient_img_bw,
			gradient_img_color=gradient_img_color,
			erosion_img=erosion_img,
			temperature_C=temperature_C,
			temperature_img=temperature_img,
			prevailing_wind_img=prevailing_wind_img,
			rainfall_cm=rainfall_cm,
			rainfall_img=rainfall_img,
			water_data=water_mask,
			land_water_img=land_water_img,
			biomes_img=biomes_img,
			graph_figure=graph_figure,
		)


def erode_mountain_cells(elevation, mountain_cells, erosion_amount=0.5):

	# TODO: scale erosion by rainfall - more rain means more erosion

	assert np.amin(mountain_cells) >= 0
	assert np.amax(mountain_cells) <= 1

	erosion = 1.0 - mountain_cells

	erosion_amount = rescale(elevation, (0.1, 1.0), (0.0, erosion_amount), clip=True)
	return elevation - erosion_amount * erosion


def scale_elevation(elevation: np.ndarray, water_amount=0.5) -> np.ndarray:

	power_scale = False

	if water_amount == 0.5:
		elevation = np.copy(elevation)

	elif power_scale:
		power = 2.0 ** rescale(water_amount, (0.0, 1.0), (-4.0, 4.0))
		elevation = elevation * 0.5 + 0.5
		elevation = np.power(elevation, power)
		elevation = elevation * 2.0 - 1.0

	else:
		water_level = water_amount * 2.0 - 1.0

		elevation = elevation - water_level
		max_elevation = 1.0 - water_level
		min_elevation = -1.0 - water_level

		elevation[elevation >= 0] = rescale(elevation[elevation >= 0], (0.0, max_elevation), (0.0, 1.0))
		elevation[elevation < 0] = rescale(elevation[elevation < 0], (0.0, min_elevation), (0.0, -1.0))

	# For flat areas by shores, continental shelves, etc
	# TODO: steeper continental shelf dropoff
	elevation[elevation >= 0] = np.square(elevation[elevation >= 0])
	elevation[elevation < 0] = -np.square(elevation[elevation < 0])

	return elevation


def make_latitude_map(width: int, height: int, latitude_range=(-90, 90), radians=False) -> np.ndarray:

	latitude = np.linspace(latitude_range[1], latitude_range[0], num=height, endpoint=True)

	# TODO: try this instead (it will work here - just make sure it doesn't break anywhere else)
	# latitude, step = np.linspace(latitude_range[1], latitude_range[0], num=height, endpoint=False, retstep=True)
	# latitude += 0.5*step

	latitude = latitude[..., np.newaxis]
	latitude = np.repeat(latitude, repeats=width, axis=1)
	if radians:
		latitude = np.radians(latitude)
	return latitude


def make_prevailing_wind(latitude_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

	southern = (latitude_deg < 0)
	abs_latitude_deg = np.abs(latitude_deg)

	ret_x = np.zeros_like(latitude_deg)
	ret_y = np.zeros_like(latitude_deg)

	polar = abs_latitude_deg >= 60
	ferrel = np.logical_and(abs_latitude_deg >= 30, np.logical_not(polar))
	hadley = abs_latitude_deg < 30

	ret_x[polar] = TODO
	ret_y[polar] = TODO

	ret_x[ferrel] = TODO
	ret_y[ferrel] = TODO

	ret_x[hadley] = TODO
	ret_y[hadley] = TODO

	ret_y[southern] = -ret_y[southern]

	# TODO: should also scale this by where is ocean and where isn't
	# - higher speed over ocean
	# - higher speed at latitudes that are mostly ocean (e.g. southern hemisphere ocean has stronger winds than northern)

	return ret_x, ret_y


def scale_temperature(
		latitude_deg: np.ndarray,
		elevation_m: np.ndarray,
		temperature_noise: np.ndarray,
		ocean_turbulence: np.ndarray,
		strength=0.75,
		turbulence_amount=5,
		temperature_range_C=DEFAULT_TEMPERATURE_RANGE_C,
		) -> np.ndarray:

	require_same_shape(elevation_m, latitude_deg, temperature_noise, ocean_turbulence)

	latitude = np.radians(latitude_deg)

	latitude_turbulent = latitude + np.radians(turbulence_amount)*ocean_turbulence
	latitude_turbulent = np.clip(latitude_turbulent, -np.pi/2, np.pi/2)

	ocean_mask = elevation_m < 0
	land_mask = np.logical_not(ocean_mask)

	# TODO: should this use domain warping instead of interpolation? or combination of both?
	# TODO: much less variation over water, but crazier domain warping
	latitude_temp_map = np.cos(2 * latitude) * 0.5 + 0.5

	temperature_01 = temperature_noise * (1.0 - strength) + latitude_temp_map * strength

	temperature_01[ocean_mask] = np.cos(2 * latitude_turbulent[ocean_mask]) * 0.5 + 0.5

	# TODO: this is probably not the best way of going about elevation...
	# elevation_temp_map = 1.0 - np.clip(elevation, 0.0, 1.0)
	# temperature_01 *= elevation_temp_map

	temperature_C = rescale(temperature_01, (0.0, 1.0), temperature_range_C)
	temperature_C -= (DEGREES_C_COLDER_PER_KM_ELEVATION / 1000) * np.maximum(elevation_m, 0.0)

	# temperature_C[ocean_mask] = np.maximum(temperature_C[ocean_mask], SEAWATER_FREEZING_POINT_C - 0.1)

	return temperature_C


def latitude_rainfall_fn(latitude_radians):
	# Roughly based on https://commons.wikimedia.org/wiki/File:Relationship_between_latitude_vs._temperature_and_precipitation.png
	# return (np.cos(2 * latitude_radians) * 0.5 + 0.5) * (np.cos(6 * latitude_radians) * 0.5 + 0.5)
	return 0.5*(np.cos(2*latitude_radians) * 0.5 + 0.5) + 0.5*(np.cos(6*latitude_radians) * 0.5 + 0.5)
	# return 0.4*(np.cos(2*latitude_radians) * 0.5 + 0.5) + 0.6*(np.cos(6*latitude_radians) * 0.5 + 0.5)


def scale_rainfall(
		rainfall: np.ndarray,
		latitude_deg: np.ndarray,
		strength=0.75,
		max_precipitation_cm=DEFAULT_MAX_PRECIPITATION_CM,
		) -> np.ndarray:

	require_same_shape(rainfall, latitude_deg)

	latitude = np.radians(latitude_deg)

	# TODO: should this use domain warping instead of interpolation? or combination of both?

	latitude_rainfall_map = latitude_rainfall_fn(latitude)
	rainfall_01 = rainfall * (1.0 - strength) + latitude_rainfall_map * strength

	rainfall_cm = rescale(rainfall_01, (0.0, 1.0), (0., max_precipitation_cm))
	return rainfall_cm


def debug_graph(water_amount):

	# https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array

	# TODO: actual histograms of temp & rainfall by latitude
	# also histograms of elevation & gradient

	# fig = Figure()
	fig, axes = plt.subplots(2, 2)
	canvas = FigureCanvas(fig)

	latitude = np.linspace(-90, 90, num=256)
	latitude_rad = np.radians(latitude)

	latitude_temp = np.cos(2 * latitude_rad) * 0.5 + 0.5
	latitude_rainfall_map = latitude_rainfall_fn(latitude_rad)

	elevation_in = np.linspace(-1, 1, num=256)
	elevation_out = scale_elevation(elevation_in, water_amount=water_amount)
	# TODO: apply elevation colormap instead
	elevation_in_ocean = elevation_in[elevation_out <= 0]
	elevation_out_ocean = elevation_out[elevation_out <= 0]
	elevation_in_land = elevation_in[elevation_out >= 0]
	elevation_out_land = elevation_out[elevation_out >= 0]

	# TODO: alpha-scale the grid by how common these biomes actually are
	biome_im = BIOME_GRID

	ax = axes[0][0]
	ax.plot(latitude, latitude_rainfall_map, label='Rainfall')
	ax.plot(latitude, latitude_temp, label='Temperature')
	ax.set_xlabel('Latitude (degrees)')
	ax.grid()
	ax.legend()

	ax = axes[0][1]
	ax.plot(elevation_in_ocean, elevation_out_ocean, color='darkblue')
	ax.plot(elevation_in_land, elevation_out_land, color='darkgreen')
	ax.set_xlabel('Elevation in')
	ax.set_ylabel('Elevation out')
	ax.grid()

	ax = axes[1][1]
	ax.imshow(biome_im, origin='lower', extent=(0.0, 1.0, 0.0, 1.0))  # extent=(left, right, bottom, top)
	ax.set_xlabel('Temperature')
	ax.set_ylabel('Rainfall')
	# ax.grid()

	canvas.draw()

	return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))


def _generate(
		coord: NoiseCoords,
		params: GeneratorParams,
		width: int,
		height: int,
		flat: bool,
		latitude_range_deg=(-90, 90),
		fbm_func=None,
		) -> Planet:

	# TODO: use coord (then no need for fbm_func)

	tprint('Generating...', is_start=True)

	elevation_steps = params.topography.elevation_steps
	water_amount = params.topography.water_amount
	base_frequency = 1.0 / params.topography.continent_size
	temperature_range_C = (params.temperature.pole_C, params.temperature.equator_C)

	mountain_cell_base_frequency = 1.0 / params.erosion.cell_size

	latitude_deg = make_latitude_map(width=width, height=height, latitude_range=latitude_range_deg, radians=False)

	# TODO: cache previous noise, and reuse if unchanged

	# TODO: octaves should also depend on lacunarity (lacunarity = log base)
	max_resolution = max(width, height)
	octaves = int(np.ceil( np.log2(max_resolution / base_frequency) ))
	valley_octaves = int(np.ceil( np.log2(max_resolution / mountain_cell_base_frequency) ))

	def _fbm(name: str, diff_steps=1, valley=False, **kwargs):
		if 'octaves' not in kwargs:
			kwargs['octaves'] = octaves
		kwargs['seed'] = (hash(name) + params.seed)
		kwargs['normalize'] = True
		if valley:
			# TODO: pass in a valley function for the specific generator (i.e. use proper sphere noise!)
			return valley_fbm(width=width, height=height, **kwargs)
		elif diff_steps == 0:
			return fbm_func(**kwargs)
		else:
			return diff_fbm(diff_steps=diff_steps, fbm_func=fbm_func, **kwargs)

	tprint('Generating noise layers')

	temperature_noise = _fbm('temperature') * 0.5 + 0.5
	rainfall = _fbm('rainfall') * 0.5 + 0.5

	ocean_turbulence = _fbm('turbulence', base_frequency=4, gain=0.75)
	# ocean_turbulence = _fbm('turbulence', diff_steps=3)

	elevation = _fbm('elevation', diff_steps=elevation_steps, base_frequency=base_frequency)
	# TODO: pass in fbm valley function as argument, for sphere noise
	mountain_cells = _fbm('erosion_cells', valley=True, base_frequency=mountain_cell_base_frequency, octaves=valley_octaves)

	# prevailing_wind = make_prevailing_wind(latitude_deg)
	prevailing_wind = None  # TODO

	tprint('Scaling elevation/temperature/rainfall')

	elevation = scale_elevation(elevation, water_amount=water_amount)
	elevation_before_erosion = elevation
	elevation = erode_mountain_cells(elevation, mountain_cells, erosion_amount=params.erosion.amount)
	erosion = (elevation - elevation_before_erosion) / elevation

	elevation_m = rescale(elevation, (-1., 1.), (-8000., 8000.))

	temperature_C = scale_temperature(latitude_deg=latitude_deg, elevation_m=elevation_m, temperature_noise=temperature_noise, ocean_turbulence=ocean_turbulence, temperature_range_C=temperature_range_C)
	rainfall_cm = scale_rainfall(rainfall, latitude_deg=latitude_deg)

	tprint('Generating graphs')

	graph_figure = debug_graph(water_amount=water_amount)

	tprint('Assembling data into planet...')
	ret = Planet.make(
		latitude_deg=latitude_deg,
		elevation_m=elevation_m,
		erosion=erosion,
		temperature_C=temperature_C,
		prevailing_wind=prevailing_wind,
		rainfall_cm=rainfall_cm,
		graph_figure=graph_figure,
		flat=flat,
	)

	tprint('Done')

	return ret



def generate_flat_map(params: GeneratorParams, resolution: int) -> Planet:
	width = resolution
	height = resolution

	# TODO: latitude options
	# TODO: option to always make it an island

	def fbm_func(**kwargs):
		return fbm(width=width, height=height, **kwargs)

	return _generate(
		coord=NoiseCoords.xy_grid(height=height, width=width),
		params=params,
		width=width,
		height=height,
		flat=True,
		fbm_func=fbm_func,
	)


def generate_planet_2d(params: GeneratorParams, resolution: int) -> Planet:
	width = resolution
	height = resolution // 2

	# TODO: Decrease high frequency amplitudes at higher latitudes

	def fbm_func(**kwargs):
		# return fbm(width=width, height=height, **kwargs)
		return wrapped_fbm(width=width, height=height, wrap_x=True, wrap_y=False, **kwargs)

	return _generate(
		coord=NoiseCoords.cylinder_coord(height=height, width=width),
		params=params,
		width=width,
		height=height,
		flat=False,
		fbm_func=fbm_func,
	)


def generate_planet_3d(params: GeneratorParams, resolution: int) -> Planet:
	width = resolution
	height = resolution // 2

	

	def fbm_func(**kwargs):
		return sphere_fbm(height=height, **kwargs)

	return _generate(
		coord=NoiseCoords.sphere_coord(height=height, width=width),
		params=params,
		width=width,
		height=height,
		flat=False,
		fbm_func=fbm_func,
	)


def generate(params: GeneratorParams, resolution: int) -> Planet:

	if params.generator	 == GeneratorType.flat_map:
		return generate_flat_map(params, resolution=resolution)

	elif params.generator == GeneratorType.planet_2d:
		return generate_planet_2d(params, resolution=resolution)

	elif params.generator == GeneratorType.planet_3d:
		return generate_planet_3d(params, resolution=resolution)

	else:
		raise ValueError(f'Invalid {params.generator=}')
