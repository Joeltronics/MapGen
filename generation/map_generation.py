#!/usr/bin/env python

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import List, Optional, Tuple, Literal, Final

from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image

from .coloring import to_image, biome_map, BIOME_GRID
from .fbm import NoiseCoords, fbm, diff_fbm, sphere_fbm, wrapped_fbm, valley_fbm
from .map_properties import MapProperties
from .precipitation import PrecipitationModel, latitude_precipitation_fn
from .temperature import calculate_temperature, DEFAULT_TEMPERATURE_RANGE_C
from .topography import Terrain, get_earth_topography, scale_topography_for_water_level, generate_topography
from .winds import WindModel, make_prevailing_wind_imgs

from utils.image import float_to_uint8, remap, matplotlib_figure_canvas_to_image, map_gradient
from utils.map_projection import make_projection_map
from utils.numeric import data_range, rescale, max_abs
from utils.utils import tprint, md5_hash


"""
TODO:
- tectonic continents
"""


PI: Final = np.pi


@unique
class GeneratorType(Enum):
	flat_map = '2D flat map'
	planet_2d = '2D planet'  # TODO: rename this - "cylinder" or something?
	planet_3d = '3D planet'


@dataclass
class TopographyParams:
	elevation_steps: int
	water_amount: float = 0.5
	continent_size: float = 1.0
	use_earth: bool = False


@dataclass
class ClimateParams:
	effective_latitude_noise_degrees: float = 5.0
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
	climate: ClimateParams
	erosion: ErosionParams

	noise_strength: float = 1.0


LAND = (0, 100, 0)  # "darkgreen"
WATER = (0, 0, 139)  # "darkblue"

GIST_EARTH = plt.get_cmap('gist_earth')
GIST_EARTH_CMAP_ZERO_POINT = 1.0 / 3.0

ELEVATION_CMAP = plt.get_cmap('seismic')
EROSION_CMAP = plt.get_cmap('inferno')
EFFECTIVE_LATITUDE_CMAP = plt.get_cmap('plasma')
TEMPERATURE_CMAP = plt.get_cmap('coolwarm')
PRECIPITATION_CMAP = plt.get_cmap('YlGn')
REL_PRECIPITATION_CMAP = plt.get_cmap('bwr_r')


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
	gradient_img_color = float_to_uint8(gradient_y / -max_gradient, bipolar=True)

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

	climate_effective_latitude_deg: Optional[np.ndarray] = None
	climate_effective_latitude_img: Optional[np.ndarray] = None

	temperature_C: Optional[np.ndarray] = None
	temperature_img: Optional[np.ndarray] = None

	prevailing_wind_data: Optional[tuple[np.ndarray, np.ndarray]] = None
	prevailing_wind_imgs: Optional[list[np.ndarray]] = None

	precipitation_cm: Optional[np.ndarray] = None
	precipitation_img: Optional[np.ndarray] = None
	rel_precipitation_img: Optional[np.ndarray] = None

	water_data: Optional[np.ndarray] = None
	land_water_img: Optional[np.ndarray] = None

	biomes_img: Optional[np.ndarray] = None

	graph_figure: Optional[np.ndarray] = None

	@classmethod
	def make(
			cls,
			map_properties: MapProperties,
			terrain: Terrain,
			climate_effective_latitude_deg: np.ndarray,
			temperature_C: np.ndarray,
			prevailing_wind_mps: np.ndarray,
			precipitation_cm: np.ndarray,
			flat_map: bool,
			base_precipitation_cm: Optional[np.ndarray] = None,
			graph_figure=None,
			) -> 'Planet':

		topography_m = terrain.terrain_m

		if not (topography_m.shape == temperature_C.shape == precipitation_cm.shape):
			raise ValueError(f'Arrays do not have the same shape: {topography_m.shape}, {temperature_C.shape}, {precipitation_cm.shape}')

		height, width = topography_m.shape

		land_mask = topography_m >= 0
		water_mask = np.invert(land_mask)

		elevation_above_sea_m = np.maximum(topography_m, 0.0)

		max_abs_elevation = max_abs(topography_m)
		elevation_11 = rescale(topography_m, range_in=(-max_abs_elevation, max_abs_elevation), range_out=(-1., 1.))
		temperature_01 = rescale(temperature_C)
		precipitation_01 = rescale(precipitation_cm)

		tprint('Calculating gradient')
		gradient_x, gradient_y = map_gradient(elevation_above_sea_m, flat_map=flat_map, latitude_span=map_properties.latitude_span)
		gradient_mag = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
		tprint('Gradient range: X [%f, %f], Y [%f, %f], mag [%f, %f]' % (
			*data_range(gradient_x), *data_range(gradient_y), *data_range(gradient_mag)
		))
		gradient_data = np.stack((gradient_x, gradient_y), axis=-1)
		gradient_img_bw, gradient_img_color = make_gradient_imgs(gradient_x=gradient_x, gradient_y=gradient_y, gradient_mag=gradient_mag)

		tprint('Making image')
		equirectangular = to_image(elevation_11=elevation_11, gradient=(gradient_x, gradient_y), temperature_C=temperature_C, precipitation_cm=precipitation_cm)

		tprint('Making other data views')

		climate_effective_latitude_img = np.abs(climate_effective_latitude_deg)
		rescale(climate_effective_latitude_img, (0., 90.), (1., 0.), clip=True, in_place=True)
		climate_effective_latitude_img = EFFECTIVE_LATITUDE_CMAP(climate_effective_latitude_img)

		elevation_img = ELEVATION_CMAP(elevation_11 * 0.5 + 0.5)
		temperature_img = TEMPERATURE_CMAP(temperature_01)
		precipitation_img = PRECIPITATION_CMAP(precipitation_01)
		prevailing_wind_imgs = make_prevailing_wind_imgs(prevailing_wind_mps, latitude_range=map_properties.latitude_range)

		rel_precipitation_img = None
		if base_precipitation_cm is not None:
			rel_precipitation = np.log10(precipitation_cm / base_precipitation_cm)
			rescale(rel_precipitation, range_in=(-1., 1.), range_out=(0., 1.), in_place=True)
			rel_precipitation_img = REL_PRECIPITATION_CMAP(rel_precipitation)

		erosion_img = EROSION_CMAP(rescale(-terrain.erosion)) if (terrain.erosion is not None) else None

		land_water_img = np.zeros((height, width, 3), dtype=np.uint8)
		land_water_img[land_mask, :] = LAND
		land_water_img[water_mask, :] = WATER

		biomes_img = biome_map(elevation_11=elevation_11, temperature_C=temperature_C, precipitation_cm=precipitation_cm)

		if not flat_map:
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
			elevation_data=topography_m,
			elevation_img=elevation_img,
			gradient_data=gradient_data,
			gradient_img_bw=gradient_img_bw,
			gradient_img_color=gradient_img_color,
			erosion_img=erosion_img,
			climate_effective_latitude_deg=climate_effective_latitude_deg,
			climate_effective_latitude_img=climate_effective_latitude_img,
			temperature_C=temperature_C,
			temperature_img=temperature_img,
			prevailing_wind_data=prevailing_wind_mps,
			prevailing_wind_imgs=prevailing_wind_imgs,
			precipitation_cm=precipitation_cm,
			precipitation_img=precipitation_img,
			rel_precipitation_img=rel_precipitation_img,
			water_data=water_mask,
			land_water_img=land_water_img,
			biomes_img=biomes_img,
			graph_figure=graph_figure,
		)


def debug_graph(water_amount):

	# TODO: actual histograms of temp & precipitation by latitude
	# also histograms of elevation & gradient

	# fig = Figure()
	fig, axes = plt.subplots(2, 2)
	canvas = FigureCanvas(fig)

	latitude = np.linspace(-90, 90, num=256)
	latitude_rad = np.radians(latitude)

	latitude_temp = np.cos(2 * latitude_rad) * 0.5 + 0.5
	latitude_precipitation_map = latitude_precipitation_fn(latitude_rad)

	elevation_in = np.linspace(-1, 1, num=512)
	elevation_out = scale_topography_for_water_level(elevation_in, water_amount=water_amount)
	# TODO: apply elevation colormap instead
	elevation_in_ocean = elevation_in[elevation_out <= 0]
	elevation_out_ocean = elevation_out[elevation_out <= 0]
	elevation_in_land = elevation_in[elevation_out >= 0]
	elevation_out_land = elevation_out[elevation_out >= 0]

	# TODO: alpha-scale the grid by how common these biomes actually are
	biome_im = BIOME_GRID

	ax = axes[0][0]
	ax.plot(latitude, latitude_precipitation_map, label='precipitation')
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
	ax.set_ylabel('precipitation')
	# ax.grid()

	return matplotlib_figure_canvas_to_image(figure=fig, canvas=canvas)


def _generate(
		params: GeneratorParams,
		map_properties: MapProperties,
		) -> Planet:

	tprint('Generating...', is_start=True)

	coord = map_properties.noise_coord
	width = map_properties.width
	height = map_properties.height
	flat_map = map_properties.flat
	latitude_range_deg = map_properties.latitude_range
	latitude_deg_2d = map_properties.latitude_map

	noise_strength = params.noise_strength
	use_noise = noise_strength > 0

	elevation_steps = params.topography.elevation_steps
	water_amount = params.topography.water_amount
	base_frequency = 1.0 / params.topography.continent_size
	temperature_range_C = (params.climate.pole_C, params.climate.equator_C)

	mountain_cell_base_frequency = 1.0 / params.erosion.cell_size

	# TODO: cache previous noise, and reuse if unchanged

	tprint('Generating noise layers')

	# TODO: octaves should also depend on lacunarity (lacunarity = log base)
	max_resolution = max(width, height)
	octaves = int(np.ceil( np.log2(max_resolution / base_frequency) ))
	valley_octaves = int(np.ceil( np.log2(max_resolution / mountain_cell_base_frequency) ))

	def _fbm(name: str, diff_steps=1, valley=False, noise_strength=noise_strength, **kwargs):

		if noise_strength <= 0:
			# TODO: do we want to return 0.5 for some noise types instead?
			return np.zeros_like(latitude_deg_2d)

		if 'octaves' not in kwargs:
			kwargs['octaves'] = octaves
		kwargs['seed'] = md5_hash(name) + params.seed

		"""
		TODO: shouldn't normalize
		- Won't work once supporting zooming in on one area
		- Can lead to slightly different results based on resolution

		But right now normalization is somewhat necessary - FBM class needs to be smarter about amplitudes
		"""
		kwargs['normalize'] = True

		if valley:
			return noise_strength * valley_fbm(coord=coord, **kwargs)
		elif diff_steps == 0:
			return noise_strength * fbm(coord=coord, **kwargs)
		else:
			return noise_strength * diff_fbm(coord=coord, diff_steps=diff_steps, **kwargs)

	# Override noise_strength here, because it's used later in calculate_temperature
	temperature_noise = _fbm('temperature', noise_strength=(1.0 if noise_strength > 0 else 0.0)) * 0.5 + 0.5
	precipitation_noise = _fbm('precipitation') * 0.5 + 0.5

	ocean_turbulence_noise = _fbm('turbulence', base_frequency=4, gain=0.75)

	latitude_noise_strength = noise_strength * params.climate.effective_latitude_noise_degrees
	effective_latitude_noise = _fbm('effective_latitude', noise_strength=latitude_noise_strength) if latitude_noise_strength > 0 else None

	topography_noise = None
	valley_noise = None
	if not params.topography.use_earth:
		topography_noise = _fbm('elevation', diff_steps=elevation_steps, base_frequency=base_frequency)
		if params.erosion.amount > 0:
			# TODO: pass in fbm valley function as argument, for sphere noise
			valley_noise = _fbm('erosion_cells', valley=True, base_frequency=mountain_cell_base_frequency, octaves=valley_octaves)

	if params.topography.use_earth:
		tprint('Loading earth topography data')
		terrain = get_earth_topography(map_properties)
	else:
		terrain = generate_topography(
			map_properties=map_properties,
			topography_noise=topography_noise,
			valley_noise=valley_noise,
			water_amount=water_amount,
			erosion_amount=params.erosion.amount
		)

	"""
	TODO: Encapsulate wind + rain + temperature into a single ClimateSimulation class
	Then make climate_effective_latitude_deg (as well as a radians version) a cached property
	(RIght now, it gets converted to radians in at least 2 differnet places)
	"""

	# "Effective latitude" for climate simulation
	# i.e. latitude + noise
	if effective_latitude_noise is None:
		climate_effective_latitude_deg = map_properties.latitude_map
		# climate_effective_latitude_deg = map_properties.latitude_column  # TODO optimization: use this (need to support everywhere)
	else:
		climate_effective_latitude_deg = effective_latitude_noise * np.cos(map_properties.latitude_map_radians)
		climate_effective_latitude_deg += map_properties.latitude_map
		assert -90. <= np.amin(climate_effective_latitude_deg) and np.amax(climate_effective_latitude_deg) <= 90.		

	"""
	TODO: precipitation should affect erosion
	i.e. calculate topography -> wind -> precipitation -> erosion -> re-calculate topography

	2 iterations is probably plenty:
	- Initial elevation estimates without erosion
	- Calculate wind & precipitation from this elevation
	- Apply erosion, scaled by precipitation
	- Recalculate wind & precipitation with new eroded elevation
	"""
	# TODO: pass in some noise for domain warping, and use this same noise for wind/temperature/precipitation
	tprint('Calculating wind')
	wind_model = WindModel(
		map_properties=map_properties,
		terrain=terrain,
		effective_latitude_deg=climate_effective_latitude_deg,
	)
	wind_model.process()
	prevailing_wind_mps = wind_model.prevailing_wind_mps

	tprint('Calculating temperature')
	temperature_C = calculate_temperature(
		effective_latitude_deg=climate_effective_latitude_deg,
		topography_m=terrain.terrain_m,
		temperature_noise=temperature_noise,
		ocean_turbulence_noise=ocean_turbulence_noise,
		temperature_range_C=temperature_range_C,
		noise_strength=(0.75*noise_strength),
	)

	tprint('Calculating precipitation')
	precipitation_model = PrecipitationModel(
		map_properties=map_properties,
		terrain=terrain,
		wind=wind_model,
		effective_latitude_deg=climate_effective_latitude_deg,
		noise=precipitation_noise,
	)
	precipitation_model.process()
	precipitation_cm = precipitation_model.precipitation_cm
	base_precipitation_cm = precipitation_model.base_precipitation_cm

	tprint('Generating graphs')
	graph_figure = debug_graph(water_amount=water_amount)

	tprint('Assembling data into planet...')
	ret = Planet.make(
		map_properties=map_properties,
		terrain=terrain,
		climate_effective_latitude_deg=climate_effective_latitude_deg,
		temperature_C=temperature_C,
		prevailing_wind_mps=prevailing_wind_mps,
		precipitation_cm=precipitation_cm,
		base_precipitation_cm=base_precipitation_cm,
		graph_figure=graph_figure,
		flat_map=flat_map,
	)

	tprint('Done')

	return ret



def generate_flat_map(params: GeneratorParams, resolution: int) -> Planet:
	width = resolution
	height = (resolution // 2) if params.topography.use_earth else resolution

	# TODO: latitude options
	# TODO: option to always make it an island

	noise_coord = NoiseCoords.make_xy_grid(height=height, width=width)
	map_props = MapProperties(flat=True, height=height, width=width, noise_coord=noise_coord)
	return _generate(params=params, map_properties=map_props)


def generate_planet_2d(params: GeneratorParams, resolution: int) -> Planet:
	width = resolution
	height = resolution // 2

	# TODO: Decrease high frequency amplitudes at higher latitudes

	noise_coord = NoiseCoords.make_cylinder(height=height, width=width)
	map_props = MapProperties(flat=False, height=height, width=width, noise_coord=noise_coord)
	return _generate(params=params, map_properties=map_props)


def generate_planet_3d(params: GeneratorParams, resolution: int) -> Planet:
	width = resolution
	height = resolution // 2

	noise_coord = NoiseCoords.make_sphere(height=height, width=width)
	map_props = MapProperties(flat=False, height=height, width=width, noise_coord=noise_coord)
	return _generate(params=params, map_properties=map_props)


def generate(params: GeneratorParams, resolution: int) -> Planet:

	if params.generator	 == GeneratorType.flat_map:
		return generate_flat_map(params, resolution=resolution)

	elif params.generator == GeneratorType.planet_2d:
		return generate_planet_2d(params, resolution=resolution)

	elif params.generator == GeneratorType.planet_3d:
		return generate_planet_3d(params, resolution=resolution)

	else:
		raise ValueError(f'Invalid {params.generator=}')
