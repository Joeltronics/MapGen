#!/usr/bin/env python3

from pathlib import Path
from typing import Final

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from .precipitation import DEFAULT_PRECIPITATION_RANGE_CM
from .temperature import DEFAULT_TEMPERATURE_RANGE_C

from utils.image import linear_to_gamma, gamma_to_linear, float_to_uint8
from utils.numeric import rescale, max_abs


SEAWATER_FREEZING_POINT_C: Final = -1.8


COLORMAP_RESOLUTION: Final = 256
COLORMAP_FILENAME: Final = Path('generation') / 'colormap.png'
COLORMAP = Image.open(COLORMAP_FILENAME).convert('RGB')  # In case of RGBA
COLORMAP = COLORMAP.resize((COLORMAP_RESOLUTION, COLORMAP_RESOLUTION), resample=Image.BILINEAR)
COLORMAP = np.array(COLORMAP, dtype=np.uint8)

OCEAN_CMAP = plt.get_cmap('ocean')

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



def _gradient_shading(
		im: np.ndarray,
		gradient: tuple[np.ndarray, np.ndarray],
		strength = 0.125,
		overlay = True,
		gamma_correct = True,
		) -> np.ndarray:

	gradient_x, gradient_y = gradient

	# From above
	gradient_shading = gradient_y / -max_abs(gradient_y)

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
		gradient: tuple[np.ndarray, np.ndarray],
		temperature_C: np.ndarray,
		precipitation_cm: np.ndarray,
		) -> np.ndarray:

	# TODO: take elevation into account (besides just ocean)

	assert COLORMAP.shape == (COLORMAP_RESOLUTION, COLORMAP_RESOLUTION, 3)
	colormap_temperature_idx = rescale(temperature_C, (0., 30.), (0., COLORMAP_RESOLUTION-1.), clip=True)
	colormap_rainfall_idx = rescale(precipitation_cm, DEFAULT_PRECIPITATION_RANGE_CM, (0., COLORMAP_RESOLUTION-1.), clip=True)
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


def biome_map(elevation_11: np.ndarray, temperature_C: np.ndarray, precipitation_cm: np.ndarray):

	if not (elevation_11.shape == temperature_C.shape == precipitation_cm.shape):
		raise ValueError(f'Arrays do not have the same shape: {elevation_11.shape}, {temperature_C.shape}, {precipitation_cm.shape}')

	temperature_01 = rescale(temperature_C, DEFAULT_TEMPERATURE_RANGE_C, (0., 1.), clip=True)
	rainfall_01 = rescale(precipitation_cm, DEFAULT_PRECIPITATION_RANGE_CM, (0., 1.0), clip=True)

	rainfall_quadrant = np.clip(np.floor(rainfall_01 * 4), 0, 3).astype(np.uint8)
	temperature_quadrant = np.clip(np.floor(temperature_01 * 4), 0, 3).astype(np.uint8)
	biomes = BIOME_GRID[rainfall_quadrant, temperature_quadrant]

	biomes[elevation_11 < 0] = CONTINENTAL_SHELF
	biomes[elevation_11 < -0.1] = OCEAN
	biomes[elevation_11 < -0.75] = TRENCH
	biomes[np.logical_and(elevation_11 < 0, temperature_C < SEAWATER_FREEZING_POINT_C)] = ICE_CAP

	return biomes
