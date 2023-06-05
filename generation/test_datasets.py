#!/usr/bin/env python3

from typing import Final

import numpy as np

from utils.numeric import rescale, magnitude
from utils.utils import tprint


CIRCLE_MAG: Final = 6000

AFRICA_LAT_RANGE = (-40, 40)
AFRICA_LON_RANGE = (-30, 60)

NA_LAT_RANGE = (10, 65)
NA_LON_RANGE = (-135, -45)

SA_LAT_RANGE = (-60, 20)
SA_LON_RANGE = (-90, -30)


def _get_region(
		full_topography_m: np.ndarray,
		*,
		lat_range: tuple[float, float],
		lon_range: tuple[float, float],
		):

	height, width = full_topography_m.shape

	y_range = (
		round(rescale(lat_range[1], (90, -90), (0, height))),
		round(rescale(lat_range[0], (90, -90), (0, height)))
	)

	x_range = (
		round(rescale(lon_range[0], (-180, 180), (0, width))),
		round(rescale(lon_range[1], (-180, 180), (0, width)))
	)

	topography_m = full_topography_m[
		y_range[0] : 1 + y_range[1],
		x_range[0] : 1 + x_range[1]
	]

	assert len(topography_m.shape) == 2 and topography_m.shape[0] > 0 and topography_m.shape[1] > 0, f"{topography_m.shape=}"

	return topography_m


def get_test_datasets(
		*,
		full_res_earth = False,
		lower_res_earch = False,
		earth_flat = False,
		africa = False,
		north_america = False,
		south_america = False,
		circle = False,
		) -> list[dict]:

	any_earth = any([full_res_earth, lower_res_earch, earth_flat, africa, north_america, south_america])

	earth_topography_m = None
	if any_earth:
		tprint('Loading Earth data...')
		from data import data
		earth_topography_m = data.get_topography()

	x = np.linspace(-4.0, 4.0, 512)
	y = np.linspace(-4.0, 4.0, 512)
	x, y = np.meshgrid(x, y)
	r = magnitude(x, y)
	circle_data = np.full((512, 512), -CIRCLE_MAG, dtype=np.float32)
	circle_data[r <= 1.0] = CIRCLE_MAG
	circle_data_latitude_range = (30, 60)

	datasets = []

	if full_res_earth:
		datasets += [
			dict(
				title=f'Earth, {earth_topography_m.shape[1]}x{earth_topography_m.shape[0]}',
				source_data=earth_topography_m,
				flat_map=False,
				high_res_arrows=True,
			)
		]

	if lower_res_earch:
		datasets += [
			dict(
				title='Earth, 1024x512',
				source_data=earth_topography_m,
				resolution=(512, 1024),
				flat_map=False,
			),
			dict(
				title='Earth, 256x128',
				source_data=earth_topography_m,
				resolution=(128, 256),
				flat_map=False,
			),
		]

	if earth_flat:
		datasets += [
			dict(
				title='Earth, 1024x512, flat',
				source_data=earth_topography_m,
				resolution=(512, 1024),
				flat_map=True,
			),
		]

	if africa:
		datasets.append(dict(
			title='Africa (flat)',
			source_data=_get_region(earth_topography_m, lat_range=AFRICA_LAT_RANGE, lon_range=AFRICA_LON_RANGE),
			latitude_range=AFRICA_LAT_RANGE,
			longitude_range=AFRICA_LON_RANGE,
			flat_map=True,
		))

	if north_america:
		datasets.append(dict(
			title='North America (flat)',
			source_data=_get_region(earth_topography_m, lat_range=NA_LAT_RANGE, lon_range=NA_LON_RANGE),
			latitude_range=NA_LAT_RANGE,
			longitude_range=NA_LON_RANGE,
			flat_map=True,
		))

	if south_america:
		datasets.append(dict(
			title='South America (flat)',
			source_data=_get_region(earth_topography_m, lat_range=SA_LAT_RANGE, lon_range=SA_LON_RANGE),
			latitude_range=SA_LAT_RANGE,
			longitude_range=SA_LON_RANGE,
			flat_map=True,
		))

	if circle:
		datasets += [
			dict(
				title='Circle test',
				source_data=circle_data,
				latitude_range=circle_data_latitude_range,
				flat_map=True,
			),
		]

	return datasets
