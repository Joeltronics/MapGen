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


PNW_LAT_RANGE = (40, 55)
PNW_LON_RANGE = (-135, -115)


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
		y_range[0] : y_range[1],
		x_range[0] : x_range[1]
	]

	assert len(topography_m.shape) == 2 and topography_m.shape[0] > 0 and topography_m.shape[1] > 0, f"{topography_m.shape=}"

	return topography_m.copy()


def get_test_datasets(
		*,
		earth_21600 = False,  # Note: dataset does not have Bathymetry
		earth_3600 = False,
		earth_1024 = False,
		earth_256 = False,
		earth_1024_flat = False,
		africa = False,
		north_america = False,
		south_america = False,
		pacific_northwest = False,
		circle = False,
		) -> list[dict]:

	any_earth = any([earth_3600, earth_1024, earth_256, earth_1024_flat, africa, north_america, south_america])
	any_high_res_earth = any([earth_21600, pacific_northwest])

	earth_topography_m = None
	earth_topography_highres_m = None
	if any_earth or any_high_res_earth:
		tprint('Loading Earth data...')
		from data import data
		if any_earth:
			earth_topography_m = data.get_topography()
		if any_high_res_earth:
			earth_topography_highres_m = data.get_elevation(high_res=True, ocean_nan=True)
			earth_topography_highres_m = np.nan_to_num(earth_topography_highres_m, nan=-1.)

	x = np.linspace(-4.0, 4.0, 512)
	y = np.linspace(-4.0, 4.0, 512)
	x, y = np.meshgrid(x, y)
	r = magnitude(x, y)
	circle_data = np.full((512, 512), -CIRCLE_MAG, dtype=np.float32)
	circle_data[r <= 1.0] = CIRCLE_MAG
	circle_data_latitude_range = (30, 60)

	datasets = []

	def add(title: str, source_data, flat_map=False, high_res_arrows=False, **kwargs):
		title += f', {source_data.shape[1]}x{source_data.shape[0]}'
		if flat_map:
			title += ', flat model'
		datasets.append(dict(title=title, source_data=source_data, flat_map=flat_map, high_res_arrows=high_res_arrows, **kwargs))

	if earth_21600:
		add('Earth', earth_topography_highres_m,
			flat_map=False,
			high_res_arrows=True,
		)

	if earth_3600:
		add('Earth', earth_topography_m,
			flat_map=False,
			high_res_arrows=True,
		)

	if earth_1024:
		add('Earth', earth_topography_m,
			resolution=(512, 1024),
			flat_map=False,
		)

	if earth_256:
		add('Earth', earth_topography_m,
			resolution=(128, 256),
			flat_map=False,
		)

	if earth_1024_flat:
		add('Earth', earth_topography_m,
			resolution=(512, 1024),
			flat_map=True,
		)

	if africa:
		add('Africa',
			_get_region(earth_topography_m, lat_range=AFRICA_LAT_RANGE, lon_range=AFRICA_LON_RANGE),
			latitude_range=AFRICA_LAT_RANGE,
			longitude_range=AFRICA_LON_RANGE,
			flat_map=True,
		)

	if north_america:
		add('North America',
			_get_region(earth_topography_m, lat_range=NA_LAT_RANGE, lon_range=NA_LON_RANGE),
			latitude_range=NA_LAT_RANGE,
			longitude_range=NA_LON_RANGE,
			flat_map=True,
		)

	if south_america:
		add('South America',
			_get_region(earth_topography_m, lat_range=SA_LAT_RANGE, lon_range=SA_LON_RANGE),
			latitude_range=SA_LAT_RANGE,
			longitude_range=SA_LON_RANGE,
			flat_map=True,
		)

	if pacific_northwest:
		add('Pacific Northwest',
			_get_region(earth_topography_highres_m, lat_range=PNW_LAT_RANGE, lon_range=PNW_LON_RANGE),
			latitude_range=PNW_LAT_RANGE,
			longitude_range=PNW_LON_RANGE,
			flat_map=True,
		)

	if circle:
		add('Circle test', circle_data,
			latitude_range=circle_data_latitude_range,
			flat_map=True,
		)

	return datasets
