#!/usr/bin/env python3

from typing import Final, Optional, Union

import numpy as np
from numpy import sin, cos, arcsin, sqrt
from numpy import arctan2 as atan2

from utils.numeric import data_range, rescale


PI: Final = np.pi


def make_projection_map(
		height: int,
		input_shape: Union[None, tuple[int, int], tuple[int, int, int]] = None,
		lat0=90,
		lon0=0,
		orthographic=False,
		) -> tuple[np.ndarray, np.ndarray]:
	"""
	Make projection map from equirectangular to orthographic or azimuthal projection, suitable for use with remap()

	:param height: Output image height
	:param lat0: Central latitude, in degrees
	:param lon0: Central longitude, in degrees
	:param orthographic: Orthographic projection, otherwise azimuthal
	:param input_shape: Shape of input image; default is (height, 2*height)

	:returns: xmap, ymap

	:note: Assumes input image resolution (height, 2*height), output (height, height).
	For different input resolutions, scale xmap & ymap after
	"""

	if input_shape is None:
		input_height = height
		input_width = 2 * input_height
	elif len(input_shape) == 3:
		input_height, input_width, _ = input_shape
	elif len(input_shape) == 2:
		input_height, input_width = input_shape
	else:
		raise TypeError('Input shape must have dimension 2 or 3')


	lat0 = np.radians(lat0)
	lon0 = np.radians(lon0)

	sin_lat0 = sin(lat0)
	cos_lat0 = cos(lat0)

	radius = height / 2
	center = (height - 1) / 2

	yvect = np.arange(height)
	xvect = np.arange(height)

	xgrid, ygrid = np.meshgrid(xvect, yvect)

	"""
	Following math from Wolfram Mathworld & Wikipedia
	
	Orthographic
	https://mathworld.wolfram.com/OrthographicProjection.html
	https://en.wikipedia.org/wiki/Orthographic_map_projection#Mathematics

	Azimuthal Equidistant
	https://mathworld.wolfram.com/AzimuthalEquidistantProjection.html
	https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection#Mathematical_definition
	"""

	# Normalize to (-1, 1), and flip Y
	x = (xgrid - center) / radius
	y = -(ygrid - center) / radius
	r = np.sqrt(x*x + y*y)

	invalid_mask = r > 1
	r[invalid_mask] = np.nan

	if orthographic:
		c = arcsin(r)
	else:
		# Not sure why this is necessary
		x *= PI/2
		y *= PI/2
		r *= PI/2
		c = r

	cos_c = cos(c)
	sin_c = sin(c)

	if orthographic:
		lat = arcsin(
			( cos_c * sin_lat0 ) + (( y * sin_c * cos_lat0 ) / r)
		)
	else:
		# TODO: I think orthographic formula ends up the same here?
		lat = arcsin(
			( cos_c * sin_lat0 ) + (( y * sin_c * cos_lat0 ) / c)
		)

	lon_y_term = x * sin_c
	if orthographic:
		lon_x_term = r * cos_lat0 * cos_c - y * sin_lat0 * sin_c
	else:
		# TODO: I think orthographic formula ends up the same here?
		lon_x_term = c * cos_lat0 * cos_c - y * sin_lat0 * sin_c
	lon = lon0 + atan2(lon_y_term, lon_x_term)

	# Phase wrap longitude to (0, 2pi)
	# TODO: is there a numpy function for always-positive modulo?
	# TODO: this probably isn't necessary now that remap supports X wrapping
	lon = lon % (2*PI)
	lon[lon < 0] += 2*PI

	# TODO: for height, actually want to add half a pixel to each
	ymap = rescale(lat, (PI/2, -PI/2), (0, input_height-1))
	xmap = rescale(lon, (0, 2*PI), (0, input_width))

	xmap[invalid_mask] = np.nan
	ymap[invalid_mask] = np.nan

	return xmap, ymap
