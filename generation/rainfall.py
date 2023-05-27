#!/usr/bin/env python3

from typing import Final

import numpy as np

from utils.numeric import rescale, require_same_shape


DEFAULT_PRECIPITATION_RANGE_CM: Final = (0.5, 400)


def latitude_rainfall_fn(latitude_radians: np.ndarray) -> np.ndarray:
	# Roughly based on https://commons.wikimedia.org/wiki/File:Relationship_between_latitude_vs._temperature_and_precipitation.png
	# return (np.cos(2 * latitude_radians) * 0.5 + 0.5) * (np.cos(6 * latitude_radians) * 0.5 + 0.5)
	return 0.5*(np.cos(2*latitude_radians) * 0.5 + 0.5) + 0.5*(np.cos(6*latitude_radians) * 0.5 + 0.5)
	# return 0.4*(np.cos(2*latitude_radians) * 0.5 + 0.5) + 0.6*(np.cos(6*latitude_radians) * 0.5 + 0.5)


def calculate_rainfall(
		noise: np.ndarray,
		effective_latitude_deg: np.ndarray,
		noise_strength=0.25,
		precipitation_range_cm=DEFAULT_PRECIPITATION_RANGE_CM,
		) -> np.ndarray:

	require_same_shape(noise, effective_latitude_deg)

	latitude = np.radians(effective_latitude_deg)
	latitude_rainfall_map = latitude_rainfall_fn(latitude)

	# TODO: should this use domain warping instead of interpolation? or combination of both?
	rainfall_01 = noise * noise_strength + latitude_rainfall_map * (1.0 - noise_strength)

	rainfall_cm = rescale(rainfall_01, (0.0, 1.0), precipitation_range_cm)
	return rainfall_cm
