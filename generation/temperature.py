#!/usr/bin/env python3

from typing import Final

import numpy as np

from utils.numeric import rescale, require_same_shape


DEFAULT_TEMPERATURE_RANGE_C: Final = (-10, 30)
DEGREES_C_COLDER_PER_KM_ELEVATION: Final = 7.5


def calculate_temperature(
		effective_latitude_deg: np.ndarray,
		topography_m: np.ndarray,
		temperature_noise: np.ndarray,
		ocean_turbulence_noise: np.ndarray,
		noise_strength = 0.5,
		ocean_turbulence_amount_deg = 5.,
		temperature_range_C = DEFAULT_TEMPERATURE_RANGE_C,
		) -> np.ndarray:

	require_same_shape(topography_m, effective_latitude_deg, temperature_noise, ocean_turbulence_noise)

	latitude = np.radians(effective_latitude_deg)

	latitude_turbulent = latitude + np.radians(ocean_turbulence_amount_deg)*ocean_turbulence_noise
	latitude_turbulent = np.clip(latitude_turbulent, -np.pi/2, np.pi/2)

	ocean_mask = topography_m < 0
	land_mask = np.logical_not(ocean_mask)

	# TODO: should this use domain warping instead of interpolation? or combination of both?
	latitude_temp_map = np.cos(2 * latitude) * 0.5 + 0.5

	temperature_01 = temperature_noise * (1.0 - noise_strength) + latitude_temp_map * noise_strength

	# More domain warping over ocean
	temperature_01[ocean_mask] = np.cos(2 * latitude_turbulent[ocean_mask]) * 0.5 + 0.5

	# TODO: this is probably not the best way of going about elevation...
	# elevation_temp_map = 1.0 - np.clip(elevation, 0.0, 1.0)
	# temperature_01 *= elevation_temp_map

	temperature_C = rescale(temperature_01, (0.0, 1.0), temperature_range_C)
	temperature_C -= (DEGREES_C_COLDER_PER_KM_ELEVATION / 1000) * np.maximum(topography_m, 0.0)

	# temperature_C[ocean_mask] = np.maximum(temperature_C[ocean_mask], SEAWATER_FREEZING_POINT_C - 0.1)

	return temperature_C
