#!/usr/bin/env python3

from math import isclose, pi
from typing import Final
import warnings

import numpy as np
from numpy import sin, cos, radians, degrees

from utils.numeric import FloatOrArrayT, rescale, require_same_shape
from utils.consts import EARTH_AXIAL_TILT_DEGREES, C_TO_K

"""
TODO: Things this model does not currently account for:
- Sunlight going through more atmosphere when at a shallower angle
- Albedo changing throughout the year
"""

"""
Approximate real values

Earth blackbody temperature: 248.572 K = -24.578 C
Average actual temperature is about 288 K = 15 C

North Pole (mean surface air temperature):
- Avg: -18
- Jan: -35
- Jul: 1

South Pole (mean surface air temperature) - value, plus corrected for elevation:
(Elevation 2,835 m - South Pole is very dry so using higher lapse rate, delta 25)
- Avg: -45 -> -20
- Jan: -28 -> -2
- Jul: -56 -> -31

Ocean at equator:
- Avg: 27 = 300 K

Total annual insolation (calculated with functions below)
Poles ~0.10
Equator ~0.31

So the total insolation at the poles is around 1/3 that of the equator, but the average temperature is around 5/6 (in K)
So temperature is proportional to (0.75 + 0.25 * I), where I is insolation relative to the equator
Or to put this in actual units:
temp_K = temp_equator_K * (0.75 + (0.25 * insolation / insolation_equator)
temp_K = 300 * (0.75 + (0.25/0.31 * insolation)
"""
DEFAULT_TEMPERATURE_RANGE_C: Final = (-18, 27)

# Lapse rate
# TODO: actual average is about 6.5, but varies with moisture (9.8 C in completely dry air)
DEGREES_C_COLDER_PER_KM_ELEVATION: Final = 7.5

SEASONAL_TEMPERATURE_VARIATION_OCEAN = 0.2
SEASONAL_TEMPERATURE_VARIATION_LAND = 0.3

# Turbulence amount (at 100% noise_strength)
OCEAN_TURBULENCE_AMOUNT_DEG: Final = 10

TEMPERATURE_NOISE_AMOUNT_LAND_C: Final = 10
TEMPERATURE_NOISE_AMOUNT_OCEAN_C: Final = 2.5

# Approximation from http://www-das.uwyo.edu/~geerts/cwx/notes/chap16/geo_clim.html
# TODO: try the better approximation that accounts for distance downwind from ocean
SEASONAL_RANGE_C_PER_DEGREE_LATITUDE: Final = 0.4


def get_insolation_at_time(*,
		latitude_rads: FloatOrArrayT,
		declination_rads: FloatOrArrayT,
		hour_angle_rads: FloatOrArrayT,
		negative = False,
		) -> FloatOrArrayT:

	# https://en.wikipedia.org/wiki/Solar_zenith_angle#Formula
	# insolation = cos(zenith) = sin(elevation) =
	# 	sin(latitude) * sin(declination) + cos(latitude) * cos(declination) * cos(hour)

	ret = (
		np.sin(latitude_rads) * np.sin(declination_rads) +
		np.cos(latitude_rads) * np.cos(declination_rads) * np.cos(hour_angle_rads)
	)
	if not negative:
		ret = np.maximum(0, ret)
	return ret


def get_insolation_over_day(*,
		latitude_rads: FloatOrArrayT,
		declination_rads: FloatOrArrayT,
		) -> FloatOrArrayT:

	if isinstance(latitude_rads, np.ndarray) and isinstance(declination_rads, np.ndarray):
		if latitude_rads.shape != declination_rads.shape:
			raise ValueError(f'Arrays do not have same shape: {latitude_rads.shape} != {declination_rads.shape}')

	if False:
		# Calculate with discrete sums

		PRECISION = 512

		hour = np.linspace(0.0, 24.0, num=PRECISION, endpoint=False)
		hour_angle_rads = radians(hour * (360 / 24) - 180)

		if isinstance(latitude_rads, np.ndarray):
			latitude_rads = latitude_rads[..., np.newaxis]

		if isinstance(declination_rads, np.ndarray):
			declination_rads = declination_rads[..., np.newaxis]

		insolation_by_time = get_insolation_at_time(
			latitude_rads=latitude_rads,
			declination_rads=declination_rads,
			hour_angle_rads=hour_angle_rads,
			negative=False,
		)

		insolation_over_day = np.sum(insolation_by_time, axis=-1) / insolation_by_time.shape[-1]

	else:
		# Calculate definite integral

		"""
		Formulas:
			insolation_signed = sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(hour)
			insolation = max(0, insolation_signed)
		"""
		# Constants that do not need to be integrated over:
		S = np.sin(latitude_rads) * np.sin(declination_rads)
		C = np.cos(latitude_rads) * np.cos(declination_rads)

		"""
		Formulas with substitution:
			insolation_signed = C * cos(hour) + S
			insolation = max(0, C * cos(hour) + S)

		To take integral of insolation despite the max() operation:
			Determine the region where insolation_signed >= 0 (i.e. calculate sunrise & sunset times)
			and take definite integral of insolation_signed over that range

		Sunrise/sunset formulas:
			0 = C * cos(hour) + S
			hour = arccos(-S/C)
		"""

		with np.errstate(divide='ignore'):
			s_over_c = -S/C
		s_over_c = np.clip(s_over_c, -1.0, 1.0)
		sunset = np.arccos(s_over_c)

		"""
		Over the region of +/- sunset_hour_angle_rads, insolation == insolation_signed, so:
			insolation = C * cos(hour) + S
		Indefinite Integral:
			integral_insolation = C * sin(hour) + S * hour + constant
		Definite integral from -sunset to +sunset:
			integral_insolation = C * (sin(sunset) - sin(-sunset)) + S * (sunset - (-sunset))
			integral_insolation = 2 * (C * sin(sunset) + S * sunset)

		But this is in radians, we want days; divide by 2pi:
		"""
		insolation_over_day = (C * np.sin(sunset) + S * sunset) / pi

	return insolation_over_day


def get_insolation_over_year(*,
		latitude_rads: FloatOrArrayT,
		axial_tilt_deg: float = EARTH_AXIAL_TILT_DEGREES,
		) -> FloatOrArrayT:

	# TODO: calculate this with 2D definite integral
	# (Much more complicated than 1D case above - would have to integrate a non-rectangular 2D region)

	PRECISION = 366 // 2  # 1/2 year
	time_of_year = np.linspace(-0.5, 0.5, PRECISION, endpoint=True)
	declination_rads = np.sin(2.0 * time_of_year * radians(axial_tilt_deg))

	orig_shape = None
	if isinstance(latitude_rads, np.ndarray):
		orig_shape = latitude_rads.shape
		declination_rads, latitude_rads = np.meshgrid(declination_rads, latitude_rads)

	insolation_over_day = get_insolation_over_day(
		latitude_rads=latitude_rads,
		declination_rads=declination_rads,
	)

	insolation_over_year = np.sum(insolation_over_day, axis=-1) / insolation_over_day.shape[-1]

	if isinstance(insolation_over_year, np.ndarray):
		assert orig_shape is not None
		assert insolation_over_year.shape == orig_shape

	return insolation_over_year


EARTH_INSOLATION_EQUATOR: Final = get_insolation_over_year(latitude_rads=0)

def _insolation_to_temperature(
		insolation: np.ndarray,
		equator_average_temperature_C = DEFAULT_EQUATOR_AVERAGE_TEMPERATURE_C,
		):
	equator_K = equator_average_temperature_C + C_TO_K
	temperature_K = equator_K * (0.75 + (0.25/EARTH_INSOLATION_EQUATOR)*insolation)
	temperature_C = temperature_K - C_TO_K
	return temperature_C



def _calculate_temperature(*,
		latitude_deg: np.ndarray,
		season_phase: float = 0.0,
		) -> np.ndarray:

	if not (-1 <= season_phase <= 1):
		raise ValueError(f'season_phase must be in range [-1, 1] (value: {season_phase})')

	average_C = TODO

	if abs(season_phase) < 1e-9:
		return average_C

	return average_C + latitude_deg * (SEASONAL_RANGE_C_PER_DEGREE_LATITUDE * 0.5 * season_phase)


def calculate_average_annual_temperature(*,
		effective_latitude_deg: np.ndarray,
		topography_m: np.ndarray,
		temperature_noise: np.ndarray,
		ocean_turbulence_noise: np.ndarray,
		axial_tilt_deg: float = EARTH_AXIAL_TILT_DEGREES,
		noise_strength = 0.5,
		equator_average_temperature_C = DEFAULT_EQUATOR_AVERAGE_TEMPERATURE_C,
		) -> np.ndarray:

	LUT_SIZE_FULL = 32
	MAX_SIZE_WITHOUT_LUT = 1024

	require_same_shape(topography_m, effective_latitude_deg, temperature_noise, ocean_turbulence_noise)

	ocean_mask = topography_m < 0

	# Turbulence noise gets added to ocean before calculating insolation (temperature noise will be added later)
	turbulence_strength_rads = radians(OCEAN_TURBULENCE_AMOUNT_DEG) * noise_strength
	input_latitude_rads = radians(effective_latitude_deg)
	input_latitude_rads[ocean_mask] += turbulence_strength_rads * ocean_turbulence_noise[ocean_mask]

	# At poles, clip rather than wrapping around
	np.clip(input_latitude_rads, -0.5*pi, 0.5*pi, out=input_latitude_rads)

	if effective_latitude_deg.size <= MAX_SIZE_WITHOUT_LUT:
		insolation = get_insolation_over_year(
			latitude_rads=input_latitude_rads,
			axial_tilt_deg=axial_tilt_deg,
		)

	else:
		# For big arrays, interpolate from smaller lookup table

		# This should be symmetric over the whole year, so we can use abs
		# (Could have done this in above case too, but there would have been no benefit)
		np.abs(input_latitude_rads, out=input_latitude_rads)

		min_latitude = input_latitude_rads.min()
		max_latitude = input_latitude_rads.max()
		latitude_span = max_latitude - min_latitude
		assert 0 <= latitude_span < (0.5*pi + 1e-6), f"{min_latitude=}, {max_latitude=}, {latitude_span=}"
		num = max(2, round(latitude_span / (0.5*pi) * LUT_SIZE_FULL))
		lut_latitude = np.linspace(min_latitude, max_latitude, num=num, endpoint=True)
		lut_insolation = get_insolation_over_year(
			latitude_rads=lut_latitude,
			axial_tilt_deg=axial_tilt_deg,
		)
		insolation = np.interp(input_latitude_rads, lut_latitude, lut_insolation)

	# Convert insolation to temperature
	temperature_C = _insolation_to_temperature(insolation, equator_average_temperature_C=equator_average_temperature_C)

	# Adjust for elevation
	topography_above_zero_m = np.maximum(topography_m, 0.0)
	temperature_C -= (DEGREES_C_COLDER_PER_KM_ELEVATION / 1000) * topography_above_zero_m

	# Add temperature noise
	temperature_C += temperature_noise * np.where(
		ocean_mask,
		TEMPERATURE_NOISE_AMOUNT_OCEAN_C * noise_strength,
		TEMPERATURE_NOISE_AMOUNT_LAND_C * noise_strength,
	)

	return temperature_C


def calculate_seasonal_temperature(*,
		effective_latitude_deg: np.ndarray,
		topography_m: np.ndarray,
		temperature_noise: np.ndarray,
		ocean_turbulence_noise: np.ndarray,
		annual_average_temperature: np.ndarray,
		declination_deg: float,
		noise_strength = 0.5,
		equator_average_temperature_C = DEFAULT_EQUATOR_AVERAGE_TEMPERATURE_C,
		) -> np.ndarray:

	require_same_shape(topography_m, effective_latitude_deg, temperature_noise, ocean_turbulence_noise, annual_average_temperature)

	ocean_mask = topography_m < 0

	# Turbulence noise gets added to ocean before calculating insolation (temperature noise will be added later)
	turbulence_strength_rads = radians(OCEAN_TURBULENCE_AMOUNT_DEG) * noise_strength
	input_latitude_rads = radians(effective_latitude_deg)
	input_latitude_rads[ocean_mask] += turbulence_strength_rads * ocean_turbulence_noise[ocean_mask]

	# At poles, clip rather than wrapping around
	np.clip(input_latitude_rads, -0.5*pi, 0.5*pi, out=input_latitude_rads)

	insolation = get_insolation_over_day(
		latitude_rads=input_latitude_rads,
		declination_rads=radians(declination_deg),
	)

	# Convert insolation to temperature
	temperature_C = _insolation_to_temperature(insolation, equator_average_temperature_C=equator_average_temperature_C)

	# Adjust for elevation
	topography_above_zero_m = np.maximum(topography_m, 0.0)
	temperature_C -= (DEGREES_C_COLDER_PER_KM_ELEVATION / 1000) * topography_above_zero_m

	# Mix with annual average temperature
	temperature_C = np.where(
		ocean_mask,
		SEASONAL_TEMPERATURE_VARIATION_OCEAN * temperature_C + (1.0 - SEASONAL_TEMPERATURE_VARIATION_OCEAN) * annual_average_temperature,
		SEASONAL_TEMPERATURE_VARIATION_LAND * temperature_C + (1.0 - SEASONAL_TEMPERATURE_VARIATION_LAND) * annual_average_temperature,
	)

	# Add temperature noise
	temperature_C += temperature_noise * np.where(
		ocean_mask,
		TEMPERATURE_NOISE_AMOUNT_OCEAN_C * noise_strength,
		TEMPERATURE_NOISE_AMOUNT_LAND_C * noise_strength,
	)

	return temperature_C


def _plot_sunlight_at_noon():
	from matplotlib import pyplot as plt

	ANGLE_STEP_MAJOR = 15
	ANGLE_STEP_MINOR = 5

	latitude_deg = np.linspace(-90, 90, num=361, endpoint=True)

	fig, (ax1, ax2)  = plt.subplots(1, 2)
	fig.suptitle('Sunlight at noon')

	for declination_deg, label in [
		(EARTH_AXIAL_TILT_DEGREES, 'Dec 21'),
		(0, 'Mar/Sep 21'),
		(-EARTH_AXIAL_TILT_DEGREES, 'Jun 21'),
	]:
		effective_latitude = radians(latitude_deg + declination_deg)  # 90 - declination
		elevation = 0.5*pi - np.abs(effective_latitude)
		elevation_deg = degrees(elevation)
		insolation = np.sin(np.maximum(elevation, 0))

		ax1.plot(elevation_deg, latitude_deg, label=label)
		ax2.plot(insolation, latitude_deg, label=label)

	for ax in [ax1, ax2]:
		ax.grid()
		ax.legend()
		ax.set_ylabel('Latitude')
		ax.set_yticks(np.arange(-90, 90 + ANGLE_STEP_MINOR, ANGLE_STEP_MINOR), minor=True)
		ax.set_yticks(np.arange(-90, 90 + ANGLE_STEP_MAJOR, ANGLE_STEP_MAJOR))

	ax1.set_xlabel('Sun Elevation')
	ax1.set_xticks(np.arange(-20, 90 + ANGLE_STEP_MINOR, ANGLE_STEP_MINOR), minor=True)
	ax1.set_xticks([-EARTH_AXIAL_TILT_DEGREES] + list(range(0, 90 + ANGLE_STEP_MAJOR, ANGLE_STEP_MAJOR)))
	ax1.set_xlim([-25, 92])

	ax2.set_xlabel('Insolation (direct)')


def _plot_sunlight_at_time():
	from matplotlib import pyplot as plt

	LATITUDES_DEG = [90, 90 - EARTH_AXIAL_TILT_DEGREES, 45, EARTH_AXIAL_TILT_DEGREES, 0]

	fig, axes = plt.subplots(len(LATITUDES_DEG), 2, sharex=True)
	fig.suptitle('Sunlight by time')

	hour = np.linspace(0.0, 24.0, num=512, endpoint=False)
	hour_angle = radians(hour * (360 / 24) - 180)

	for idx, latitude_deg in enumerate(LATITUDES_DEG):
 
		ax1 = axes[idx, 0]
		ax2 = axes[idx, 1]

		latitude = radians(latitude_deg)

		for declination_deg, label in [
			(-EARTH_AXIAL_TILT_DEGREES, 'Dec 21'),
			(0, 'Mar/Sep 21'),
			(EARTH_AXIAL_TILT_DEGREES, 'Jun 21'),
		]:
			declination = radians(declination_deg)
			# insolation_signed = sin(latitude) * sin(declination) + cos(latitude) * cos(declination) * cos(hour_angle)
			insolation_signed = get_insolation_at_time(
				latitude_rads=latitude, declination_rads=declination, hour_angle_rads=hour_angle, negative=True)
			insolation = np.maximum(insolation_signed, 0)

			elevation = np.arcsin(insolation_signed)
			elevation_deg = degrees(elevation)

			ax1.plot(hour, elevation_deg, label=label)
			ax2.plot(hour, insolation, label=label)

		ax1.grid()
		ax1.set_ylabel(f'{latitude_deg}° N')

		ax1.set_yticks(range(0, 90 + 15, 15))
		ax1.set_ylim([0, 90])
		# ax.set_yticks(range(-90, 90 + 30, 30))
		# ax.set_ylim([-90, 90])

		ax2.set_ylim([0., 1.])
		ax2.grid()

	axes[0, 0].legend()
	axes[0, 0].set_title('Elevation')
	axes[0, 1].set_title('Insolation')

	for col in [0, 1]:
		axes[-1, col].set_xlabel('Hour')
		axes[-1, col].set_xticks(np.arange(0, 24 + 1, 1), minor=True)
		axes[-1, col].set_xticks(np.arange(0, 24 + 6, 6))


def _double_cosine_fit(vals: np.ndarray, theta_rads: np.ndarray) -> np.ndarray:
	import scipy.optimize

	def f(theta_rads, c0, c1, c2):
		return c0 + c1 * np.cos(theta_rads) + c2 * np.cos(2.0 * theta_rads)

	assert isinstance(vals, np.ndarray) and vals.ndim == 1
	vals_mid = vals[vals.size // 2]
	vals_end = 0.5*(vals[0] + vals[-1])
	avg = 0.5 * (vals_mid + vals_end)
	delta = 0.5 * (vals_mid - vals_end)

	popt, _ = scipy.optimize.curve_fit(f, theta_rads, vals, p0=[avg, 0.0, delta])

	return f(theta_rads, *popt)


def _plot_total_sunlight_3d(axial_tilt_deg=EARTH_AXIAL_TILT_DEGREES):
	from matplotlib import pyplot as plt

	ANGLE_STEP_MAJOR = 15
	ANGLE_STEP_MINOR = 5

	# LATITUDES_DEG = [90, 90 - EARTH_AXIAL_TILT_DEGREES, 45, EARTH_AXIAL_TILT_DEGREES, 0]

	#
	# 3D plot of total sunlight in a day
	#

	# TODO: plot as 3D surface instead of image (or maybe both?)

	fig, ax = plt.subplots(1, 3, sharey=True)
	fig.suptitle(f'Total insolation over time (axial tilt {axial_tilt_deg}°)')

	latitude_deg = np.linspace(-90, 90, num=361, endpoint=True)
	latitude_rads = radians(latitude_deg)

	declination_linstep_deg = np.linspace(-axial_tilt_deg, axial_tilt_deg, 366//2, endpoint=True)
	declination_linstep_rads = radians(declination_linstep_deg)

	time_of_year = np.linspace(-0.5, 0.5, num=366//2, endpoint=True)
	declination_daystep_deg = np.sin(time_of_year * pi) * axial_tilt_deg
	declination_daystep_rads = radians(declination_daystep_deg)

	declination_by_day_grid, latitude_grid = np.meshgrid(declination_daystep_rads, latitude_rads)
	insolation_over_day_by_day = get_insolation_over_day(
		latitude_rads=latitude_grid, declination_rads=declination_by_day_grid)

	declination_lin_grid, latitude_grid = np.meshgrid(declination_linstep_rads, latitude_rads)
	insolation_over_day_by_declination  = get_insolation_over_day(
		latitude_rads=latitude_grid, declination_rads=declination_lin_grid)

	insolation_over_year = get_insolation_over_year(latitude_rads=latitude_rads, axial_tilt_deg=axial_tilt_deg)

	assert isinstance(insolation_over_year, np.ndarray) and insolation_over_year.ndim == 1
	insolation_equator = insolation_over_year[insolation_over_year.size // 2]
	insolation_pole = insolation_over_year[0]
	insolation_over_year_naive_cosine_fit = 0.5*(insolation_equator + insolation_pole) + 0.5*(insolation_equator - insolation_pole) * np.cos(2 * latitude_rads)

	insolation_over_year_double_cosine_fit = _double_cosine_fit(insolation_over_year, latitude_rads)

	latitude_pwl_rads = np.linspace(-0.5*pi, 0.5*pi, num=31, endpoint=True)
	latitude_pwl_deg = degrees(latitude_pwl_rads)
	insolation_over_year_pwl = get_insolation_over_year(latitude_rads=latitude_pwl_rads, axial_tilt_deg=axial_tilt_deg)

	# lut_latitude = np.linspace(min_latitude, max_latitude, num=num, endpoint=True)
	# lut_insolation = get_insolation_over_year(
	# 	latitude_rads=lut_latitude,
	# 	axial_tilt_deg=axial_tilt_deg,
	# )
	# insolation = np.interp(effective_latitude_with_turbulence_rads, lut_latitude, lut_insolation)


	im = ax[0].imshow(insolation_over_day_by_declination, aspect='auto', extent=[-axial_tilt_deg, axial_tilt_deg, -90, 90], cmap='inferno')
	ax[0].set_title('Total Daily (by declination)')
	ax[0].set_xlabel('Declination')
	ax[0].set_ylabel('Latitude')
	ax[0].grid()
	plt.colorbar(im, ax=ax[0])

	im = ax[1].imshow(insolation_over_day_by_day, aspect='auto', extent=[-0.5, 0.5, -90, 90], cmap='inferno')
	ax[1].set_title('Total Daily (by day)')
	ax[1].set_xlabel('Time of year')
	ax[1].set_ylabel('Latitude')
	ax[1].grid()
	plt.colorbar(im, ax=ax[1])

	ax[2].plot(insolation_over_year, latitude_deg, label='Actual')
	ax[2].plot(insolation_over_year_naive_cosine_fit, latitude_deg, '--', label='Naive Single Cosine fit')
	ax[2].plot(insolation_over_year_double_cosine_fit, latitude_deg, '--', label='Optimal Double Cosine fit')
	ax[2].plot(insolation_over_year_pwl, latitude_pwl_deg, '--', label='Piecewise linear')
	ax[2].set_title('Total Annual')
	ax[2].set_xlabel('Insolation')
	ax[2].set_xlim([0, 0.5])
	ax[2].set_yticks(np.arange(-90, 90 + ANGLE_STEP_MINOR, ANGLE_STEP_MINOR), minor=True)
	ax[2].set_yticks(np.arange(-90, 90 + ANGLE_STEP_MAJOR, ANGLE_STEP_MAJOR))
	ax[2].grid()
	ax[2].legend()


def _plot_total_sunlight_2d(axial_tilt_deg=EARTH_AXIAL_TILT_DEGREES):
	from matplotlib import pyplot as plt

	LATITUDES_DEG = [90, 90 - EARTH_AXIAL_TILT_DEGREES, 45, EARTH_AXIAL_TILT_DEGREES, 0]

	time_of_year = np.linspace(-0.5, 0.5, num=366//2, endpoint=True)
	declination_deg = np.sin(time_of_year * pi) * axial_tilt_deg
	declination_rads = radians(declination_deg)

	fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
	fig.suptitle(f'Total daily insolation (axial tilt {axial_tilt_deg}°)')

	for latitude_deg in LATITUDES_DEG:
		latitude_rads=radians(latitude_deg)

		insolation_over_day_this_latitude = get_insolation_over_day(
			latitude_rads=latitude_rads,
			declination_rads=declination_rads,
		)

		ax1.plot(declination_deg, insolation_over_day_this_latitude, label=f'{latitude_deg}° N')
		ax2.plot(time_of_year, insolation_over_day_this_latitude, label=f'{latitude_deg}° N')

	ax1.grid()
	ax1.legend()
	ax1.set_xlabel('Declination')
	
	ax2.grid()
	ax2.set_xlabel('Time of year')


def main(args=None):
	from argparse import ArgumentParser
	from matplotlib import pyplot as plt

	parser = ArgumentParser()
	parser.add_argument('--extra', action='store_true')
	args = parser.parse_args(args)

	_plot_sunlight_at_noon()
	_plot_sunlight_at_time()

	axial_tilts_deg = [EARTH_AXIAL_TILT_DEGREES]
	if args.extra:
		axial_tilts_deg += [ 5, 45, 75, 90 ]

	for axial_tilt_deg in axial_tilts_deg:
		_plot_total_sunlight_3d(axial_tilt_deg)
		_plot_total_sunlight_2d(axial_tilt_deg)

	plt.show()


if __name__ == "__main__":
	main()
