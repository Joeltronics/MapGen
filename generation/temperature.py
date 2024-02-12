#!/usr/bin/env python3

from typing import Final

import numpy as np

from utils.numeric import FloatOrArrayT, rescale, require_same_shape
from utils.consts import EARTH_AXIAL_TILT_DEGREES

DEFAULT_TEMPERATURE_RANGE_C: Final = (-10, 30)
DEGREES_C_COLDER_PER_KM_ELEVATION: Final = 7.5


def get_insolation_at_time(
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


def get_insolation_over_day(
		*,
		latitude_rads: FloatOrArrayT,
		declination_rads: FloatOrArrayT,
		precision: int = 512,
		) -> FloatOrArrayT:

	if isinstance(latitude_rads, np.ndarray) and isinstance(declination_rads, np.ndarray):
		if latitude_rads.shape != declination_rads.shape:
			raise ValueError(f'Arrays do not have same shape: {latitude_rads.shape} != {declination_rads.shape}')

	hour = np.linspace(0.0, 24.0, num=precision, endpoint=False)
	hour_angle_rads = np.radians(hour * (360 / 24) - 180)

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

	return insolation_over_day


def get_insolation_over_year(
		*,
		latitude_rads: FloatOrArrayT,
		axial_tilt_deg: float = EARTH_AXIAL_TILT_DEGREES,
		precision_hours: int = 512,
		precision_days: int = (366 // 2),
		) -> FloatOrArrayT:

	axial_tilt_rads = np.radians(axial_tilt_deg)

	declination_rads = np.linspace(-axial_tilt_rads, axial_tilt_rads, precision_days, endpoint=True)

	if isinstance(latitude_rads, np.ndarray):
		orig_shape = latitude_rads.shape
		declination_rads, latitude_rads = np.meshgrid(declination_rads, latitude_rads)
	else:
		orig_shape = None

	insolation_over_day = get_insolation_over_day(
		latitude_rads=latitude_rads,
		declination_rads=declination_rads,
		precision=precision_hours,
	)

	insolation_over_year = np.sum(insolation_over_day, axis=-1) / insolation_over_day.shape[-1]

	new_shape = insolation_over_year.shape if isinstance(insolation_over_year, np.ndarray) else None
	assert new_shape == orig_shape

	return insolation_over_year


def calculate_temperature(
		effective_latitude_deg: np.ndarray,
		topography_m: np.ndarray,
		temperature_noise: np.ndarray,
		ocean_turbulence_noise: np.ndarray,
		axial_tilt_deg: float = 0.0,
		declination_deg: float = 0.0,
		noise_strength = 0.5,
		ocean_turbulence_amount_deg = 5.,
		temperature_range_C = DEFAULT_TEMPERATURE_RANGE_C,
		) -> np.ndarray:

	require_same_shape(topography_m, effective_latitude_deg, temperature_noise, ocean_turbulence_noise)

	# TODO seasons: declination should affect ocean differently from land
	latitude = np.radians(effective_latitude_deg + declination_deg)

	latitude_turbulent = latitude + np.radians(ocean_turbulence_amount_deg)*ocean_turbulence_noise
	# TODO seasons: find a model that doesn't need to clip
	latitude_turbulent = np.clip(latitude_turbulent, -np.pi/2, np.pi/2)

	ocean_mask = topography_m < 0
	land_mask = np.logical_not(ocean_mask)

	# TODO: should this use domain warping instead of interpolation? or combination of both?
	latitude_temp_map = np.cos(2 * latitude) * 0.5 + 0.5

	temperature_01 = temperature_noise * noise_strength + latitude_temp_map * (1.0 - noise_strength)

	# More domain warping over ocean
	temperature_01[ocean_mask] = np.cos(2 * latitude_turbulent[ocean_mask]) * 0.5 + 0.5

	# TODO: this is probably not the best way of going about elevation...
	# elevation_temp_map = 1.0 - np.clip(elevation, 0.0, 1.0)
	# temperature_01 *= elevation_temp_map

	temperature_C = rescale(temperature_01, (0.0, 1.0), temperature_range_C)
	temperature_C -= (DEGREES_C_COLDER_PER_KM_ELEVATION / 1000) * np.maximum(topography_m, 0.0)

	# temperature_C[ocean_mask] = np.maximum(temperature_C[ocean_mask], SEAWATER_FREEZING_POINT_C - 0.1)

	return temperature_C


def main(args=None):
	from argparse import ArgumentParser
	from matplotlib import pyplot as plt
	from numpy import sin, cos, radians, degrees

	parser = ArgumentParser()
	parser.add_argument('--extra', action='store_true')
	args = parser.parse_args(args)

	HALF_PI = np.pi / 2.0

	ANGLE_STEP_MAJOR = 15
	ANGLE_STEP_MINOR = 5

	# def get_insolation_at_time(
	# 		latitude: np.ndarray | float,
	# 		declination: np.ndarray | float,
	# 		hour_angle: np.ndarray | float,
	# 		signed = False,
	# 		):
	# 	ret = sin(latitude) * sin(declination) + cos(latitude) * cos(declination) * cos(hour_angle)
	# 	if not signed:
	# 		if isinstance(ret, np.ndarray):
	# 			np.maximum(0, ret, out=ret)
	# 		else:
	# 			ret = max(0, ret)
	# 	return ret

	#
	# Sunlight at noon
	#

	latitude_deg = np.linspace(-90, 90, num=361, endpoint=True)

	fig, (ax1, ax2)  = plt.subplots(1, 2)
	fig.suptitle('Sunlight at noon')

	for declination_deg, label in [
		(EARTH_AXIAL_TILT_DEGREES, 'Dec 21'),
		(0, 'Mar/Sep 21'),
		(-EARTH_AXIAL_TILT_DEGREES, 'Jun 21'),
	]:
		effective_latitude = radians(latitude_deg + declination_deg)  # 90 - declination
		elevation = HALF_PI - np.abs(effective_latitude)
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

	latitudes_deg = [90, 90 - EARTH_AXIAL_TILT_DEGREES, 45, EARTH_AXIAL_TILT_DEGREES, 0]

	#
	# Sunlight by time
	#

	fig, axes = plt.subplots(len(latitudes_deg), 2, sharex=True)
	fig.suptitle('Sunlight by time')

	hour = np.linspace(0.0, 24.0, num=512, endpoint=False)
	hour_angle = radians(hour * (360 / 24) - 180)

	for idx, latitude_deg in enumerate(latitudes_deg):
 
		ax1 = axes[idx, 0]
		ax2 = axes[idx, 1]

		latitude = np.radians(latitude_deg)

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

	axial_tilts_deg = [EARTH_AXIAL_TILT_DEGREES]
	if args.extra:
		axial_tilts_deg += [
			5,
			45,
			82.23,  # Uranus
			90,
		]

	for axial_tilt_deg in axial_tilts_deg:

		#
		# 3D plot of total sunlight in a day
		#

		# TODO: plot as 3D surface instead of image (or maybe both)

		fig, ax = plt.subplots(1, 2, sharey=True)
		fig.suptitle(f'Total insolation over time (axial tilt {axial_tilt_deg}°)')

		latitude_deg = np.linspace(-90, 90, num=361, endpoint=True)
		declination_deg = np.linspace(-axial_tilt_deg, axial_tilt_deg, 183, endpoint=True)
		declination = radians(declination_deg)
		hour = np.linspace(0.0, 24.0, num=512, endpoint=False)
		hour_angle = radians(hour * (360 / 24) - 180)

		if True:
			declination_grid, latitude_grid = np.meshgrid(declination, radians(latitude_deg))
			insolation_over_day = get_insolation_over_day(latitude_rads=latitude_grid, declination_rads=declination_grid)
		else:
			declination_grid, latitude_grid, hour_grid = np.meshgrid(declination, radians(latitude_deg), hour_angle)
			insolation = get_insolation_at_time(
				latitude_rads=latitude_grid, declination_rads=declination_grid, hour_angle_rads=hour_grid)
			insolation_over_day = np.sum(insolation, axis=2) / insolation.shape[2]

		if True:
			insolation_over_year = get_insolation_over_year(latitude_rads=radians(latitude_deg), axial_tilt_deg=axial_tilt_deg)
		else:
			insolation_over_year = np.sum(insolation_over_day, axis=1) / insolation_over_day.shape[1]

		im = ax[0].imshow(insolation_over_day, aspect='auto', extent=[-axial_tilt_deg, axial_tilt_deg, -90, 90], cmap='inferno')
		ax[0].set_title('Total Daily')
		ax[0].set_xlabel('Declination')
		ax[0].set_ylabel('Latitude')
		ax[0].grid()
		plt.colorbar(im, ax=ax[0])

		ax[1].plot(insolation_over_year, latitude_deg)
		ax[1].set_title('Total Annual')
		ax[1].set_xlabel('Insolation')
		ax[1].set_xlim([0, 0.5])
		ax[1].set_yticks(np.arange(-90, 90 + ANGLE_STEP_MINOR, ANGLE_STEP_MINOR), minor=True)
		ax[1].set_yticks(np.arange(-90, 90 + ANGLE_STEP_MAJOR, ANGLE_STEP_MAJOR))
		ax[1].grid()

		#
		# 2D plot of total daily insolation
		#

		fig, ax = plt.subplots(1, 1, sharex=True)
		fig.suptitle(f'Total daily insolation (axial tilt {axial_tilt_deg}°)')

		for latitude_deg in latitudes_deg:
			if True:
				insolation_over_day_this_latitude = get_insolation_over_day(
					latitude_rads=radians(latitude_deg),
					declination_rads=declination,
				)
			else:
				y_idx = round((latitude_deg + 90) / 180 * (insolation_over_day.shape[0] - 1))
				insolation_over_day_this_latitude = insolation_over_day[y_idx, :]
			ax.plot(declination_deg, insolation_over_day_this_latitude, label=f'{latitude_deg}° N')
			ax.grid()

		ax.legend()
		ax.set_xlabel('Declination')

	plt.show()


if __name__ == "__main__":
	main()
