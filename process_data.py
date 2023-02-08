#!/usr/bin/env python

# data from:
# https://visibleearth.nasa.gov/view_cat.php?categoryID=1484
# https://neo.sci.gsfc.nasa.gov/dataset_index.php

import argparse
from collections.abc import Sequence
import os
from typing import Optional

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import tqdm

from data.data import _import_csv, _import_img, get_elevation, get_bathymetry, get_topography, get_land_ocean_ice, \
	get_average_wv, get_average_veg, get_average_rain, get_mask, get_average_surface_temp, sat_vapor_density_from_temp

from utils.image import float_to_uint8, array_to_image, image_to_array, resize_array, average_color, sphere_gradient
from utils.numeric import lerp, reverse_lerp, rescale, data_range
from utils.utils import tprint

PI = np.pi
TWOPI = 2.0*np.pi

DEFAULT_HIST_BINS = 256
DEFAULT_QUANTIZATION = 16
DEFAULT_SMOOTHING = 50

#SCATTER_ALPHA = 0.002
SCATTER_ALPHA = 0.005


# COLORMAP_TEMP_RANGE = (-25., 40.)
COLORMAP_TEMP_RANGE = (-5., 30.)
# COLORMAP_MAX_ELEVATION = 3800
COLORMAP_MAX_ELEVATION = 2000


OUT_DIR = 'out'


def img_y_to_latitude(y):
	if np.amin(y) < 0 or np.amax(y) >= 1800:
		raise ValueError('y must be between 0 and 1799')
	if not isinstance(y, np.ndarray):
		y = np.array(y)
	y = y.astype(float)
	return rescale(y, (0, 1800-1), (90,-90), clip=False)


def get_gradient(magnitude=False, including_ocean=False, scale_dx_with_latitude=True):
	im = get_topography() if including_ocean else get_elevation()
	return sphere_gradient(im, magnitude=magnitude, latitude_adjust=scale_dx_with_latitude)


def hist_2d(x_data: Sequence, y_data: Sequence, bins=256, alpha=SCATTER_ALPHA, log=True, ignore_nan=True):

	# TODO: option to pass in axes

	if not isinstance(x_data, np.ndarray):
		x_data = np.array(x_data)

	if not isinstance(y_data, np.ndarray):
		y_data = np.array(y_data)

	# TODO: matplotlib/pyplot might already ignore NaN by default
	if ignore_nan:
		mask = np.logical_not(np.logical_or(np.isnan(x_data), np.isnan(y_data)))
		x_data = x_data[mask]
		y_data = y_data[mask]

	x_data = x_data[:]
	y_data = y_data[:]

	if False:
		plt.scatter(x_data, y_data, 1, alpha=alpha)
	else:
		H, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins)
		if log:
			H = np.log10(H + 1)
		plt.imshow(H.T, interpolation='nearest', origin='lower', aspect='auto', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
		# TODO: plt.colorbar()


def hist_3d_median(x: np.ndarray, y: np.ndarray, z: np.ndarray, ax, *, bins=32):

	x_range_in = data_range(x)
	y_range_in = data_range(y)
	x_quant = np.round(rescale(x, range_in=x_range_in, range_out=(0, bins-1))).astype(int)
	y_quant = np.round(rescale(y, range_in=y_range_in, range_out=(0, bins-1))).astype(int)

	x_grid = np.zeros((bins, bins), dtype=x.dtype)
	y_grid = np.zeros((bins, bins), dtype=y.dtype)
	z_grid_25 = np.zeros((bins, bins), dtype=z.dtype)
	z_grid_50 = np.zeros((bins, bins), dtype=z.dtype)
	z_grid_75 = np.zeros((bins, bins), dtype=z.dtype)
	num_points = np.zeros((bins, bins), dtype=int)

	# TODO: get this working
	# x_grid, y_grid = np.meshgrid(np.arange(bins), np.arange(bins))

	# TODO: option for min_num_points, to remove data that's not statistically significant

	for y_val in range(bins):
		y_mask = (y_quant == y_val)
		for x_val in range(bins):
			# points_mask = np.logical_and(y_quant == y_val, x_quant == x_val)
			points_mask = np.logical_and(y_mask, x_quant == x_val)

			z_vals = z[points_mask]

			this_num_points = len(z_vals)
			num_points[y_val, x_val] = this_num_points

			if this_num_points:
				z25 = np.percentile(z_vals, 25)
				z50 = np.median(z_vals)
				z75 = np.percentile(z_vals, 75)
			else:
				z25 = z50 = z75 = np.nan

			# TODO: use np.meshgrid for X & Y instead (didn't work for some reason)
			x_grid[y_val, x_val] = x_val
			y_grid[y_val, x_val] = y_val

			z_grid_25[y_val, x_val] = z25
			z_grid_50[y_val, x_val] = z50
			z_grid_75[y_val, x_val] = z75

	x_grid = rescale(x_grid.astype(float), range_in=(0, bins-1), range_out=x_range_in)
	y_grid = rescale(y_grid.astype(float), range_in=(0, bins-1), range_out=y_range_in)

	ax.plot_surface(x_grid, y_grid, z_grid_50, cmap='viridis')

	# TODO: do something with percentiles
	# ax.plot_surface(x_grid, y_grid, z_grid_25, cmap='viridis')
	# ax.plot_surface(x_grid, y_grid, z_grid_75, cmap='viridis')

	# TODO: do something with num_points too


def latitude_plot(
		value: np.ndarray,
		/,
		value_label: str,
		title='',
		*,
		mask: Optional[np.ndarray]=None,
		abs_lat=False,
		lat_range=None,
		alpha=SCATTER_ALPHA,
		smoothing=DEFAULT_SMOOTHING,
		):

	if title:
		tprint('Making latitude plot for %s' % title)
	else:
		tprint('Making latitude plot for %s' % value_label)

	# TODO: remove NaN values (or is this unnecessary?)

	num_points = 900 if abs_lat else 1800

	medians = np.zeros(num_points)
	means = np.zeros(num_points)
	stdev_m = np.zeros(num_points)
	stdev_p = np.zeros(num_points)
	mins = np.zeros(num_points)
	maxes = np.zeros(num_points)

	valid_data = np.zeros(num_points, dtype=bool)

	hist_x = []
	hist_y = []

	for y in range(num_points):

		row_data = value[y, :]
		if mask is not None:
			row_data = row_data[mask[y, :]]

		if abs_lat:
			y2 = 1800 - y - 1
			row_data_2 = value[y2, :]
			if mask is not None:
				row_data_2 = row_data_2[mask[y2, :]]
			row_data = np.concatenate((row_data, row_data_2))

		if len(row_data) == 0:
			valid_data[y] = False
			continue

		valid_data[y] = True
		mean = np.mean(row_data)
		std = np.std(row_data)
		means[y] = mean
		medians[y] = np.median(row_data)
		stdev_m[y] = mean - std
		stdev_p[y] = mean + std
		mins[y] = np.amin(row_data)
		maxes[y] = np.amax(row_data)

		hist_x.extend([val for val in row_data])
		hist_y.extend([y] * len(row_data))

	if not valid_data.any():
		print('WARNING: no data for latitude plot')
		return

	# TODO: do the smoothing before the valid_data resolution

	means = means[valid_data]
	medians = medians[valid_data]
	stdev_m = stdev_m[valid_data]
	stdev_p = stdev_p[valid_data]
	mins = mins[valid_data]
	maxes = maxes[valid_data]

	data_y = np.array([y for y in range(num_points) if valid_data[y]])
	data_y = img_y_to_latitude(data_y)
	hist_y = img_y_to_latitude(hist_y)

	assert len(data_y) == len(means)

	if smoothing:
		# TODO: try triangular kernel?
		w = np.ones(smoothing, dtype=float)
		w /= w.sum()

		def smooth(data):
			return np.convolve(w, data, mode='valid')

		means = smooth(means)
		medians = smooth(medians)
		stdev_m = smooth(stdev_m)
		stdev_p = smooth(stdev_p)
		mins = smooth(mins)
		maxes = smooth(maxes)

		data_y = data_y[smoothing//2:-smoothing//2+1]

	hist_2d(hist_x, hist_y, alpha=alpha)
	plt.plot(mins, data_y, 'b-')
	plt.plot(maxes, data_y, 'b-')
	plt.plot(stdev_m, data_y, 'c-')
	plt.plot(stdev_p, data_y, 'c-')
	plt.plot(medians, data_y, 'm-')
	plt.plot(means, data_y, 'r-')

	if title is not None:
		plt.title(title)
	else:
		plt.title('%s by Latitude' % value_label)

	plt.xlabel(value_label)
	plt.ylabel('abs(Latitude)' if abs_lat else 'Latitude')
	plt.ylim(lat_range if lat_range else ([0, 90] if abs_lat else [-90, 90]))
	plt.grid()


def water_corr(
		land_mask: np.ndarray,
		ocean_mask: np.ndarray,
		temp: np.ndarray,
		wv: np.ndarray,
		veg: np.ndarray,
		rain: np.ndarray,
		smoothing=DEFAULT_SMOOTHING,
		):

	log_rain = np.log10(rain)

	# TODO: convert absolute wv to humidity and use that

	# y = np.repeat(np.vstack(np.arange(0, 1800, dtype=float)), 3600, axis=1)
	# latitude = img_y_to_latitude(y)

	temp_mask = np.logical_not(np.isnan(temp))
	rain_mask = np.logical_not(np.isnan(rain))
	wv_mask = np.logical_not(np.isnan(wv))
	veg_mask = np.logical_not(np.isnan(veg))

	# land_temp_mask = np.logical_and(land_mask, temp_mask)
	land_rain_mask = np.logical_and(land_mask, rain_mask)
	ocean_rain_mask = np.logical_and(ocean_mask, rain_mask)

	# wv by temp

	fig = plt.figure()
	fig.suptitle('Water Vapor by Temperature')

	mask = np.logical_and(temp_mask, wv_mask)
	#temp_masked = temp[temp_mask][:]
	#svd = sat_vapor_density_from_temp(temp_masked) * 6.0/(100000.)

	hist_2d(temp[mask][:], wv[mask][:]*6.0)
	#plt.plot(temp_masked, svd, '.')
	plt.xlabel('Temperature (C)')
	plt.ylabel('Water Vapor (cm)')
	plt.grid()

	# humidity by latitude
	pass
	
	# veg by latitude
	
	fig = plt.figure()
	fig.suptitle('Vegetation by Latitude')
	latitude_plot(veg, "Vegetation (NDVI)", mask=land_mask, abs_lat=False, smoothing=smoothing)

	# rainfall by latitude
	
	fig = plt.figure()
	fig.suptitle('Rainfall by Latitude')
	plt.subplot(1, 3, 1)
	latitude_plot(log_rain, "Log Rainfall", mask=rain_mask, lat_range=[-50, 50], smoothing=smoothing)
	plt.title('Total')
	plt.subplot(1, 3, 2)
	latitude_plot(log_rain, "Log Rainfall (land)", mask=land_rain_mask, lat_range=[-50, 50], smoothing=smoothing)
	plt.title('Land')
	plt.subplot(1, 3, 3)
	latitude_plot(log_rain, "Log Rainfall (ocean)", mask=ocean_rain_mask, lat_range=[-50, 50], smoothing=smoothing)
	plt.title('Ocean')

	# humidity/veg
	pass

	# veg/rain

	fig = plt.figure()
	fig.suptitle('Vegetation by Rainfall')
	hist_2d(log_rain[land_rain_mask][:], veg[land_rain_mask][:])
	plt.xlabel('Log10 Rainfall (cm)')
	plt.ylabel('Vegetation (NDVI)')
	plt.grid()


	# veg/rain/temp
	fig = plt.figure()
	fig.suptitle('Vegetation by Rainfall & Temperature')
	ax = fig.add_subplot(111, projection='3d')
	hist_3d_median(x=log_rain[land_rain_mask][:], y=temp[land_rain_mask][:], z=veg[land_rain_mask], ax=ax)
	ax.set_xlabel('Log10 Rainfall (cm)')
	ax.set_ylabel('Temperature (C)')
	ax.set_zlabel('Vegetation (NDVI)')

	fig = plt.figure()
	fig.suptitle('Rainfall by Vegetation & Temperature')
	ax = fig.add_subplot(111, projection='3d')
	hist_3d_median(x=veg[land_rain_mask], y=temp[land_rain_mask][:], z=log_rain[land_rain_mask][:], ax=ax)
	ax.set_xlabel('Vegetation (NDVI)')
	ax.set_ylabel('Temperature (C)')
	ax.set_zlabel('Log10 Rainfall (cm)')

	# humidity/rain
	pass


def plot_temp_corr(temp, topo, do_3d_plot=False, smoothing=DEFAULT_SMOOTHING):

	temp_mask = np.logical_not(np.isnan(temp))

	plt.figure()

	if do_3d_plot:
		img_y = np.repeat(np.vstack(np.arange(0, 1800, dtype=int)), 3600, axis=1)
		latitude = img_y_to_latitude(img_y.astype(float))

		mask = np.logical_and(temp_mask, topo <= 500.)
		latitude_plot(temp, "Temperature (C) at elevation < 500m", mask=mask, abs_lat=True, smoothing=smoothing)

		fig = plt.figure()
		fig.suptitle('Temperature by elevation')
		for idx, this_latitude in enumerate([60, 30, 0, -30]):
			plt.subplot(2, 2, 1+idx)
			mask = np.logical_and(temp_mask, np.logical_and(latitude > this_latitude - 1, latitude < this_latitude + 1))
			hist_2d(topo[mask], temp[mask])
			plt.title(f'Near {abs(this_latitude)}°{" N" if this_latitude > 0 else " S" if this_latitude < 0 else ""}')
			plt.xlabel('Elevation (m)')
			plt.ylabel('Temperature (C)')
			plt.grid()

		# 3d plot of latitude-elevation-temp

		fig = plt.figure()
		fig.suptitle('Temperature by Latitude & Elevation')
		ax = fig.add_subplot(111, projection='3d')
		hist_3d_median(x=latitude[temp_mask], y=topo[temp_mask], z=temp[temp_mask], ax=ax)
		ax.set_xlabel('Latitude')
		ax.set_ylabel('Elevation (m)')
		ax.set_zlabel('Temperature (C)')

		fig = plt.figure()
		fig.suptitle('Temperature by Latitude & Elevation')
		ax = fig.add_subplot(111, projection='3d')
		hist_3d_median(x=np.abs(latitude[temp_mask]), y=topo[temp_mask], z=temp[temp_mask], ax=ax)
		ax.set_xlabel('Abs Latitude')
		ax.set_ylabel('Elevation (m)')
		ax.set_zlabel('Temperature (C)')

	else:
		topo_ranges = (500, 1000, 1500, 2000, 2500, 3000, 4000, 5000)

		for n in range(len(topo_ranges)+1):

			idx_min = n-1 if n > 0 else None
			idx_max = n if n < len(topo_ranges) else None

			range_min = topo_ranges[idx_min] if idx_min is not None else None
			range_max = topo_ranges[idx_max] if idx_max is not None else None

			if (range_min is not None) and (range_max is not None):
				mask = np.logical_and(temp_mask, topo <= range_max)
				mask = np.logical_and(mask, topo > range_min)
				range_text = '%i - %i m' % (range_min, range_max)

			elif range_max is not None:
				mask = np.logical_and(temp_mask, topo <= range_max)
				range_text = '< %i m' % (range_max)

			elif range_min is not None:
				mask = np.logical_and(temp_mask, topo > range_min)
				range_text = '> %i m' % (range_min)

			else:
				raise AssertionError("range_min and range_max both unset! (n %i)" % n)
				#assert False and ""range_min and range_max both unset!"
			
			plt.subplot(331 + n)
			latitude_plot(
				temp,
				#value_label="Temperature (C)",
				value_label='',
				title=("Temperature (C) at elevation %s" % range_text),
				mask=mask,
				abs_lat=True,
				alpha=0.01)


def make_colormap(mask, temp, moisture, land_ocean_ice=None, quant_steps=DEFAULT_QUANTIZATION):

	# TODO: try this in higher resolution land_ocean_ice (8192x4096)
	# TODO: also factor in elevation & slope

	mask = np.logical_and(mask, np.logical_not(np.isnan(temp)))
	mask = np.logical_and(mask, np.logical_not(np.isnan(moisture)))

	temp = np.copy(temp)
	moisture = np.copy(moisture)

	if land_ocean_ice is None:
		land_ocean_ice = get_land_ocean_ice(resize=True, as_img=False)

	tprint('temp range: (%f,%f)' % data_range(temp[mask]))
	tprint('moisture range: (%f,%f)' % data_range(moisture[mask]))

	temp = np.clip(temp, 0., 1.)
	moisture = np.clip(moisture, 0., 1.)

	temp_quant = np.floor(temp*quant_steps).astype(int)
	moisture_quant = np.floor(moisture*quant_steps).astype(int)
	tprint('temp_quant range: %i, %i' % data_range(temp_quant[mask]))
	tprint('moisture_quant range: %i, %i' % data_range(moisture_quant[mask]))

	result = np.zeros((quant_steps, quant_steps, 3))
	num_points = np.zeros((quant_steps, quant_steps), dtype=int)

	with tqdm.tqdm(total=quant_steps*quant_steps, desc='Creating colormap') as pbar:
		for t in range(quant_steps):

			temp_mask = temp_quant == t

			for m in range(quant_steps):
				points_mask = np.logical_and(temp_mask, moisture_quant == m)
				points_mask = np.logical_and(points_mask, mask)

				points = land_ocean_ice[points_mask]

				this_num_points = len(points)
				num_points[t, m] = this_num_points

				if this_num_points != 0:
					# TODO: modify average_color to take points in directly, instead of unstacked
					# result[t, m, :] = average_color(points_r, points_g, points_b)
					result[t, m, :] = average_color(points)

				pbar.update()

	tprint('Done quantizing')

	return result, num_points


def tidy_colormap(cmap: np.ndarray, colormap_num_points: np.ndarray, bias_x=True) -> np.ndarray:

	height, width, _ = cmap.shape
	assert colormap_num_points.shape == (height, width)

	total_num_points = np.sum(colormap_num_points)
	average_num_points_per_pixel = total_num_points / (cmap.shape[0] * cmap.shape[1])

	# Pixels with count below this threshold will be replaced with average of neighbors
	num_points_to_nan_thresh = average_num_points_per_pixel / 1000

	# Pixels with count below this threshold (but above nan threshold) will still be used, but weighted lower in averaging
	num_points_max_weight_thresh = average_num_points_per_pixel / 100

	cmap = np.copy(cmap)
	input_nan_mask = (colormap_num_points < num_points_to_nan_thresh)
	cmap[input_nan_mask, 0] = np.nan
	cmap[input_nan_mask, 1] = np.nan
	cmap[input_nan_mask, 2] = np.nan

	# TODO: this is a quick & dirty algorithm with a lot of problems - could do much better than this with proper interpolation (and in a single pass)

	# TODO: degamma this and average in linear domain!

	num_nan_pixels = np.sum(np.isnan(cmap))

	with tqdm.tqdm(total=num_nan_pixels//3, desc='Filling in gaps') as pbar:
		while True:
			if not num_nan_pixels:
				break

			cmap_out = np.copy(cmap)

			for y in range(height):
				for x in range(width):
					if not np.isnan(cmap[y, x, 0]):
						continue

					WEIGHT_ADJACENT_X =     1.0
					WEIGHT_ADJACENT_Y =     0.5    if bias_x else 1.0
					WEIGHT_CORNER =         0.25   if bias_x else 0.5
					WEIGHT_2_ADJACENT_X =   0.5    if bias_x else 0.25
					WEIGHT_2_ADJACENT_Y =   0.125  if bias_x else 0.25
					WEIGHT_KNIGHTS_MOVE_X = 0.125
					WEIGHT_KNIGHTS_MOVE_Y = 0.0625 if bias_x else 0.125

					neighbor_coords = [
						(-1, 0, WEIGHT_ADJACENT_Y), (1, 0, WEIGHT_ADJACENT_Y),
						(0, -1, WEIGHT_ADJACENT_X), (0, 1, WEIGHT_ADJACENT_X),
						(-1, -1, WEIGHT_CORNER), (-1, 1, WEIGHT_CORNER), (+1, -1, WEIGHT_CORNER), (+1, +1, WEIGHT_CORNER),

						(-2, 0, WEIGHT_2_ADJACENT_Y), (2, 0, WEIGHT_2_ADJACENT_Y),
						(0, -2, WEIGHT_2_ADJACENT_X), (0, 2, WEIGHT_2_ADJACENT_X),

						(1, 2, WEIGHT_KNIGHTS_MOVE_X), (2, 1, WEIGHT_KNIGHTS_MOVE_Y),
						(-1, -2, WEIGHT_KNIGHTS_MOVE_X), (-2, -1, WEIGHT_KNIGHTS_MOVE_Y),
						(1, -2, WEIGHT_KNIGHTS_MOVE_X), (-2, 1, WEIGHT_KNIGHTS_MOVE_Y),
						(-1, 2, WEIGHT_KNIGHTS_MOVE_X), (2, -1, WEIGHT_KNIGHTS_MOVE_Y),
					]

					def f_weight(coord_weight: float, num_points: int):
						return coord_weight * rescale(
							float(num_points),
							range_in=(0., float(num_points_max_weight_thresh)),
							range_out=(0., 1.),
							clip=True)

					neighbor_pixels = [
						(cmap[y+ny, x+nx, :], (f_weight(coord_weight, colormap_num_points[y+ny, x+nx]), coord_weight))
						for (ny, nx, coord_weight) in neighbor_coords
						if (0 <= (y + ny) < height) and (0 <= (x + nx) < width) and not np.isnan(cmap[y+ny, x+nx, 0])
					]

					if not neighbor_pixels:
						continue

					sum_pixels = sum(pixel * weight[0] for pixel, weight in neighbor_pixels)
					sum_weights = sum(weight[0] for _, weight in neighbor_pixels)

					if not sum_weights:
						# As fallback, use only coord_weight
						# This shouldn't be possible on 1st pass, but it can happen on later passes
						sum_pixels = sum(pixel * weight[1] for pixel, weight in neighbor_pixels)
						sum_weights = sum(weight[1] for _, weight in neighbor_pixels)

					avg_neighbor = sum_pixels / sum_weights
					assert not np.isnan(avg_neighbor).any()
					cmap_out[y, x, :] = avg_neighbor
					pbar.update()

			cmap = cmap_out

			new_num_nan_pixels = np.sum(np.isnan(cmap))
			if new_num_nan_pixels == num_nan_pixels:
				raise ValueError('Failed to reduce number of NaN pixels!')
			num_nan_pixels = new_num_nan_pixels

	assert not np.isnan(cmap).any()

	# TODO: also apply a small Gaussian blur to whole colormap

	return cmap


def plot_topography(elevation, bathymetry, topography, land_ocean_mask, args):
	tprint('topo range: (%f,%f)' % data_range(elevation))
	bathy_for_hist = bathymetry[np.logical_not(land_ocean_mask)]
	topo_for_hist = elevation[land_ocean_mask]

	fig = plt.figure()
	fig.suptitle('Topography histograms')

	plt.subplot(2, 2, 1)
	plt.hist(bathy_for_hist, bins=args.hist_bins)
	plt.title('Bathymetry')

	plt.subplot(2, 2, 2)
	plt.hist(topo_for_hist, bins=args.hist_bins)
	plt.title('Elevation')

	tprint('topography + bathymetry range: (%f,%f)' % data_range(topography))

	tb_for_hist = topography.flatten()

	plt.subplot(2, 1, 2)
	plt.hist(tb_for_hist, bins=args.hist_bins)
	plt.title('Elevation + Bathymetry histogram')

	# TODO: gradient histogram


def get_parser(add_help=True):
	parser = argparse.ArgumentParser(add_help=add_help)

	parser.add_argument('-q', dest='quantization', metavar='QUANTIZATION', type=int, default=DEFAULT_QUANTIZATION, help=f'Quantization lookup table steps, default {DEFAULT_QUANTIZATION}')
	parser.add_argument('--no-save', dest='save', action='store_false', help="Don't save anything")
	parser.add_argument('--no-cmap', dest='calculate_colormap', action='store_false', help="Don't calculate colormap")

	g = parser.add_argument_group('Plotting')
	g.add_argument('--plot', dest='show_plots', action='store_true', help='Show plots')
	g.add_argument('--hist-bins', default=DEFAULT_HIST_BINS, help=f'Histogram number of bins, default {DEFAULT_HIST_BINS}')
	g.add_argument('--corr-smoothing', dest='smoothing', metavar='SMOOTHING', type=int, default=DEFAULT_SMOOTHING, help=f'Correlation plot smoothing, default {DEFAULT_SMOOTHING}')

	return parser


def parse_args():
	parser = get_parser()
	return parser.parse_args()


def estimate_rainfall(land_mask, ocean_mask, temp, wv, veg, rain):
	# TODO
	return rain


def run(args):
	start_time = tprint('Starting', is_start=True)

	if False and args.show_plots:
		# DEBUG

		land_mask = get_mask(land=True)
		ocean_mask = get_mask(ocean=True)

		tprint('Getting topography')
		elevation = get_elevation()

		tprint('Getting average surface temperature')
		temp = get_average_surface_temp()
		tprint('Data range: (%f,%f)' % data_range(temp))
		tprint('Getting average water vapor')
		wv = get_average_wv()
		tprint('Data range: (%f,%f)' % data_range(wv))

		tprint('Getting average vegetation')
		veg = get_average_veg()
		tprint('Data range: (%f,%f)' % data_range(veg))

		tprint('Getting average rainfall')
		rain = get_average_rain()
		tprint('Data range: (%f,%f)' % data_range(rain))

		# tprint('Estimating rainfall where data is missing')
		# estimated_rainfall = estimate_rainfall(land_mask, ocean_mask, temp, wv, veg, rain)
		# tprint('Data range: (%f,%f)' % data_range(rain))
		# fig, ax = plt.subplots(1, 1)
		# fig.suptitle('Estimated rainfall')
		# ax.imshow(estimated_rainfall, cmap='YlGn')

		tprint('Plotting 3D temperature correlations')
		plot_temp_corr(temp, elevation, do_3d_plot=True)

		tprint('Plotting moisture correlations')
		water_corr(land_mask, ocean_mask, temp, wv, veg, rain, smoothing=args.smoothing)
		plt.show()
		return



	if not os.path.exists(OUT_DIR):
		os.mkdir(OUT_DIR)

	#test_csv_jpg()

	land_ocean_mask = get_mask(land=True, lakes=True)
	land_mask = get_mask(land=True)
	ocean_mask = get_mask(ocean=True)
	water_mask = np.logical_not(land_mask)
	#water_mask = get_land_mask(invert=True, ocean_only=False)

	tprint('Getting topography')
	elevation = get_elevation()
	bathymetry = get_bathymetry()
	topography = get_topography()

	tprint('Getting gradient')
	gradient_mag = get_gradient(magnitude=True, including_ocean=False)
	tprint('Gradient range: (%f,%f)' % data_range(gradient_mag))

	if args.show_plots:
		plot_topography(elevation, bathymetry, topography, land_ocean_mask, args)

	tprint('Getting average surface temperature')
	temp = get_average_surface_temp(ocean=False)
	temp_with_ocean = get_average_surface_temp(ocean=True)
	tprint('Data range: (%f,%f)' % data_range(temp))
	
	temp_for_im = rescale(temp, (-25.,45.), (0.,1.), clip=True)
	# temp_for_im[water_mask] = 1.
	temp_for_im[water_mask] = np.nan
	temp_im = array_to_image(temp_for_im, nan=(1, 0, 1))

	if args.save:
		temp_im.save(os.path.join(OUT_DIR, 'temp.png'), 'PNG')

	if args.show_plots:
		# TODO: plot temperature by elevation at a single latitude, maybe for a few latitudes
		tprint('Plotting temperature correlations (2D)')
		plot_temp_corr(temp, elevation, smoothing=args.smoothing, do_3d_plot=False)
		tprint('Plotting temperature correlations (3D)')
		plot_temp_corr(temp, elevation, smoothing=args.smoothing, do_3d_plot=True)

	tprint('Getting average water vapor')
	wv = get_average_wv()
	tprint('Data range: (%f,%f)' % data_range(wv))

	tprint('Getting average vegetation')
	veg = get_average_veg()
	veg_valid = np.nan_to_num(veg, copy=True, nan=0)
	tprint('Data range: (%f,%f)' % data_range(veg))

	tprint('Getting average rainfall')
	rain = get_average_rain()
	tprint('Data range: (%f,%f)' % data_range(rain))

	# tprint('Estimating rainfall where data is missing')
	# estimated_rainfall = estimate_rainfall(land_mask, ocean_mask, temp, wv, veg, rain)
	# tprint('Data range: (%f,%f)' % data_range(rain))
	# fig, ax = plt.subplots(1, 1)
	# fig.suptitle('Estimated rainfall')
	# ax.imshow(estimated_rainfall, cmap='YlGn')

	if args.show_plots:
		tprint('Plotting moisture correlations')
		water_corr(land_mask, ocean_mask, temp, wv, veg, rain, smoothing=args.smoothing)

	# TODO: estimate rainfall where it's missing

	land_ocean_ice = get_land_ocean_ice(resize=True, as_img=False)

	if args.calculate_colormap:

		temp_range = COLORMAP_TEMP_RANGE

		colormap_mask = np.logical_and(land_mask, elevation < COLORMAP_MAX_ELEVATION)

		# TODO: some sort of averaged mix of these?
		# Precipitation is ideal here (at least for our use case), but it's missing data near poles
		moisture_proxy, moisture_proxy_name, moisture_proxy_range = veg, 'Vegetation', (0.1, 0.9)
		# moisture_proxy, moisture_proxy_name, moisture_proxy_range = wv, 'Water vapor', (0., 6.)
		# moisture_proxy, moisture_proxy_name, moisture_proxy_range = rain, 'Precipitation', (0., 300.)
		# moisture_proxy, moisture_proxy_name, moisture_proxy_range = np.log10(rain), 'log Precipitation', (1., np.log10(300.))

		# temp_scaled = rescale(temp, temp_range, (0., 1.), clip=True)
		temp_scaled = rescale(temp_with_ocean, temp_range, (0., 1.), clip=True)
		moisture_proxy = rescale(moisture_proxy, range_in=moisture_proxy_range, range_out=(0., 1.), clip=True)

		tprint('Calculating colormap', start=start_time)
		colormap, colormap_num_points = make_colormap(mask=colormap_mask, temp=temp_scaled, moisture=moisture_proxy, land_ocean_ice=land_ocean_ice, quant_steps=args.quantization)
		tprint('Done calculating colormap', start=start_time)

		# TODO: remove pixels with insignificant colormap_num_points

		tprint('Tidying up colormap...')
		colormap_tidied = tidy_colormap(colormap, colormap_num_points)

		colormap_img = array_to_image(colormap)
		if args.save:
			colormap_img.save(os.path.join(OUT_DIR, 'colormap_%i.png' % args.quantization), 'PNG')

		imshow_kwargs = dict(
			origin='lower',
			extent=(moisture_proxy_range[0], moisture_proxy_range[1], temp_range[0], temp_range[1]),
			aspect=((moisture_proxy_range[1] - moisture_proxy_range[0]) / (temp_range[1] - temp_range[0])),
		)

		fig, axes = plt.subplots(1, 3)
		# fig, axes = plt.subplots(1, 2)
		fig.set_tight_layout(True)

		axes[0].imshow(colormap_img, **imshow_kwargs)
		axes[0].set_title('Colormap')

		axes[1].imshow(colormap_tidied, **imshow_kwargs)
		axes[1].set_title('Tidied')

		colormap_num_points = colormap_num_points.astype(float)
		colormap_num_points = np.log10(colormap_num_points + 1)
		axes[-1].imshow(colormap_num_points, **imshow_kwargs)
		axes[-1].set_title('log Num points')

		for ax in axes:
			ax.set_xlabel(moisture_proxy_name)
			ax.set_ylabel('Temperature')

	tprint('Done', start=start_time)

	# if args.show_plots:
	# 	tprint('Showing plots')
	# 	plt.show()

	tprint('Showing plots')
	plt.show()


def main():
	args = parse_args()
	run(args)


if __name__ == "__main__":
	main()
