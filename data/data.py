#!/usr/bin/env python

# data from:
# https://visibleearth.nasa.gov/view_cat.php?categoryID=1484
# https://neo.sci.gsfc.nasa.gov/dataset_index.php

# TODO: bulk download in script: https://neo.gsfc.nasa.gov/about/bulk.php

import argparse
from functools import lru_cache
import os
from os import PathLike
from pathlib import Path
from typing import Optional, Tuple, Iterable
import warnings

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import requests
from tqdm import trange, tqdm

from utils.image import array_to_image, disable_pil_max_pixels, resize_array, sphere_gradient
import utils.numeric
from utils.numeric import rescale, data_range, max_abs
from utils.utils import tprint


"""
TODO:

There are a lot of ways this could be cleaned up:
- De-duplicate a lot of code
- Use multiprocessing for averaging (possibly also for downloading?)

However, it's not really worth doing, because I would have to re-download all the data (a few GB worth) to test this
"""


# NASA data appears to have opposite endianness from what PIL expects
FLOAT_TIFF_SWAP_ENDIANNESS = True


"""
Use data 2000-2009:
- A decade is plenty of data to average
- Much of this data is only available starting around roughly 2000 (since launch of Terra/Aqua)
- NASA "Blue Marble" images we're basing this off of were released 2004-2005
  - unclear exactly what date range this is supposed to cover, but want data from around same period
"""
DATA_START_YEAR = 2000
DATA_END_YEAR = 2010


DEFAULT_REQUESTS_TIMEOUT_S = 30


ANTARCTICA_TEMP_CORRECTION_LATITUDE_S = 55
ANTARCTICA_TEMP_CORRECTION_INVALID_TEMP_C = 10


_AVERAGE_CACHE_DIR = Path('data_cache') / 'average'
_DOWNLOAD_CACHE_DIR = Path('data_cache') / 'download'


_load_cached_averages = True


def _import_img(
		# filepath: PathLike,
		*filepath,
		verbose=False,
		as_float=True,
		data_range: Optional[Tuple[float, float]] = None,
		pixel_range: Optional[Tuple[int, int]] = None,
		pixel_out_of_range_value=np.nan,
		very_large=False,
		):

	if len(filepath) == 1 and isinstance(filepath[0], Path):
		filepath = filepath[0]
	else:
		filepath = Path(os.path.join(*filepath))

	if (data_range or pixel_range) and not as_float:
		warnings.warn('data_range and pixel_range do nothing with as_float=False')

	if verbose:
		tprint('Importing file: %s' % filepath)

	is_float = str(filepath).lower().endswith('.float.tiff')

	if is_float and not as_float:
		raise ValueError('Cannot use as_float=False for file that is already in float format')

	with disable_pil_max_pixels(very_large):
		with Image.open(str(filepath)) as im:

			if is_float:
				data = np.array(im)

				if FLOAT_TIFF_SWAP_ENDIANNESS:
					data.byteswap(inplace=True)

				# Like with CSV, data uses 99999 as NaN
				data[data > 99998] = np.nan
				return data

			if as_float and (pixel_range is None):
				data = np.asarray(im, dtype=float) / 255.
			else:
				data = np.asarray(im)

	if (not is_float) and as_float and (pixel_range is not None):
		valid_pixel_mask = np.logical_and(data >= pixel_range[0], data <= pixel_range[1])
		data = np.asarray(im, dtype=float) / 255.
		invalid_pixel_mask = np.logical_not(valid_pixel_mask)
		data[invalid_pixel_mask] = pixel_out_of_range_value

	if as_float:
		assert data.dtype == float
		if data_range is not None:
			rescale(data, range_in=(0., 1.), range_out=data_range, in_place=True)
	else:
		assert data.dtype == np.uint8

	if verbose:
		tprint('Shape: %s, Range: [%f, %f]' % (data.shape, *utils.numeric.data_range(data)))
	return data


def _import_csv(filename: PathLike, dtype=float, verbose=False):

	if verbose:
		tprint('Importing file: %s' % filename)

	data = np.loadtxt(filename, dtype=dtype, delimiter=',')

	data[data > 99998] = np.nan

	if verbose:
		tprint('Shape: %s, Range: [%f, %f]' % (data.shape, *data_range(data)))

	return data


def _download_cache_file(url: str, filename: Optional[str]=None, subdir=None, timeout=DEFAULT_REQUESTS_TIMEOUT_S, verbose=True) -> Path:

	if not filename:
		url_split = url.split('/')
		if not url_split:
			raise ValueError(f'Invalid URL: {url}')
		filename = url_split[-1].rstrip()
		if not filename:
			raise ValueError(f'Invalid URL: {url}')

	filepath = (_DOWNLOAD_CACHE_DIR / subdir / filename) if subdir is not None else (_DOWNLOAD_CACHE_DIR / filename)

	filepath.parent.mkdir(parents=True, exist_ok=True)

	if not filepath.exists():
		if verbose:
			tprint(f'Downloading data from {url}')

		# For some reason urllib gives 403 for most of these, but requests works
		# (And no, it doesn't seem to be a User-Agent problem, at least as far as I've tried)

		r = requests.get(url, stream=True, timeout=timeout)
		if r.status_code != 200:
			raise IOError(f'Request to {url} returned HTTP {r.status_code}')

		if verbose:
			tprint(f'Returned {len(r.content)} bytes; saving as {filepath}')

		filepath.write_bytes(r.content)

	if not filepath.exists():
		raise ValueError(f'Failed to download {url}')

	return filepath


def _load_cached_average(filename: str, subdir=None) -> Optional[np.ndarray]:
	if not _load_cached_averages:
		return None

	cache_dir = _AVERAGE_CACHE_DIR if subdir is None else (_AVERAGE_CACHE_DIR / subdir)

	npy_filepath = cache_dir / (filename + '.npy')

	if npy_filepath.exists():
		return np.load(npy_filepath)

	return None


def _save_cached_average(data: np.ndarray, filename: str, subdir=None) -> None:

	cache_dir = _AVERAGE_CACHE_DIR if subdir is None else (_AVERAGE_CACHE_DIR / subdir)

	cache_dir.mkdir(parents=True, exist_ok=True)

	npy_filepath = cache_dir / (filename + '.npy')
	img_filepath = cache_dir / (filename + '.png')

	np.save(npy_filepath, data)

	data_rescaled = rescale(data)
	img = array_to_image(data_rescaled, nan=(1, 0, 1))
	img.save(img_filepath, format='PNG')



@lru_cache(maxsize=1)
def _get_elevation_standard_res():
	# High precision (32-bit float), but not high res
	url = 'https://neo.gsfc.nasa.gov/archive/geotiff.float/SRTM_RAMP2_TOPO/SRTM_RAMP2_TOPO_2000.FLOAT.TIFF'
	filename = _download_cache_file(url=url, subdir='topography')
	return _import_img(filename)


def _get_elevation_high_res():
	# Don't cache, it's too large
	# TODO: this is high res, but not high precision (only 8 bits!) - is there a better data source?
	url = 'https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73934/gebco_08_rev_elev_21600x10800.png'
	filename = _download_cache_file(url=url, subdir='topography')
	return _import_img(
		filename,
		very_large=True,
		data_range=(0., 6400.),
		pixel_range=(0, 254),
	)


def get_elevation(high_res=False, ocean_nan=False):
	"""
	:param high_res: Return topography 21600x10800 instead of 3600x1800 (but only currently 8-bit precision!)
	:param ocean_nan: if False, oceans will be reported as 0 meters; if True, they will be np.nan
	:returns: Elevation in meters, range [0, 6400]
	"""

	topo = _get_elevation_high_res() if high_res else _get_elevation_standard_res()

	if ocean_nan:
		return np.copy(topo)
	else:
		return np.nan_to_num(topo, nan=0.0)


@lru_cache(maxsize=1)
def _get_bathymetry_data():
	url = 'https://neo.gsfc.nasa.gov/archive/geotiff.float/GEBCO_BATHY/GEBCO_BATHY_2002.FLOAT.TIFF'
	filename = _download_cache_file(url=url, subdir='topography')
	return _import_img(filename)


def get_bathymetry(land_nan=False):
	"""
	:returns: Bathymetry in meters, range [-8000, 0]
	"""

	bathy = _get_bathymetry_data()

	if land_nan:
		return np.copy(bathy)
	else:
		return np.nan_to_num(bathy, nan=0.0)


@lru_cache(maxsize=1)
def get_topography():
	"""
	Get full topography, i.e. including both elevation & bathymetry
	:returns: Topography in meters, range [-8000, 6400]
	"""
	return get_elevation(ocean_nan=False) + get_bathymetry(land_nan=False)


def get_land_ocean_ice(resize=True, as_img=False):
	filename = os.path.join('data', 'nasa_blue_marble', 'land_ocean_ice_8192.png')
	img = Image.open(filename)

	if resize:
		img = img.resize((3600, 1800))

	if as_img:
		return img
	else:
		return np.asarray(img, dtype=float) / 255


def _average_dir(
		dir_name: PathLike,
		file_prefix: str,
		**kwargs
		):
	"""
	:param any_nan_makes_result_nan: if True, any one invalid pixel will cause resulting averaged pixel to be NaN; if False, pixel will be skipped in averaging
	"""
	dir_path = Path('data') / 'nasa_earth_observatory' / dir_name
	files = [
		dir_path / filename
		for filename in os.listdir(dir_path)
		if filename.startswith(file_prefix) and not (dir_path / filename).is_dir() and not filename.suffix.lower() == '.act'
	]

	if not files:
		raise FileNotFoundError(f'No images in directory {dir_name}')

	files.sort()
	return _average_files(files, **kwargs)


def _average_files(
		files: Iterable[Path],
		verbose=False,
		data_range: Optional[Tuple[float, float]] = None,
		pixel_range: Optional[Tuple[int, int]] = None,
		pixel_out_of_range_value=np.nan,
		missing_data_fill_mask: Optional[np.ndarray] = None,
		missing_data_fill_value=0.0,
		any_nan_makes_result_nan=False,
		):

	img_sum = None
	n_img_per_pixel = None
	n_img_total = 0

	for filepath in files:
		assert os.path.isfile(filepath)
		filename = filepath.name

		ext = filepath.suffix[1:].lower()

		# FIXME: doesn't work with tiff, n_img_per_pixel ends up None
		if ext in ['jpg', 'png', 'tiff']:
			data = _import_img(filepath, verbose=verbose, data_range=data_range, pixel_range=pixel_range)
		elif ext == 'csv':
			data = _import_csv(filepath, verbose=verbose)
		else:
			print(f'WARNING: Unknown extension for filename: {filename} (ext: {ext})')
			continue

		if img_sum is None:
			img_sum = np.zeros_like(data)

		if data.shape != img_sum.shape:
			if 'scale' in filename.lower():
				continue
			raise RuntimeError(f'File {filename} has mismatched dimensions! (expected {img_sum.shape}, got {data.shape})')

		data_is_nan = np.isnan(data)

		if missing_data_fill_mask is not None:
			fill_mask = np.logical_and(data_is_nan, missing_data_fill_mask)
			if isinstance(missing_data_fill_value, np.ndarray):
				data[fill_mask] = missing_data_fill_value[fill_mask]
			else:
				data[fill_mask] = missing_data_fill_value

			data_is_nan = np.isnan(data)

		valid_pixel_mask = np.logical_not(data_is_nan)
		if n_img_per_pixel is None:
			n_img_per_pixel = np.zeros_like(data, dtype=int)
		img_sum[valid_pixel_mask] += data[valid_pixel_mask]
		n_img_per_pixel += valid_pixel_mask

		n_img_total += 1

	if not n_img_total:
		raise ValueError(f'No images were averaged: {files}')

	if (n_img_per_pixel is None) or ((n_img_per_pixel if np.isscalar(n_img_per_pixel) else np.amax(n_img_per_pixel)) == 0):
		raise ValueError(f'Files were averaged, but something went wrong - no resulting data: {n_img_per_pixel=}')

	if np.isscalar(n_img_per_pixel):
		averaged = img_sum / n_img_total
	else:
		assert n_img_per_pixel is not None
		if any_nan_makes_result_nan:
			zero_mask = (n_img_per_pixel < n_img_total)
		else:
			zero_mask = (n_img_per_pixel == 0)
		n_img_per_pixel = np.maximum(n_img_per_pixel, 1)
		averaged = img_sum.astype(float) / n_img_per_pixel.astype(float)
		averaged[zero_mask] = pixel_out_of_range_value

	return averaged


def _average_arrays(
		arrays: Iterable[np.ndarray],
		missing_data_fill_mask: Optional[np.ndarray] = None,
		missing_data_fill_value=0.0,
		any_nan_makes_result_nan=False,
		):

	if not arrays:
		raise ValueError('No arrays!')

	img_sum = None
	n_img_per_pixel = None
	n_img_total = 0

	for data in arrays:

		if img_sum is None:
			img_sum = np.zeros_like(data)

		if data.shape != img_sum.shape:
			raise RuntimeError(f'Arrays have mismatched dimensions! (expected {img_sum.shape}, got {data.shape})')

		data_is_nan = np.isnan(data)

		if (missing_data_fill_mask is not None) and data_is_nan.any():
			fill_mask = np.logical_and(data_is_nan, missing_data_fill_mask)
			if isinstance(missing_data_fill_value, np.ndarray):
				data[fill_mask] = missing_data_fill_value[fill_mask]
			else:
				data[fill_mask] = missing_data_fill_value

			data_is_nan = np.isnan(data)

		valid_pixel_mask = np.logical_not(data_is_nan)
		if n_img_per_pixel is None:
			n_img_per_pixel = np.zeros_like(data, dtype=int)
		img_sum[valid_pixel_mask] += data[valid_pixel_mask]
		n_img_per_pixel += valid_pixel_mask

		n_img_total += 1

	assert n_img_per_pixel is not None

	if any_nan_makes_result_nan:
		zero_mask = (n_img_per_pixel < n_img_total)
	else:
		zero_mask = (n_img_per_pixel == 0)
	n_img_per_pixel = np.maximum(n_img_per_pixel, 1)
	averaged = img_sum.astype(float) / n_img_per_pixel.astype(float)
	averaged[zero_mask] = np.nan

	return averaged


@lru_cache(maxsize=1)
def get_average_wv():
	# Range 0-6 cm

	# TODO: make a decorator for caching
	cached = _load_cached_average('average_water_vapor')
	if cached is not None:
		return cached

	# TODO: no need to keep all the files once everything is cached - it takes > 1 GB!
	# (or rather, make this optional)
	# (same for other datasets)

	# TODO: tqdm progress bar (same for other datasets)
	files_by_month = [[] for _ in range(12)]
	for year in range(DATA_START_YEAR, DATA_END_YEAR):
		for month_idx in range(12):
			month_num = 1 + month_idx

			if (year, month_num) >= (2000, 2):
				filename ='MODAL2_M_SKY_WV_%i-%02i.FLOAT.TIFF' % (year, month_num)
				url = 'https://neo.gsfc.nasa.gov/archive/geotiff.float/MODAL2_M_SKY_WV/' + filename
				filepath = _download_cache_file(url=url, subdir='water_vapor')
				files_by_month[month_idx].append(filepath)

			if (year, month_num) >= (2002, 7):
				filename = 'MYDAL2_M_SKY_WV_%i-%02i.FLOAT.TIFF' % (year, month_num)
				url = 'https://neo.gsfc.nasa.gov/archive/geotiff.float/MYDAL2_M_SKY_WV/' + filename
				filepath = _download_cache_file(url=url, subdir='water_vapor')
				files_by_month[month_idx].append(filepath)

	month_averages = [_average_files(files, any_nan_makes_result_nan=False) for files in files_by_month]
	average = _average_arrays(month_averages, any_nan_makes_result_nan=True)

	_save_cached_average(average, 'average_water_vapor')
	return average


@lru_cache(maxsize=2)
def get_average_veg(assume_pole_missing_data_zero=True):
	# Range [-0.1, 0.9]

	cache_filename = 'average_vegetation'
	if assume_pole_missing_data_zero:
		cache_filename += '_poles_filled_in'

	cached = _load_cached_average(cache_filename)
	if cached is not None:
		return cached

	# Not all files will have 100% coverage, some have height cut off
	missing_data_zero_mask = None
	if assume_pole_missing_data_zero:
		missing_data_zero_mask = np.copy(get_mask(land=True))
		assert missing_data_zero_mask.shape[0] == 1800
		missing_data_zero_mask[300:-300, :] = False

	files_by_month = [[] for _ in range(12)]
	for year in range(DATA_START_YEAR, DATA_END_YEAR):
		for month_idx in range(12):
			month_num = 1 + month_idx
			if (year, month_num) < (2000, 3):
				continue
			filename ='MOD_NDVI_M_%i-%02i.FLOAT.TIFF' % (year, month_num)
			url = 'https://neo.gsfc.nasa.gov/archive/geotiff.float/MOD_NDVI_M/' + filename
			filepath = _download_cache_file(url=url, subdir='vegetation')
			files_by_month[month_idx].append(filepath)

	month_averages = [
		_average_files(
			files,
			any_nan_makes_result_nan=False,
			missing_data_fill_mask=missing_data_zero_mask,
			missing_data_fill_value=0.0,
		) for files in files_by_month]
	average = _average_arrays(month_averages, any_nan_makes_result_nan=True)

	_save_cached_average(average, cache_filename)
	return average


@lru_cache(maxsize=1)
def get_average_rain():
	# Range [1, 2000]

	cached = _load_cached_average('average_rainfall')
	if cached is not None:
		return cached

	files_by_month = [[] for _ in range(12)]
	for year in range(DATA_START_YEAR, DATA_END_YEAR):
		for month_idx in range(12):
			month_num = 1 + month_idx
			if (year, month_num) < (2000, 6):
				continue
			filename ='GPM_3IMERGM_%i-%02i.FLOAT.TIFF' % (year, month_num)
			url = 'https://neo.gsfc.nasa.gov/archive/geotiff.float/GPM_3IMERGM/' + filename
			filepath = _download_cache_file(url=url, subdir='rainfall')
			files_by_month[month_idx].append(filepath)

	month_averages = [_average_files(files, any_nan_makes_result_nan=False) for files in files_by_month]
	average = _average_arrays(month_averages, any_nan_makes_result_nan=True)

	_save_cached_average(average, 'average_rainfall')
	return average


@lru_cache(maxsize=1)
def get_antarctica_greenland_mask() -> np.ndarray:

	url = 'https://neo.gsfc.nasa.gov/archive/gs/ICESAT_ELEV_G/ICESAT_ELEV_G_2003.PNG'
	filename = _download_cache_file(url=url, subdir='topography')
	mask = _import_img(filename, as_float=True, pixel_range=(0, 254))

	# Actual is 8640 x 4320, resize to 3600x1800
	mask = resize_array(mask, (3600, 1800))

	mask = np.logical_not(np.isnan(mask))

	# greenland_antarctica_topography.png is missing some data right near south pole
	assert mask.shape[0] == 1800, f'{mask.shape=}'
	mask[1700:, :] = True

	# _antarctica_greenland_mask = mask
	return np.copy(mask)


@lru_cache(maxsize=1)
def _get_land_mask():
	url = 'https://neo.gsfc.nasa.gov/archive/gs/MOD_NDVI_M/MOD_NDVI_M_2002-09.PNG'
	filename = _download_cache_file(url=url, subdir='vegetation')
	veg_as_mask = _import_img(filename, as_float=False)
	antarctica_mask = get_antarctica_greenland_mask()
	mask = np.logical_or(veg_as_mask < 255, antarctica_mask)
	assert mask.dtype == bool
	return mask


@lru_cache(maxsize=1)
def _get_ocean_mask():
	topo = get_elevation(ocean_nan=True)
	mask = np.isnan(topo)
	assert mask.dtype == bool
	return mask


def get_mask(land=False, ocean=False, lakes=False) -> np.ndarray:
	"""
	:param lakes: includes all inland water, not just lakes
	"""

	land_mask = _get_land_mask()
	ocean_mask = _get_ocean_mask()

	# Don't have good enough match between land & ocean masks, so lake mask doesn't work all that well
	# (ends up with extra data at ocean shores)
	# lake_mask = np.logical_not(np.logical_or(land_mask, ocean_mask))
	# assert lake_mask.dtype == bool

	if land and ocean and lakes:
		warnings.warn('Full mask!')
		return np.ones_like(land_mask, dtype=bool)
	elif land and ocean:
		# return np.logical_not(lake_mask)
		raise NotImplementedError('Land + Ocean (no lakes) mask is not currently supported!')
	elif land and lakes:
		return np.logical_not(ocean_mask)
	elif ocean and lakes:
		return np.logical_not(land_mask)
	elif land:
		return land_mask
	elif ocean:
		return ocean_mask
	elif lakes:
		# return lake_mask
		raise NotImplementedError('Lakes-only mask is not currently supported!') 
	else:
		warnings.warn('Empty mask!')
		return np.zeros_like(land_mask, dtype=bool)


@lru_cache(maxsize=1)
def _get_average_day_temperature():

	average_day = _load_cached_average('average_day', subdir='surface_temperature')

	if average_day is None:
		tprint('Averaging day temperatures...')

		averages_by_month = []
		for month_idx in trange(12):
			month_num = 1 + month_idx
			files = []
			for year in range(DATA_START_YEAR, DATA_END_YEAR):

				if (year, month_num) < (2000, 2):
					continue

				filename = 'MOD_LSTD_M_%i-%02i.FLOAT.TIFF' % (year, month_num)
				url = 'https://neo.gsfc.nasa.gov/archive/geotiff.float/MOD_LSTD_M/' + filename
				filepath = _download_cache_file(url=url, subdir='surface_temperature', verbose=False)
				files.append(filepath)

			month_average = _average_files(files, any_nan_makes_result_nan=False)
			averages_by_month.append(month_average)

		average_day = _average_arrays(averages_by_month, any_nan_makes_result_nan=True)
		del averages_by_month
		_save_cached_average(average_day, 'average_day', subdir='surface_temperature')
	
	return average_day


@lru_cache(maxsize=1)
def _get_average_night_temperature():
	average_night = _load_cached_average('average_night', subdir='surface_temperature')

	if average_night is None:
		tprint('Averaging night temperatures...')

		averages_by_month = []
		for month_idx in trange(12):
			month_num = 1 + month_idx
			files = []
			for year in range(DATA_START_YEAR, DATA_END_YEAR):

				if (year, month_num) < (2000, 2):
					continue

				filename = 'MOD_LSTN_M_%i-%02i.FLOAT.TIFF' % (year, month_num)
				url = 'https://neo.gsfc.nasa.gov/archive/geotiff.float/MOD_LSTN_M/' + filename
				filepath = _download_cache_file(url=url, subdir='surface_temperature', verbose=False)
				files.append(filepath)

			month_average = _average_files(files, any_nan_makes_result_nan=False)
			averages_by_month.append(month_average)

		average_night = _average_arrays(averages_by_month, any_nan_makes_result_nan=True)
		del averages_by_month
		_save_cached_average(average_night, 'average_night', subdir='surface_temperature')
	
	return average_night


@lru_cache(maxsize=1)
def _get_average_ocean_temperature():
	average_ocean = _load_cached_average('average_ocean', subdir='surface_temperature')
	if average_ocean is None:

		tprint('Averaging ocean temperatures...')

		averages_by_month = []
		for month_idx in trange(12):
			month_num = 1 + month_idx
			files = []
			for year in range(DATA_START_YEAR, DATA_END_YEAR):

				if (year, month_num) < (2002, 7):
					continue

				filename ='MYD28M_%i-%02i.FLOAT.TIFF' % (year, month_num)
				url = 'https://neo.gsfc.nasa.gov/archive/geotiff.float/MYD28M/' + filename
				filepath = _download_cache_file(url=url, subdir='surface_temperature', verbose=False)
				files.append(filepath)

			# NaN values are a huge problem here, since they often indicate ice
			# Could fill NaN with freezing point of seawater, but this wouldn't indicate actual ice temperature, which
			# is what we want
			month_average = _average_files(files, any_nan_makes_result_nan=True)
			averages_by_month.append(month_average)

		average_ocean = _average_arrays(averages_by_month, any_nan_makes_result_nan=True)
		del averages_by_month
		_save_cached_average(average_ocean, 'average_ocean', subdir='surface_temperature')

	return average_ocean


def get_average_surface_temp(land_day=True, land_night=True, ocean=False):

	average = None

	if land_day:
		average = _get_average_day_temperature()

	if land_night:
		average_night = _get_average_night_temperature()
		if average is None:
			average = average_night
		else:
			average = 0.5 * (average + average_night)

	if ocean:
		average_ocean = _get_average_ocean_temperature()

		if average is None:
			average = average_ocean
		else:
			average_nan = np.isnan(average)
			average[average_nan] = average_ocean[average_nan]

	return average


def sat_vapor_density_from_temp(t):
	# http://hyperphysics.phy-astr.gsu.edu/hbase/Kinetic/relhum.html
	# http://hyperphysics.phy-astr.gsu.edu/hbase/Kinetic/watvap.html
	t2 = np.square(t)
	t3 = t2 * t
	return \
		5.018 + \
		0.32321 * t + \
		8.1847 * t2 + \
		3.1243 * t3


# def wv_to_rel_humid(wv, t):
# 	svd = sat_vapor_density_from_temp(t)

# 	# wv is in cm, range 0-6
# 	# vapor density is in g/m3, range ~0-50 for T <= 40C

# 	pass  # TODO


def parse_args(args=None) -> argparse.Namespace:
	parser = argparse.ArgumentParser()

	parser.add_argument('--recalculate', action='store_true', help="Recalculate averages instead of using cached (still keeps cached downloads)")

	g = parser.add_argument_group('Specific data')
	g.add_argument('--color',     action='store_true', help='Color land/ocean/ice image')
	g.add_argument('--topo',      action='store_true', dest='topography', help='Topography')
	g.add_argument('--gradient',  action='store_true', help='Gradient')
	g.add_argument('--masks',     action='store_true', help='Masks')
	g.add_argument('--veg',       action='store_true', dest='vegetation', help='Vegeation, water vapor, rainfall')
	g.add_argument('--temp',      action='store_true', dest='temperature', help='Temperature')

	return parser.parse_args(args)


def _imshow(im: np.ndarray, title='', *, cmap='gray', nan=None, **kwargs):

	if nan is not None:
		im = np.ma.array(im, mask=np.isnan(im))
		cmap = plt.get_cmap(name=cmap)
		cmap.set_bad(nan, alpha=1.0)

	fig, ax = plt.subplots(1, 1)
	fig.set_tight_layout(True)
	if title:
		fig.suptitle(title)
	imshow_result = ax.imshow(im, interpolation='none', cmap=cmap, **kwargs)
	plt.colorbar(imshow_result)


def _show_gradient(elevation, gmag, gx, gy, title=''):
	fig, ax = plt.subplots(2, 2)
	fig.set_tight_layout(True)
	if title:
		fig.suptitle(title)
	
	imshow_result = ax[0][0].imshow(elevation, interpolation='none', cmap='gray')
	ax[0][0].set_title('Elevation')
	plt.colorbar(imshow_result, ax=ax[0][0])

	imshow_result = ax[0][1].imshow(gmag, interpolation='none', cmap='inferno')
	ax[0][1].set_title('Gradient magnitude')
	plt.colorbar(imshow_result, ax=ax[0][1])

	maxgradient = max(max_abs(gx), max_abs(gy))

	imshow_result = ax[1][0].imshow(gx, interpolation='none', cmap='seismic', vmin=-maxgradient, vmax=maxgradient)
	ax[1][0].set_title('Gradient X')
	plt.colorbar(imshow_result, ax=ax[1][0])

	imshow_result = ax[1][1].imshow(gy, interpolation='none', cmap='seismic', vmin=-maxgradient, vmax=maxgradient)
	ax[1][1].set_title('Gradient Y')
	plt.colorbar(imshow_result, ax=ax[1][1])


def main(args=None):
	global _load_cached_averages

	args = parse_args(args)

	if args.recalculate:
		_load_cached_averages = False

	if not any([args.color, args.topography, args.gradient, args.masks, args.vegetation, args.temperature]):
		args.color = args.topography = args.gradient = args.masks = args.vegetation = args.temperature = True

	if args.color:
		tprint('Processing color data...')
		land_ocean_ice = get_land_ocean_ice(resize=False, as_img=False)
		_imshow(land_ocean_ice, 'get_land_ocean_ice(resize=False)')
		land_ocean_ice = get_land_ocean_ice(resize=True, as_img=False)
		_imshow(land_ocean_ice, 'get_land_ocean_ice(resize=True)')
		del land_ocean_ice
		print()

	if args.topography:
		tprint('Processing topography & bathymetry data...')
		elevation = get_elevation()
		tprint('get_elevation() range: [%f, %f]' % data_range(elevation))
		_imshow(elevation, 'get_elevation()')
		elevation = get_elevation(ocean_nan=True)
		num_nan = np.sum(np.isnan(elevation))
		tprint(f'{num_nan} NaN values')
		_imshow(elevation, 'Elevation with ocean as NaN', nan=(1, 0, 1))
		elevation = get_elevation(high_res=True)
		tprint('get_elevation(high_res=True) shape: %s, range: [%f, %f]' % (elevation.shape, *data_range(elevation)))
		del elevation

		bathymetry = get_bathymetry()
		tprint('get_bathymetry() range: [%f, %f]' % data_range(bathymetry))
		_imshow(bathymetry, 'get_bathymetry()')
		del bathymetry

		topography = get_topography()
		tprint('get_topography() range: [%f, %f]' % data_range(topography))
		_imshow(topography, 'get_topography()', cmap='gist_earth')
		del topography
		print()

	if args.gradient:
		tprint('Processing gradient...')
		elevation = get_elevation()

		gx, gy = sphere_gradient(elevation, scale_earth=True, latitude_adjust=True, sobel=False)
		gm = np.sqrt(np.square(gx) + np.square(gy))
		tprint('sphere_gradient(sobel=False) range: mag [%f,%f] X [%f, %f] Y [%f, %f]' % (*data_range(gm), *data_range(gx), *data_range(gy)))
		_show_gradient(elevation, gm, gx, gy, 'sphere_gradient(sobel=False)')

		gx, gy = sphere_gradient(elevation, scale_earth=True, latitude_adjust=True, sobel=True)
		tprint('sphere_gradient(sobel=True) range: mag [%f,%f] X [%f, %f] Y [%f, %f]' % (*data_range(gm), *data_range(gx), *data_range(gy)))
		_show_gradient(elevation, gm, gx, gy, 'sphere_gradient(sobel=True)')

		gx, gy = sphere_gradient(elevation, scale_earth=True, latitude_adjust=True, large_sobel=True)
		tprint('sphere_gradient(large_sobel=True) range: mag [%f,%f] X [%f, %f] Y [%f, %f]' % (*data_range(gm), *data_range(gx), *data_range(gy)))
		_show_gradient(elevation, gm, gx, gy, 'sphere_gradient(large_sobel=True)')

		gx, gy = sphere_gradient(elevation, scale_earth=True, latitude_adjust=False, sobel=True)
		tprint('sphere_gradient(sobel=True, latitude_adjust=False) range: mag [%f,%f] X [%f, %f] Y [%f, %f]' % (*data_range(gm), *data_range(gx), *data_range(gy)))
		_show_gradient(elevation, gm, gx, gy, 'sphere_gradient(sobel=True, latitude_adjust=False)')

		del elevation, gx, gy
		print()

	if args.masks:
		tprint('Processing masks...')
		mask = get_mask(land=True)
		_imshow(mask, 'Land mask')
		mask = get_mask(land=True, lakes=True)
		_imshow(mask, 'Land + Lake mask')
		mask = get_mask(ocean=True)
		_imshow(mask, 'Ocean mask')
		mask = get_mask(ocean=True, lakes=True)
		_imshow(mask, 'Ocean + Lake mask')
		mask = get_antarctica_greenland_mask()
		_imshow(mask, 'Greenland + Antarctica mask')
		del mask
		print()

	if args.vegetation:
		tprint('Processing rain & vegetation...')
		wv = get_average_wv()
		tprint('get_average_wv() range: [%f, %f]' % data_range(wv))
		_imshow(wv, 'get_average_wv()', cmap='YlGn', nan=(0, 0, 0))
		del wv

		veg = get_average_veg(assume_pole_missing_data_zero=True)
		tprint('get_average_veg(assume_pole_missing_data_zero=True) range: [%f, %f]' % data_range(veg))
		_imshow(veg, 'get_average_veg(assume_pole_missing_data_zero=True)', cmap='YlGn', nan=(0, 0, 0))
		veg = get_average_veg(assume_pole_missing_data_zero=False)
		tprint('get_average_veg(assume_pole_missing_data_zero=False) range: [%f, %f]' % data_range(veg))
		_imshow(veg, 'get_average_veg(assume_pole_missing_data_zero=False)', cmap='YlGn', nan=(0, 0, 0))
		del veg

		rain = get_average_rain()
		tprint('get_average_rain() range: [%f, %f]' % data_range(rain))
		_imshow(rain, 'get_average_rain()', cmap='YlGn', nan=(0, 0, 0))
		del rain
		print()

	if args.temperature:
		tprint('Processing temperature...')
		average_temp = get_average_surface_temp(ocean=True)
		_imshow(average_temp, 'get_average_surface_temp(all)', cmap='coolwarm', nan=(0, 0, 0))
		average_temp = get_average_surface_temp(ocean=False)
		_imshow(average_temp, 'get_average_surface_temp(land)', cmap='coolwarm', nan=(0, 0, 0))
		average_temp = get_average_surface_temp(land_day=False, land_night=False, ocean=True)
		_imshow(average_temp, 'get_average_surface_temp(ocean only)', cmap='coolwarm', nan=(0, 0, 0))
		average_temp = get_average_surface_temp(land_day=True, land_night=False)
		_imshow(average_temp, 'get_average_surface_temp(land, day only)', cmap='coolwarm', nan=(0, 0, 0))
		average_temp = get_average_surface_temp(land_day=False, land_night=True)
		_imshow(average_temp, 'get_average_surface_temp(land, night only)', cmap='coolwarm', nan=(0, 0, 0))
		del average_temp
		print()

	tprint('Showing plots')
	plt.show()


if __name__ == "__main__":
	main()
