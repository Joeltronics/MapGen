#!/usr/bin/env python

import colorsys
from contextlib import contextmanager
from os import PathLike
from typing import Optional, Tuple, Union, Literal
import warnings

import colour
import numpy as np
from matplotlib import pyplot as plt
import scipy.signal
from PIL import Image

from utils.numeric import rescale, rescale_in_place, data_range, max_abs
import utils.numeric


PI = np.pi
TWOPI = 2 * np.pi

SHOW_IMG_USE_MATPLOTLIB = False

GRADIENT_CORRECT_WRAPPING = True

EARTH_POLAR_CIRCUMFERENCE_M = 40_007_863
EARTH_EQUATORIAL_CIRCUMFERENCE_M = 40_075_017
MIN_LATITUDE_ADJUST_CIRCUMFERENCE = EARTH_POLAR_CIRCUMFERENCE_M // 10


@contextmanager
def disable_pil_max_pixels(do_it: bool = True):
	if not do_it:
		yield None
		return
	max_pixels_was = Image.MAX_IMAGE_PIXELS
	Image.MAX_IMAGE_PIXELS = None
	try:
		yield None
	finally:
		Image.MAX_IMAGE_PIXELS = max_pixels_was


def linear_to_gamma(im: np.ndarray, gamma=2.2) -> np.ndarray:
	return np.power(im, 1.0/gamma)


def gamma_to_linear(im: np.ndarray, gamma=2.2) -> np.ndarray:
	return np.power(im, gamma)


def image_to_array(im, as_float=True) -> np.ndarray:
	if as_float:
		return np.asarray(im, dtype=float) / 255.
	else:
		return np.asarray(im)


def float_to_uint8(arr: np.ndarray, bipolar=False) -> np.ndarray:

	if bipolar:
		arr = arr + 1.0
		arr *= 255.0 / 2.0
	else:
		arr = arr * 255.0

	return arr.astype(np.uint8)


def _show_img_matplotlib(im: Union[np.ndarray, Image.Image], /, title='', cmap='gray'):
	fig, ax = plt.subplots(1, 1)
	if title:
		fig.suptitle(title)
	# TODO: colorbar
	ax.imshow(im, interpolation='none', cmap=cmap)
	plt.tight_layout()


def show_img(im: Union[np.ndarray, Image.Image], /, title='', *, bipolar=False, nan=None, rescale=False, clip=False):

	# TODO: argument to apply colormap

	if not SHOW_IMG_USE_MATPLOTLIB:
		if title:
			print('Showing image: ' + title)
		else:
			print('Showing image')

	if isinstance(im, np.ndarray):
		if rescale:
			if clip:
				warnings.warn('clip does nothing when rescale enabled')
			if bipolar:
				im /= max_abs(im)
			else:
				im = utils.numeric.rescale(im)
		elif clip:
			im = np.clip(im, -1 if bipolar else 0, 1)

		if not SHOW_IMG_USE_MATPLOTLIB:
			im = array_to_image(im, bipolar=bipolar, nan=nan)

	elif isinstance(im, Image.Image):
		if rescale or clip:
			raise ValueError('rescale & clip are only supported with numpy array, not PIL.Image')
		if SHOW_IMG_USE_MATPLOTLIB:
			im = image_to_array(im)

	else:
		raise TypeError(type(im).__name__)

	if SHOW_IMG_USE_MATPLOTLIB:
		assert isinstance(im, np.ndarray)
		_show_img_matplotlib(im, title=title)
	else:
		assert isinstance(im, Image.Image)
		im.show(title)


def array_to_image(arr: np.ndarray, bipolar=False, mode=None, nan=None) -> Image.Image:
	"""
	:param arr: Array with shape (height, width) or (height, width, num_channels)
	:param bipolar: If True, expected input range will be [-1, 1]; otherwise will be [0, 1]
	:param nan: Value to heal NaN values to - scalar, or array of size 3. If None, will throw if there are any NaN in arr
	"""

	# TODO: if dtype is already uint8, handle this
	# TODO: Option to make NaN the alpha channel
	# TODO: Also heal or clip infinity values (and/or clip, or warn if present)

	arr = arr.astype(float)

	if np.isnan(arr).any():
		if nan is None:
			raise ValueError('Array has NaN values, but no nan argument specified')

		if np.isscalar(nan):
			arr = np.nan_to_num(arr, nan=nan)
		else:
			if len(arr.shape) >= 3:
				if arr.shape[2] != 1:
					raise NotImplementedError(f'array_to_image() not implemented for {arr.shape=}, {nan=}')
				arr = arr.reshape((arr.shape[0], arr.shape[1]))
			nan_mask = np.isnan(arr)
			if len(arr.shape) == 2:
				arr = np.stack([arr, arr, arr], axis=-1)
			arr[nan_mask] = nan

	if bipolar:
		if (np.amin(arr) < -1 or np.amax(arr) > 1):
			print("WARNING: clipped output image; old image bounds: [%.3f,%.3f]" % (np.amin(arr),np.amax(arr)))
			arr = np.clip(arr, -1., 1.)
		arr = (arr + 1) / 2
	else:
		if (np.amin(arr) < 0 or np.amax(arr) > 1):
			print("WARNING: clipped output image; old image bounds: [%.3f,%.3f]" % (np.amin(arr),np.amax(arr)))
			arr = np.clip(arr, 0., 1.)

	arr_int = np.floor(arr * 255).astype(np.uint8)

	return Image.fromarray(arr_int, mode=mode)


def resize_array(
		arr: np.ndarray,
		new_size: Tuple[int, int],
		data_range: Optional[Tuple[float, float]] = None,
		resampling=Image.BILINEAR,
		verbose=False,
		) -> np.ndarray:
	"""
	:note: Loses a lot of data precision due to conversion to uint8!
	"""

	nan_mask = np.isnan(arr)
	any_nan = nan_mask.any()

	mode = None
	num_channels = None

	# TODO: get this working while keeping image as float, or even at least uint16
	dtype = np.uint8
	dtype_max_val = 255

	if data_range is None:
		data_range = utils.numeric.data_range(arr)
	arr = np.floor(rescale(arr, range_in=data_range, range_out=(0., dtype_max_val))).astype(dtype)

	if any_nan:
		arr = np.nan_to_num(arr, nan=0.0)

		if len(arr.shape) == 2:
			arr = arr.reshape((arr.shape[0], arr.shape[1], 1))
		num_channels = arr.shape[2]

		# FIXME: this likely doesn't work properly if arr is RGB
		alpha_channel = dtype_max_val * np.ones_like(arr)
		alpha_channel[nan_mask] = 0
		assert arr.shape[2] == num_channels
		arr = np.stack((arr, alpha_channel), axis=2)
		assert arr.shape[2] == num_channels + 1

		if num_channels == 1:
			mode = 'LA'
		elif num_channels == 3:
			mode = 'RGBA'
		else:
			raise ValueError(f'Cannot handle array with NaN and {num_channels} channels')

		if verbose:
			for channel in range(num_channels + 1):
				print(f'Channel {channel}, range: {utils.numeric.data_range(arr[:, :, channel])}')

	im = Image.fromarray(arr, mode=mode)
	im = im.resize(new_size, resample=resampling)
	arr = np.asarray(im, dtype=float)

	if any_nan:
		assert num_channels is not None

		if verbose:
			print(f'Image shape: {arr.shape}')
			for channel in range(num_channels + 1):
				print(f'Channel {channel}, range: {utils.numeric.data_range(arr[..., channel])}')

		alpha_channel_after = arr[..., -1]

		alpha_thresh = 1
		alpha_mask_after = alpha_channel_after < alpha_thresh
		assert alpha_mask_after.any()  # TODO: remove this assertion later
		assert not alpha_mask_after.all()  # TODO: remove this assertion later

		arr = arr[..., :-1]
		assert arr.shape[2] == num_channels

		for channel in range(num_channels):
			arr[..., channel][alpha_mask_after] = np.nan

		if num_channels == 1:
			arr = arr.reshape((arr.shape[0], arr.shape[1]))

	rescale_in_place(arr, range_in=(0., dtype_max_val), range_out=data_range)

	return arr


def _pad_sphere_image_for_convolution(im: np.ndarray, large_sobel: bool) -> np.ndarray:
	height, width = im.shape

	first_col = im[:, :1]
	last_col = im[:, -1:]
	if large_sobel:
		im = np.concatenate((last_col, last_col, im, first_col, first_col), axis=1)
		assert im.shape == (height, width + 4)
	else:
		im = np.concatenate((last_col, im, first_col), axis=1)
		assert im.shape == (height, width + 2)

	first_row = im[:1, :]
	last_row = im[-1:, :]
	if large_sobel:
		im = np.concatenate((first_row, first_row, im, last_row, last_row), axis=0)
		assert im.shape == (height + 4, width + 4)
	else:
		im = np.concatenate((first_row, im, last_row), axis=0)
		assert im.shape == (height + 2, width + 2)
	
	return im


def gradient(
		im: np.ndarray, /,
		scale01=False, magnitude=False, sobel=True, large_sobel=False,
		) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

	height, width = im.shape

	if large_sobel:
		gx = np.array([
			[1, 2, 0, -2, -1],
			[4, 8, 0, -8, -4],
			[6, 12, 0, -12, -6],
			[4, 8, 0, -8, -4],
			[1, 2, 0, -2, -1]])
	elif sobel:
		gx = np.array([
			[1, 0, -1],
			[2, 0, -2],
			[1, 0, -1]])
	else:
		gx = np.array([[1, 0, -1]])

	kernel_magnitude = np.sum(np.abs(gx))
	gx = gx / kernel_magnitude

	gy = np.transpose(gx)

	gradient_x = scipy.signal.convolve2d(im, gx, mode='same')
	gradient_y = scipy.signal.convolve2d(im, gy, mode='same')

	assert gradient_x.shape == gradient_y.shape == (height, width)

	if magnitude:
		gradient_mag = np.sqrt(np.square(gradient_x), np.square(gradient_y))
		if scale01:
			gradient_mag = rescale(gradient_mag, data_range(gradient_mag), (0., 1.))
		return gradient_mag
	else:
		if scale01:
			max_abs_gradient = max(np.amax(gradient_x), np.amax(gradient_y))
			gradient_x = rescale(gradient_x, (-max_abs_gradient, max_abs_gradient), (-1., 1.))
			gradient_y = rescale(gradient_y, (-max_abs_gradient, max_abs_gradient), (-1., 1.))
		return gradient_x, gradient_y


def sphere_gradient(
		im: np.ndarray, /,
		scale01=False, magnitude=False, latitude_adjust=True, sobel=True, large_sobel=False,
		) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

	height, width = im.shape
	assert width == 2 * height

	if large_sobel:
		gx = np.array([
			[1, 2, 0, -2, -1],
			[4, 8, 0, -8, -4],
			[6, 12, 0, -12, -6],
			[4, 8, 0, -8, -4],
			[1, 2, 0, -2, -1]])
	elif sobel:
		gx = np.array([
			[1, 0, -1],
			[2, 0, -2],
			[1, 0, -1]])
	else:
		gx = np.array([[1, 0, -1]])

	kernel_magnitude = np.sum(np.abs(gx))
	gy = np.transpose(gx)

	# Want to wrap image in X dimension, but not in Y (only matters for Sobel, not for 1D)
	if GRADIENT_CORRECT_WRAPPING and (sobel or large_sobel):
		im = _pad_sphere_image_for_convolution(im, large_sobel=large_sobel)
		assert im.shape == (height + 4, width + 4) if large_sobel else (height + 2, width + 2)
		gradient_x = scipy.signal.convolve2d(im, gx, mode='valid')
		gradient_y = scipy.signal.convolve2d(im, gy, mode='valid')
	else:
		gradient_x = scipy.signal.convolve2d(im, gx, mode='same', boundary='wrap')
		gradient_y = scipy.signal.convolve2d(im, gy, mode='same', boundary='symm')

	# Y gradient is easy since we're using equirectangular projection so y step size is constant
	dy = EARTH_POLAR_CIRCUMFERENCE_M / (2 * height)
	gradient_y /= (kernel_magnitude * dy)

	if not latitude_adjust:
		dx = dy
		gradient_x /= (kernel_magnitude * dx)
	else:
		latitude_radians, latitude_step = np.linspace(start=-PI/2, stop=PI/2, num=height, endpoint=False, retstep=True)
		latitude_radians += 0.5*latitude_step
		circumference_at_latitude = EARTH_EQUATORIAL_CIRCUMFERENCE_M * np.cos(latitude_radians)
		assert np.amin(circumference_at_latitude) > 0
		circumference_at_latitude_clipped = np.maximum(circumference_at_latitude, MIN_LATITUDE_ADJUST_CIRCUMFERENCE)

		dx = circumference_at_latitude_clipped / width

		dx = np.tile(dx.reshape((height, 1)), (1, width))
		assert dx.shape == gradient_x.shape == (height, width), f"{gradient_x.shape=}, {dx.shape=}"

		gradient_x /= (kernel_magnitude * dx)

	assert gradient_x.shape == gradient_y.shape == (height, width)

	if magnitude:
		gradient_mag = np.sqrt(np.square(gradient_x), np.square(gradient_y))
		if scale01:
			gradient_mag = rescale(gradient_mag, data_range(gradient_mag), (0., 1.))
		return gradient_mag
	else:
		if scale01:
			max_abs_gradient = max(np.amax(gradient_x), np.amax(gradient_y))
			gradient_x = rescale(gradient_x, (-max_abs_gradient, max_abs_gradient), (-1., 1.))
			gradient_y = rescale(gradient_y, (-max_abs_gradient, max_abs_gradient), (-1., 1.))
		return gradient_x, gradient_y


def average_color(values: np.ndarray, /, *, median=False, luv_space=True) -> np.ndarray:

	if len(values) == 0:
		return np.array([np.nan, np.nan, np.nan])

	# TODO: possibly try to find "dominant color" instead of averaging? (mode? most common color with saturation >= average?)

	if luv_space:

		luv = colour.XYZ_to_Luv(colour.sRGB_to_XYZ(values))

		f_average = np.median if median else np.mean

		l_avg = f_average(luv[..., 0])
		u_avg = f_average(luv[..., 1])
		v_avg = f_average(luv[..., 2])

		luv_avg = np.array([l_avg, u_avg, v_avg])

		return colour.XYZ_to_sRGB(colour.Luv_to_XYZ(luv_avg))

	else:

		# Average in HSL space, using circular mean for hue:
		# https://en.wikipedia.org/wiki/Mean_of_circular_quantities
		# http://mkweb.bcgsc.ca/color-summarizer/?faq

		if median:
			raise NotImplementedError('median not currently supported with luv_space=False')

		sum_luma = 0.0
		sum_sat = 0.0
		num_sine_cos = 0
		sum_sine = 0.0
		sum_cos = 0.0
		num_points = len(values)

		r_vect = values[..., 0][:]
		g_vect = values[..., 1][:]
		b_vect = values[..., 2][:]

		for r, g, b in zip(r_vect, g_vect, b_vect):
				
			# HLS vs HSV:
			# HSV(red): H=0, S=1, V=1 - cone
			# HLS(red): H=0, S=1, L=0.5 - double-cone (diamond)
			# These are mapped to a cylinder, i.e. rgb_to_hls(0,0,0.1) returns S=1

			h, l, s = colorsys.rgb_to_hls(r, g, b)

			h_rads = h * TWOPI

			sum_luma += l
			sum_sat += s

			if s > 0:
				# Don't average hue for totally desaturated colors
				# TODO: use weighted average
				num_sine_cos += 1
				sum_sine += np.sin(h_rads)
				sum_cos += np.cos(h_rads)

		if num_sine_cos:
			avg_sin = sum_sine / num_sine_cos
			avg_cos = sum_cos / num_sine_cos
			h_rads = np.arctan2(avg_sin, avg_cos)

			avg_h = h_rads / TWOPI
			avg_h %= 1.0
		else:
			avg_h = None

		avg_l = sum_luma / num_points
		avg_s = sum_sat / num_points

		if avg_h is None and avg_s != 0:
			print('WARN: no avg hue, but avg saturation nonzero (%f)' % avg_s)

		return np.array(colorsys.hls_to_rgb(avg_h if avg_h else 0.0, avg_l, avg_s))


def _handle_bounds(arr: np.ndarray, dim_size: int, mode: Literal['clip', 'wrap', 'fold']) -> np.ndarray:
	if mode == 'clip':
		arr = np.clip(arr, 0, dim_size - 1)

	elif mode == 'wrap':
		arr = np.mod(arr, dim_size)
		arr[arr < 0] += dim_size

	elif mode == 'fold':
		arr = np.mod(arr, 2*dim_size)
		arr[arr >= dim_size] -= 2*dim_size
		arr[arr <= -dim_size] += 2*dim_size
		arr = np.abs(arr)

	else:
		raise TypeError(f'Invalid boundary mode: {mode}')

	assert (arr >= 0).all()
	assert (arr < dim_size).all()
	return arr


def remap(
		im: np.ndarray,
		xmap: np.ndarray,
		ymap: np.ndarray,
		bilinear=False,
		x_bounds: Literal[None, 'clip', 'wrap', 'fold'] = None,
		y_bounds: Literal[None, 'clip', 'wrap', 'fold'] = None,
		nan=None,
		) -> np.ndarray:
	"""
	Equivalent to cv2.remap; only supports bilinear & nearest-neighbor interpolation
	"""

	if nan is None:
		nan_map = None
	else:
		nan_map = np.logical_or(np.isnan(xmap), np.isnan(ymap))
		xmap = np.nan_to_num(xmap, nan=0)
		ymap = np.nan_to_num(ymap, nan=0)

	if not bilinear:
		# Nearest neighbor

		xmap = np.round(xmap).astype(int)
		ymap = np.round(ymap).astype(int)

		if x_bounds is not None:
			xmap = _handle_bounds(xmap, dim_size=im.shape[1], mode=x_bounds)

		if y_bounds is not None:
			ymap = _handle_bounds(ymap, dim_size=im.shape[0], mode=y_bounds)

		ret = im[ymap, xmap, :]

	else:
		# Bilinear

		xmap_floor = np.floor(xmap)
		ymap_floor = np.floor(ymap)

		x = xmap - xmap_floor
		y = ymap - ymap_floor

		xmap0 = xmap_floor.astype(int)
		ymap0 = ymap_floor.astype(int)
		xmap1 = xmap0 + 1
		ymap1 = ymap0 + 1
		del xmap_floor, ymap_floor

		if x_bounds is not None:
			xmap0 = _handle_bounds(xmap0, dim_size=im.shape[1], mode=x_bounds)
			xmap1 = _handle_bounds(xmap1, dim_size=im.shape[1], mode=x_bounds)

		if y_bounds is not None:
			ymap0 = _handle_bounds(ymap0, dim_size=im.shape[0], mode=y_bounds)
			ymap1 = _handle_bounds(ymap1, dim_size=im.shape[0], mode=y_bounds)

		# TODO: Technically, should degamma, interpolate in linear domain, and regamma
		# Unlikely to make much of a noticeable difference though

		# TODO: Could optimize this further with in-place operations
		# (need to deal with differing datatypes though)

		# X interpolation

		nx = 1.0 - x

		im_x0_y0 = im[ymap0, xmap0, :]
		im_x1_y0 = im[ymap0, xmap1, :]
		im_y0 = nx[..., None] * im_x0_y0 + x[..., None] * im_x1_y0
		del im_x0_y0, im_x1_y0

		im_x0_y1 = im[ymap1, xmap0, :]
		im_x1_y1 = im[ymap1, xmap1, :]
		im_y1 = nx[..., None] * im_x0_y1 + x[..., None] * im_x1_y1
		del im_x0_y1, im_x1_y1
		del x, nx

		# Y interpolation

		ny = 1.0 - y

		ret = ny[..., None] * im_y0 + y[..., None] * im_y1
		del im_y0, im_y1
		del y, ny

		# Convert back to original dtype

		if np.issubdtype(im.dtype, np.integer):
			ret = np.round(ret)

		ret = ret.astype(im.dtype)

	if nan_map is not None:
		ret[nan_map, :] = nan

	return ret
