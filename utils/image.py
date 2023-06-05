#!/usr/bin/env python

import colorsys
from contextlib import contextmanager
from math import ceil, floor, isclose, log2
from numbers import Number
from typing import Final, Optional, Tuple, Union, Literal
import warnings

import colour
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from PIL import Image
import scipy.ndimage

import utils.numeric
from utils.numeric import rescale, data_range, max_abs, magnitude, linspace_midpoint, require_same_shape

from utils.consts import PI, TWOPI, EARTH_POLAR_CIRCUMFERENCE_M, EARTH_POLAR_CIRCUMFERENCE_KM, EARTH_EQUATORIAL_CIRCUMFERENCE_M


SHOW_IMG_USE_MATPLOTLIB: Final = False

GRADIENT_CORRECT_WRAPPING: Final = True


@contextmanager
def disable_pil_max_pixels(do_it: bool = True, /):
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


def nan_to_alpha(
		arr: np.ndarray,
		*,
		nan_val: Optional[Number] = 0,
		alpha_non_nan_val: Optional[Number] = None,
		nan_mask: Optional[np.ndarray] = None,
		) -> np.ndarray:
	"""
	Add an alpha channel, based on NaN values in array

	:param arr: Array to convert. Currently must have 1 or 3 channels.
	:param nan_val: Value to be used to fill NaN values in channels besides alpha.
		If None, will leave NaN values in place.
	:param alpha_non_nan_val: Value to be used for alpha channel for non-NaN values.
		Defaults to 1.0 if arr.dtype is floating, or max of type otherwise.
	:param nan_mask: Mask of NaN values in array, if already calculated.
		Must be provided if arr.dtype is not floating.
		Currently must be provided if arr has more than 1 channel.
	"""

	arr_is_float = np.issubdtype(arr.dtype, np.floating)
	nan_mask_was_provided = (nan_mask is not None)

	if (not arr_is_float) and (not nan_mask_was_provided):
		raise ValueError('Must provide nan_mask if using with non-float types')

	if nan_mask is None:
		nan_mask = np.isnan(arr)

	if len(nan_mask.shape) == 3:
		if nan_mask_was_provided:
			raise NotImplementedError('Multi-channel nan_mask not currently supported')
		else:
			raise NotImplementedError('Multi-channel inputs are only supported if single-channel nan_mask is provided')

	if nan_val is not None:
		arr = np.nan_to_num(arr, nan=nan_val)

	if len(arr.shape) == 2:
		arr = arr.reshape((arr.shape[0], arr.shape[1], 1))
		assert len(arr.shape) == 3, f"{arr.shape=}"
	elif len(arr.shape) != 3:
		raise ValueError(f'arr must have 2 or 3 dimensions (shape={arr.shape})')

	num_channels_in = arr.shape[2]
	num_channels_out = num_channels_in + 1

	if alpha_non_nan_val is None:
		alpha_non_nan_val = 1.0 if arr_is_float else np.iinfo(arr.dtype).max

	alpha_channel = np.full_like(arr, alpha_non_nan_val)
	alpha_channel[nan_mask] = 0
	assert len(arr.shape) == 3 and arr.shape[2] == num_channels_in, f"{arr.shape=}"
	arr = np.concatenate((arr, alpha_channel), axis=2)
	assert len(arr.shape) == 3 and arr.shape[2] == num_channels_out, f"{arr.shape=}"

	return arr


def alpha_to_nan(arr: np.ndarray, *, nan_thresh: Number = 0) -> np.ndarray:

	if (len(arr.shape) != 3) or (arr.shape[2] == 1):
		raise ValueError(f'arr does not have alpha channel (shape={arr.shape})')

	num_channels_in = arr.shape[-1]
	num_channels_out = num_channels_in - 1

	alpha_channel = arr[..., -1]
	nan_mask = alpha_channel <= nan_thresh

	arr = arr[..., :-1]
	assert arr.shape[2] == num_channels_out

	for channel in range(num_channels_out):
		arr[..., channel][nan_mask] = np.nan

	if num_channels_out == 1:
		arr = arr.reshape((arr.shape[0], arr.shape[1]))

	return arr


def resize_array(
		arr: np.ndarray,
		new_size: Tuple[int, int],  # TODO: change these to separate width & height arguments, so keywords should prvent getting dimensions swapped around
		data_range: Optional[Tuple[float, float]] = None,
		resampling = Image.BILINEAR,
		verbose = False,
		) -> np.ndarray:
	"""
	:param new_size: in image dimensions, i.e. (width, height)
	:param data_range: deprecated
	"""

	if len(arr.shape) not in [2, 3]:
		raise ValueError(f'Array must have 2 or 3 dimensions (shape={arr.shape})')

	input_dtype = arr.dtype

	nan_mask = np.isnan(arr)
	any_nan = nan_mask.any()

	mode = None
	if any_nan:
		arr = nan_to_alpha(arr, nan_val=0.0, nan_mask=nan_mask)
		assert len(arr.shape) == 3

		if verbose:
			for channel in range(arr.shape[2]):
				print(f'Channel {channel}, range: {utils.numeric.data_range(arr[:, :, channel])}')

		if arr.shape[2] == 2:
			mode = 'LA'
		elif arr.shape[2] == 4:
			mode = 'RGBA'
		else:
			raise NotImplementedError(f'Cannot handle array with {arr.shape[-1] - 1} channels with NaN values (shape={arr.shape})')

	new_size = list(new_size)

	im = Image.fromarray(arr, mode=mode)
	im = im.resize(tuple(new_size), resample=resampling)
	arr = np.asarray(im, dtype=input_dtype)

	if any_nan:
		if verbose:
			print(f'Image shape: {arr.shape}')
			for channel in range(arr.shape[-1]):
				print(f'Channel {channel}, range: {utils.numeric.data_range(arr[..., channel])}')

		arr = alpha_to_nan(arr)

	if len(arr.shape) == 3 and arr.shape[2] == 1:
		arr = arr.reshape((arr.shape[0], arr.shape[1]))

	return arr


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

	num_dim = len(im.shape)

	if num_dim == 2:
		im = im[..., np.newaxis]
	elif num_dim != 3:
		raise ValueError(f'Invalid image shape for remap: {im.shape}')

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

	if num_dim == 2:
		assert len(ret.shape) == 3 and ret.shape[2] == 1
		ret = ret.reshape((ret.shape[0], ret.shape[1]))

	return ret


def matplotlib_figure_canvas_to_image(figure: Figure, canvas: FigureCanvas):

	# https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
	canvas.draw()
	return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(figure.canvas.get_width_height()[::-1] + (3,))


def _pad_sphere_image_for_convolution(im: np.ndarray, /, large_sobel: bool) -> np.ndarray:
	height, width = im.shape

	first_col = im[:, :1]
	last_col = im[:, -1:]
	if large_sobel:
		second_col = im[:, 1:2]
		second_last_col = im[:, -2:-1]
		im = np.concatenate((second_last_col, last_col, im, first_col, second_col), axis=1)
		assert im.shape == (height, width + 4)
	else:
		im = np.concatenate((last_col, im, first_col), axis=1)
		assert im.shape == (height, width + 2)

	first_row = np.fliplr(im[:1, :])
	last_row = np.fliplr(im[-1:, :])
	if large_sobel:
		second_row = np.fliplr(im[1:2, :])
		second_last_row = np.fliplr(im[-2:-1, :])
		im = np.concatenate((second_row, first_row, im, last_row, second_last_row), axis=0)
		assert im.shape == (height + 4, width + 4)
	else:
		im = np.concatenate((first_row, im, last_row), axis=0)
		assert im.shape == (height + 2, width + 2)
	
	return im


def gaussian_blur(im: np.ndarray, sigma: float, truncate=4.0, mode='nearest', min_sigma_px: float = 0.0, **kwargs):
	if min_sigma_px and sigma < min_sigma_px:
		return im
	return scipy.ndimage.gaussian_filter(im, sigma=sigma, truncate=truncate, mode=mode, **kwargs)


def sphere_gaussian_blur(
		im: np.ndarray,
		sigma: float,
		min_sigma_px: float = 0.5,
		truncate=4.0,
		latitude_sections=6,
		):

	height = im.shape[0]
	width = im.shape[1]

	# Blur X
	# Vary sigma with latitude

	im_blur = im.copy()
	latitude_boundaries = np.linspace(-90, 90, latitude_sections + 1, endpoint=True)

	latitude = linspace_midpoint(90, -90, height)

	for i in range(latitude_sections):
		latitude_start = latitude_boundaries[i]
		latitude_end = latitude_boundaries[i + 1]

		# Technically not correct - want average sigma_scale, not average latitude
		# Should be a close enough approximation except maybe for center secion (especially if odd latitude_sections)
		avg_latitude = 0.5 * (latitude_start + latitude_end)
		sigma_scale = np.cos(np.radians(avg_latitude))
		latitude_x_sigma = sigma / sigma_scale

		if not (min_sigma_px and latitude_x_sigma < min_sigma_px):
			latitude_mask = np.logical_and(
				latitude >= latitude_start if i == 0 else latitude > latitude_start,
				latitude <= latitude_end)

			im_blur[latitude_mask, ...] = scipy.ndimage.gaussian_filter1d(
				im[latitude_mask, ...], sigma=latitude_x_sigma, truncate=truncate, axis=1, mode='wrap')

	# Blur Y

	"""
	Do Y after X
	Normally Gaussian blur is totally separable so order shouldn't matter, but in this case the X blur can result in
	artifacts at section boundaries, which the Y blur will hide
	"""

	if not (min_sigma_px and sigma < min_sigma_px):
		im_rot180 = np.flip(im_blur, axis=None)

		if True:
			im_blur = np.concatenate((im_blur, im_rot180), axis=0)
			im_blur = scipy.ndimage.gaussian_filter1d(im_blur, sigma=sigma, truncate=truncate, axis=0, mode='wrap')
			im_blur = im_blur[:height, ...]
		else:
			# TODO: likely faster, but need to test to make sure it works properly!
			needed_rows = min(height, ceil(sigma * truncate))
			im_blur = np.concatenate((im_rot180[-needed_rows:, ...], im_blur, im_rot180[:needed_rows, ...]), axis=0)
			im_blur = scipy.ndimage.gaussian_filter1d(im_blur, sigma=sigma, truncate=truncate, axis=0, mode='nearest')
			im_blur = im_blur[needed_rows : needed_rows + height, ...]

	assert im_blur.shape == im.shape

	return im_blur


def gradient(
		im: np.ndarray, /, *,
		magnitude: bool = False,
		d_step: Optional[float] = None,
		auto_scale: bool = False,
		sobel: bool = True,
		large_sobel: bool = False,
		) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
	"""
	Calculate gradient

	:param im: data to calculate gradient
	:param magnitude: return magnitude; otherwise, return (x, y)
	:param d_step: dx & dy step size; ignored if auto_scale; default is 1/height
	:param auto_scale: scale resulting data to [0, 1] if magnitude, or [-1, 1] if not
	:param sobel: use 3x3 Sobel filters
	:param large_sobel: use 5x5 Sobel filters

	:returns: gradient magnitude if magnitude=True, otherwise (gx, gy)
	"""

	height, width = im.shape

	if d_step is None:
		d_step = 1.0 / height

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
		# TODO: can use np.gradient() in this case instead of needing convolve2d(), likely faster
		gx = np.array([[1, 0, -1]])

	kernel_magnitude = np.sum(np.abs(gx))
	gx = gx / (d_step * kernel_magnitude)

	# Negate Y axis so gradient will be relative to Cartesian coordinates (not screen)
	gy = -np.transpose(gx)

	gradient_x = scipy.signal.convolve2d(im, gx, mode='same', boundary='symm')
	gradient_y = scipy.signal.convolve2d(im, gy, mode='same', boundary='symm')

	assert gradient_x.shape == gradient_y.shape == (height, width)

	if magnitude:
		gradient_mag = magnitude(gradient_x, gradient_y)
		if auto_scale:
			gradient_mag = rescale(gradient_mag)
		return gradient_mag
	else:
		if auto_scale:
			max_abs_gradient = max(np.amax(gradient_x), np.amax(gradient_y))
			gradient_x = rescale(gradient_x, (-max_abs_gradient, max_abs_gradient), (-1., 1.))
			gradient_y = rescale(gradient_y, (-max_abs_gradient, max_abs_gradient), (-1., 1.))
		return gradient_x, gradient_y


def sphere_gradient(
		im: np.ndarray, /, *,
		magnitude: bool = False,

		d_step: Optional[float] = None,
		auto_scale: bool = False,
		scale_earth: bool = False,

		latitude_adjust: bool = True,
		min_latitude_adjust_circumference = 0.1,

		sobel: bool = True,
		large_sobel: bool = False,
		) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
	"""
	Calculate gradient of equirectangular projection data

	:param im: data to calculate gradient
	:param magnitude: return magnitude; otherwise, return (x, y)

	:param d_step: dx & dy step size; ignored if auto_scale; if scale_earth, will set size relative to Earth; default (if not scale_earth) is 1/height
	:param auto_scale: scale resulting data to [0, 1] if magnitude, or [-1, 1] if not
	:param scale_earth: scale relative to Earth circumference (in meters)

	:param latitude_adjust: adjust X gradient with latitude to correct for equirectangular distortion
	:param min_latitude_adjust_circumference: Do not adjust latitude when circumference is less than this (relative to total)

	:param sobel: use 3x3 Sobel filters
	:param large_sobel: use 5x5 Sobel filters

	:returns: gradient magnitude if magnitude=True, otherwise (gx, gy)
	"""

	height, width = im.shape
	if width != 2 * height:
		raise ValueError(f'sphere_gradient() image must have aspect ratio 2 ({im.shape=})')

	if auto_scale:
		# Doesn't matter, scale will be adjusted later anyway
		circumference_x = circumference_y = 1.0
	elif scale_earth:
		if d_step is None:
			d_step = 1.0
		circumference_y = d_step * EARTH_POLAR_CIRCUMFERENCE_M
		circumference_x = d_step * EARTH_EQUATORIAL_CIRCUMFERENCE_M
	else:
		if d_step is None:
			d_step = 1.0 / height
		circumference_x = circumference_y = d_step * TWOPI

	min_latitude_adjust_circumference *= circumference_y

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
		# TODO: can use np.gradient() in this case instead of needing convolve2d(), likely faster
		gx = np.array([[1, 0, -1]])

	# Negate Y axis so gradient will be relative to Cartesian coordinates (not screen)
	gy = -np.transpose(gx)

	kernel_magnitude = np.sum(np.abs(gx))

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
	dy = circumference_y / (2 * height)
	gradient_y /= (kernel_magnitude * dy)

	if not latitude_adjust:
		dx = dy
	else:
		latitude_radians = linspace_midpoint(start=-PI/2, stop=PI/2, num=height)
		circumference_at_latitude = circumference_x * np.cos(latitude_radians)
		assert np.amin(circumference_at_latitude) > 0

		if min_latitude_adjust_circumference:
			circumference_at_latitude_clipped = np.maximum(circumference_at_latitude, min_latitude_adjust_circumference)

		# TODO: could be slightly more efficient here by calculating all at once (1 less division)

		dx = circumference_at_latitude_clipped / width

		# TODO: broadcast instead of tile
		dx = np.tile(dx.reshape((height, 1)), (1, width))
		assert dx.shape == gradient_x.shape == (height, width), f"{gradient_x.shape=}, {dx.shape=}"

	gradient_x /= (kernel_magnitude * dx)

	assert gradient_x.shape == gradient_y.shape == (height, width)

	if magnitude:
		gradient_mag = magnitude(gradient_x, gradient_y)
		if auto_scale:
			gradient_mag = rescale(gradient_mag, data_range(gradient_mag), (0., 1.))
		return gradient_mag
	else:
		if auto_scale:
			max_abs_gradient = max(np.amax(gradient_x), np.amax(gradient_y))
			gradient_x = rescale(gradient_x, (-max_abs_gradient, max_abs_gradient), (-1., 1.))
			gradient_y = rescale(gradient_y, (-max_abs_gradient, max_abs_gradient), (-1., 1.))
		return gradient_x, gradient_y


def gaussian_blur_map(
		map_im: np.ndarray,
		/,
		sigma_km: float,
		*,
		flat_map: bool,
		latitude_span: float = 180,
		min_sigma_px: float = 0.5,
		truncate = 4.0,
		resize: Literal[False, 'internal', True] = 'internal',
		resize_target_sigma_px = 8.0,
		) -> np.ndarray:
	"""
	Gaussian blur map at Earth scale

	:param map_im: Map image to blur
	:param sigma_km: Gaussian blur size
	:param flat_map: if True, map is treated as flat 2D; if False, map is equirectangular projection of full sphere
	:param latitude_span: Latitude range covered by map. Must be 180 if not flat_map
	:param min_sigma_px: If sigma is fewer than this many pixels at any particular latitude, skip blur
	:param truncate: How many sigma to truncate kernel
	:param resize: Allow resizing when sigma is large.
		False: Always process blur at full resolution.
		'internal': Downscale, blur, and upscale back to original resolution. Can be much faster, but slightly lower quality
		True: Downscale & blur, and return lower-resolution image
	:param resize_target_sigma_px: Target sigma (in pixels) for internal resize. Lower = faster, but lower quality
	"""

	resize_before = bool(resize)
	resize_after = (isinstance(resize, str) and resize.lower() == 'internal')

	if isclose(latitude_span, 180):
		latitude_span = 180

	if (not flat_map) and latitude_span != 180:
		raise ValueError(f'latitude_span must be 180 for non-flat map ({latitude_span=})')

	height, width = map_im.shape[0], map_im.shape[1]

	sigma = height * (sigma_km / EARTH_POLAR_CIRCUMFERENCE_KM) * (360 / latitude_span)

	scale = 1
	internal_size = (width, height)
	if resize_before:
		max_scale = min(width, height)
		# TODO: floor to nearest divisor of image width & height, not necessarily power of 2
		scale = 2 ** floor(log2(sigma / resize_target_sigma_px))
		scale = min(scale, max_scale)

		if scale <= 2:
			# Don't bother - it's probably not worth the extra computation to perform the resize
			scale = 1

	internal_size = (width // scale, height // scale)

	if scale > 1:
		sigma /= scale
		blur_input = resize_array(map_im, new_size=internal_size)
	else:
		blur_input = map_im

	# TODO optimization: resize X per sphere_gaussian_blur latitude section instead of entire image
	blur_func = gaussian_blur if flat_map else sphere_gaussian_blur
	blurred = blur_func(blur_input, sigma=sigma, truncate=truncate, min_sigma_px=min_sigma_px)

	if resize_after and scale > 1:
		out = resize_array(blurred, new_size=(width, height))
		assert out.shape == map_im.shape
	else:
		out = blurred

	return out


def map_gradient(
		map_im: np.ndarray,
		/,
		flat_map: bool,
		*,
		magnitude: bool = False,
		latitude_span: float = 180.0,
		sigma_km: float = 0.0,
		min_sigma_px: float = 0.5,
		truncate = 4.0,
		resize: Literal[False, 'internal', True] = 'internal',
		resize_target_sigma_px = 8.0,
		sobel = False,
		large_sobel = False,
		**gradient_kwargs,
		) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
	"""
	Calculate gradient of map at Earth scale

	:param map_im: Map image to calculate gradient. Data is assumed to be in meters.
	:param flat_map: if True, map is treated as flat 2D; if False, map is equirectangular projection of full sphere
	:param magnitude: if True, return magnitude; otherwise return (x, y)
	:param latitude_span: Latitude range covered by map. Must be 180 if not flat_map
	:param sigma_km: Gaussian blur size
	:param min_sigma_px: If sigma is fewer than this many pixels at any particular latitude, skip blur
	:param truncate: How many sigma to truncate Gaussian blur kernel
	:param resize: Allow resizing when sigma is large.
		False: Always process blur at full resolution.
		'internal': Downscale, blur, and upscale back to original resolution. Can be much faster, but slightly lower quality
		True: Downscale & blur, and return lower-resolution image
	:param resize_target_sigma_px: Target sigma (in pixels) for internal resize. Lower = faster, but lower quality
	:param sobel: Use 3x3 Sobel operators for gradient
	:param large_sobel: Use 5x5 Sobel operators for gradient
	"""

	if isclose(latitude_span, 180):
		latitude_span = 180

	if (not flat_map) and latitude_span != 180:
		raise ValueError(f'latitude_span must be 180 for non-flat map ({latitude_span=})')

	if sigma_km > 0:
		# TODO: with allow_internal_resize, can leave at low resolution for calculating gradient
		map_im_blurred = gaussian_blur_map(
			map_im,
			sigma_km=sigma_km,
			flat_map=flat_map,
			latitude_span=latitude_span,
			min_sigma_px=min_sigma_px,
			truncate=truncate,
			resize=bool(resize),
			resize_target_sigma_px=resize_target_sigma_px,
		)
	else:
		map_im_blurred = map_im

	gradient_kwargs['magnitude'] = magnitude
	gradient_kwargs['sobel'] = sobel
	gradient_kwargs['large_sobel'] = large_sobel

	if flat_map:
		d_step_m = (latitude_span * EARTH_POLAR_CIRCUMFERENCE_M) / (360 * map_im_blurred.shape[0])
		ret = gradient(map_im_blurred, d_step=d_step_m, **gradient_kwargs)
	else:
		ret = sphere_gradient(map_im_blurred, scale_earth=True, **gradient_kwargs)

	if resize in [False, 'internal'] and map_im_blurred.shape != map_im.shape:
		if isinstance(ret, np.ndarray):
			ret = resize_array(ret, (map_im.shape[1], map_im.shape[0]))
		else:
			assert isinstance(ret, tuple)
			ret = tuple([resize_array(r, (map_im.shape[1], map_im.shape[0])) for r in ret])

	return ret


def divergence(*, x: np.ndarray, y: np.ndarray, d_step: Optional[float]=None):

	if d_step is None:
		d_step = 1.0 / x.shape[0]

	return (np.gradient(y, axis=0) + np.gradient(x, axis=1)) / d_step


def sphere_divergence(*, x: np.ndarray, y: np.ndarray, d_step: Optional[float]=None, latitude_adjust=True):

	require_same_shape(x, y)

	if d_step is None:
		d_step = 1.0 / x.shape[0]

	x_wrapped = np.concatenate((x[:, -1:], x, x[:, :1]), axis=1)
	assert x_wrapped.shape == (x.shape[0], x.shape[1] + 2)
	gradient_x = np.gradient(x_wrapped, axis=1)
	gradient_x = gradient_x[:, 1:-1]
	assert gradient_x.shape == x.shape

	gradient_y = np.gradient(y, axis=0)

	if latitude_adjust:
		height, width = x.shape

		latitude_radians = linspace_midpoint(start=-PI/2, stop=PI/2, num=height)
		dx = np.cos(latitude_radians)

		assert np.amin(dx) > 0

		# TODO: broadcast instead of tile
		dx = np.tile(dx.reshape((height, 1)), (1, width))
		assert dx.shape == gradient_x.shape == (height, width), f"{gradient_x.shape=}, {dx.shape=}"

		gradient_x /= dx

	return (gradient_x + gradient_y) / d_step
