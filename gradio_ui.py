#!/usr/bin/env python

import sys
import warnings

if sys.version_info < (3, 10):
	warnings.warn('This was written for Python >= 3.10; it may work in lower versions but is untested!')

import argparse
from copy import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, unique
from typing import List, Optional, Tuple

import gradio as gr
from matplotlib import pyplot as plt
import numpy as np

from generation.fbm import fbm, diff_fbm, diff_fbm, valley_fbm, ridge_fbm, sphere_fbm, wrapped_fbm, domain_warped_fbm, sphere_domain_warped_fbm
from generation.map_generation import GeneratorParams, GeneratorType, TopographyParams, TemperatureParams, ErosionParams, Planet, generate

from utils.image import float_to_uint8
from utils.numeric import rescale
from utils.utils import tprint


INITIAL_SEED = 0


NOISE_TEST_CMAP_NAMES = [
	'inferno', 'bwr', 'gist_earth', 'prism'
]
NOISE_TEST_CMAPS = [plt.get_cmap(name) for name in NOISE_TEST_CMAP_NAMES]


def gradio_callback_map_generator(
		generator: str,
		seed: str,
		resolution: int,
		use_earth: bool,
		water_amount: float,
		continent_size: float,
		elevation_steps: int,
		pole_temp_C: float,
		equator_temp_C: float,
		erosion_amount: float,
		erosion_cell_size: float,
		):

	generator = GeneratorType[generator]

	assert isinstance(seed, int)

	if isinstance(resolution, float):
		resolution = int(round(resolution))

	if isinstance(elevation_steps, float):
		elevation_steps = int(round(elevation_steps))

	water_amount /= 100.0
	continent_size /=  100.0

	erosion_amount /= 100.0
	erosion_cell_size /= 100.0

	params = GeneratorParams(
		generator=generator,
		seed=seed,
		topography=TopographyParams(
			use_earth=use_earth,
			elevation_steps=elevation_steps,
			water_amount=water_amount,
			continent_size=continent_size,
		),
		temperature=TemperatureParams(
			pole_C=pole_temp_C,
			equator_C=equator_temp_C,
		),
		erosion=ErosionParams(
			amount=erosion_amount,
			cell_size=erosion_cell_size,
		),
		noise_strength=(0.0 if use_earth else 1.0),  # TODO: make this a slider?
	)

	print_str = str(params)
	print(f'Generating {generator.name}...')
	planet = generate(params, resolution=resolution)
	print('Done')


	polar_azimuthal = None
	if planet.polar_azimuthal is not None:
		polar_azimuthal = np.concatenate(planet.polar_azimuthal, axis=1)

	# TODO: save these to files, and show filepath
	# (likely faster than Gradio default of embedding Base64-encoded, at least for large images)

	return (
		print_str,
		[planet.equirectangular],
		planet.views,
		[polar_azimuthal],
		[planet.biomes_img],
		[planet.elevation_img, planet.gradient_img_bw, planet.gradient_img_color, planet.erosion_img],
		[planet.temperature_img] + [planet.rainfall_img] + planet.prevailing_wind_imgs,
		[planet.graph_figure],
	)


def make_planet_generator_tab():

	inputs = []
	outputs = []

	with gr.Row():
		with gr.Column():
			generate_button = gr.Button("Generate")

			with gr.Box():
				inputs += [
					gr.Dropdown(choices=[generator.name for generator in GeneratorType], value=GeneratorType.planet_3d.name, label='Generator'),
					gr.Number(precision=0, value=INITIAL_SEED, label='Seed'),
					gr.Slider(64, 3600, step=16, value=512, label='Resolution (width)'),
				]

			with gr.Accordion('Topography'):
				inputs += [
					gr.Checkbox(value=False, label='Use Earth topography'),
					gr.Slider(0, 100, step=5, value=70, label='Water amount'),
					gr.Slider(5, 200, step=5, value=25, label='Continent size scale'),
					gr.Slider(0, 8, step=1, value=0, label='Elevation difference steps'),
				]

			with gr.Accordion('Temperature'):
				inputs += [
					gr.Slider(-40, 80, step=1, value=-5, label='Average pole temperature (C)'),
					gr.Slider(-40, 80, step=1, value=30, label='Average equator temperature (C)'),
				]

			with gr.Accordion('Erosion'):
				inputs += [
					gr.Slider(0, 100, step=5, value=75, label='Amount'),
					gr.Slider(0, 25, step=0.25, value=1, label='Cell size'),
				]

		with gr.Column():
			gr.Label('Outputs')
			outputs += [
				gr.Textbox(),
				gr.Gallery(label='Equirectangular'),
				gr.Gallery(label='Planet Views'),
				gr.Gallery(label='Polar Azimuthal'),
				gr.Gallery(label='Biomes'),
				gr.Gallery(label='Elevation/Gradient/Erosion'),
				gr.Gallery(label='Temperature/Precipitation/Wind'),
				gr.Gallery(label='Graphs'),
			]

	generate_button.click(fn=gradio_callback_map_generator, inputs=inputs, outputs=outputs)


@unique
class FbmType(Enum):
	fbm = 'Basic FBM'
	fbm_wrap_h = 'Wrapped FBM (horizontal)'
	fbm_wrap_v = 'Wrapped FBM (vertical)'
	fbm_wrap_both = 'Wrapped FBM (both)'
	fbm_sphere = 'Sphere FBM'
	diff = 'Diff'
	valley = 'Valley'
	ridge = 'Ridge'
	domain_warp = 'Domain Warping'
	domain_warp_sphere = 'Domain Warped Sphere'


def gradio_callback_noise_test(
		fbm_type: str,
		seed: str,
		use_fns: bool,
		width: int,
		height: int,
		octaves: float,
		steps: int,
		amount: float,
		gain: float,
		lacunarity: float,
		base_frequency: float,
		normalize: bool,
		):

	fbm_type = FbmType[fbm_type]

	assert isinstance(seed, int)

	if isinstance(width, float):
		width = int(round(width))

	if isinstance(height, float):
		height = int(round(height))

	if isinstance(octaves, float):
		octaves = int(round(octaves))

	params_str = f'fbm_type={fbm_type.name}, {seed=}, {use_fns=}, {width=}, {height=}, {octaves=}, {steps=}, {gain=}, {lacunarity=}, {base_frequency=}, {normalize=}'

	kwargs = dict(
		seed=seed,
		width=width,
		height=height,
		octaves=octaves,
		lacunarity=lacunarity,
		gain=gain,
		base_frequency=base_frequency,
		normalize=normalize,
		use_fns=use_fns,
	)

	bipolar = True

	start = tprint(f'Generating {fbm_type.name}...', is_start=True)

	if fbm_type == FbmType.fbm:
		img = fbm(**kwargs)

	elif fbm_type in [FbmType.fbm_wrap_h, FbmType.fbm_wrap_v, FbmType.fbm_wrap_both]:
		wrap_x = fbm_type in [FbmType.fbm_wrap_h, FbmType.fbm_wrap_both]
		wrap_y = fbm_type in [FbmType.fbm_wrap_v, FbmType.fbm_wrap_both]
		img = wrapped_fbm(wrap_x=wrap_x, wrap_y=wrap_y, **kwargs)

	elif fbm_type == FbmType.fbm_sphere:
		img = sphere_fbm(**kwargs)

	elif fbm_type == FbmType.diff:
		img = diff_fbm(diff_steps=steps, **kwargs)

	elif fbm_type == FbmType.domain_warp:
		# TODO: parameter for warp_amount
		# TODO: separate parameter for domain warp base freq vs final base freq
		img = domain_warped_fbm(warp_steps=steps, warp_amount=amount, **kwargs)

	elif fbm_type == FbmType.domain_warp_sphere:
		# TODO: parameter for warp_amount
		# TODO: separate parameter for domain warp base freq vs final base freq
		img = sphere_domain_warped_fbm(warp_steps=steps, warp_amount=amount, **kwargs)

	elif fbm_type == FbmType.valley:
		bipolar = False
		img = valley_fbm(**kwargs)

	elif fbm_type == FbmType.ridge:
		bipolar = False
		img = ridge_fbm(**kwargs)

	else:
		raise ValueError(f'Invalid {fbm_type=}')

	end = tprint('Done generating, converting noise to images')
	elapsed = end - start

	info_str = f'Generation time: {elapsed}\nData range: min={np.amin(img):.3f}, median={np.median(img):.3f}, mean={np.mean(img):.3f}, max={np.amax(img):.3f}'

	img_greyscale = float_to_uint8(img, bipolar=bipolar)

	img_tiled = np.tile(img_greyscale, [2, 2])

	if bipolar:
		img = 0.5*(img + 1.0)

	imgs_cmapped = [cmap(img) for cmap in NOISE_TEST_CMAPS]

	tprint('Done converting')

	return params_str, info_str, img_greyscale, img_tiled, *imgs_cmapped


def make_noise_test_tab():
	inputs = []
	outputs = []

	with gr.Row():
		with gr.Column():
			generate_button = gr.Button("Generate")

			with gr.Box():
				inputs += [
					gr.Dropdown(choices=[generator.name for generator in FbmType], value=FbmType.fbm.name, label='Type'),
					gr.Number(precision=0, value=INITIAL_SEED, label='Seed'),
					gr.Checkbox(value=True, label='Use FastNoiseSimd'),
					gr.Slider(32, 1024, step=32, value=512, label='Width'),
					gr.Slider(32, 1024, step=32, value=512, label='Height'),
				]

			# with gr.Box():
				# gr.Label('Parameters')
			with gr.Accordion('Parameters'):
				inputs += [
					gr.Slider(1, 12, step=1, value=10, label='Octaves'),
					gr.Slider(1, 12, step=1, value=1, label='Steps (certain generators only)'),
					gr.Slider(0, 2, step=0.05, value=1, label='Amount (certain generators only)'),
					gr.Slider(0.0, 2.0, step=0.05, value=0.5, label='Gain'),
					gr.Slider(1.0, 4.0, value=2.0, label='Lacunarity'),
					gr.Slider(0.25, 32.0, value=1.0, label='Frequency'),
					gr.Checkbox(value=True, label='Normalize'),
				]

		with gr.Column():
			# gr.Label('Outputs')
			outputs += [
				gr.Textbox(label='Parameters'),
				gr.Textbox(label='Info'),
				gr.Image(label='Result (greyscale)'),
				gr.Image(label='Result (tiled)'),
			]
			for cmap_name in NOISE_TEST_CMAP_NAMES:
				outputs.append(gr.Image(label=f'Result ({cmap_name})'))

	generate_button.click(fn=gradio_callback_noise_test, inputs=inputs, outputs=outputs)


def parse_args(args=None):
	parser = argparse.ArgumentParser()

	g = parser.add_argument_group('Gradio launch() options')
	mx = g.add_mutually_exclusive_group()
	mx.add_argument('--share-lan', dest='server_name', action='store_const', const='0.0.0.0', default=None, help='Share over LAN')
	mx.add_argument('--share-public', dest='share', action='store_true', help='Share publically (share=True)')
	g.add_argument('--show-error', action='store_true', help='show_error=True')
	g.add_argument('--gradio-debug', action='store_true', help='debug=True')

	return parser.parse_args(args)


def main(args=None):
	args = parse_args(args)

	print('Building UI...')
	with gr.Blocks(title='Map generator') as demo:
		with gr.Tab('Map generator'):
			make_planet_generator_tab()
		with gr.Tab('Noise test'):
			make_noise_test_tab()

	print('Launching...')
	demo.launch(
		server_name=args.server_name,
		share=args.share,
		debug=args.gradio_debug,
		show_error=args.show_error,
	)


if __name__ == "__main__":
	main()
