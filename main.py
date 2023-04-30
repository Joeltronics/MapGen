#!/usr/bin/env python

import sys
import warnings

if sys.version_info < (3, 10):
	warnings.warn('This was written for Python >= 3.10; it may work in lower versions but is untested!')

import argparse


def main():

	def _run_main(args):
		import gradio_ui
		gradio_ui.main(args)

	def _process_data_main(args):
		import process_data
		process_data.main(args)

	def _data_main(args):
		from data import data
		data.main(args)

	parser = argparse.ArgumentParser()

	subparsers = parser.add_subparsers(required=True)

	run_parser = subparsers.add_parser('run', add_help=False)
	run_parser.set_defaults(main=_run_main)

	process_parser = subparsers.add_parser('process', add_help=False)
	process_parser.set_defaults(main=_process_data_main)

	data_parser = subparsers.add_parser('data', add_help=False)
	data_parser.set_defaults(main=_data_main)

	args, remaining_args = parser.parse_known_args()

	args.main(remaining_args)


if __name__ == "__main__":
	main()
