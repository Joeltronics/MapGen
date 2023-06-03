#!/usr/bin/env python

import sys
import warnings

if sys.version_info < (3, 10):
	warnings.warn('This was written for Python >= 3.10; it may work in lower versions but is untested!')

import argparse
import importlib


def main():

	parser = argparse.ArgumentParser()
	parser.set_defaults(module='gradio_ui')
	subparsers = parser.add_subparsers(required=False)

	common_kwarg = dict(add_help=False)

	subparsers.add_parser('run', help="Run Gradio UI", **common_kwarg)
	subparsers.add_parser('data', help="Download & show NASA data", **common_kwarg).set_defaults(module='data')
	subparsers.add_parser('process', help="Process NASA data", **common_kwarg).set_defaults(module='process_data')
	subparsers.add_parser('wind', help="Test wind simulation", **common_kwarg).set_defaults(module='generation.winds')
	subparsers.add_parser('rain', help="Test precipitation simulation", **common_kwarg).set_defaults(module='generation.precipitation')

	args, remaining_args = parser.parse_known_args()

	module = importlib.import_module(args.module)
	module.main(remaining_args)


if __name__ == "__main__":
	main()
