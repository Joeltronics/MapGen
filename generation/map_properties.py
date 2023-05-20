#!/usr/bin/env python3

from functools import cached_property
from typing import Optional

import numpy as np

from .fbm import NoiseCoords

from utils.numeric import linspace_midpoint


class MapProperties:
	def __init__(
			self,
			flat: bool,
			height: float,
			width: Optional[float] = None,
			latitude_range: tuple[float, float] = (-90., 90.),
			longitude_range: Optional[tuple[float, float]] = None,
			noise_coord: Optional[NoiseCoords] = None,
			):

		
		if (latitude_range[0] == latitude_range[1]) or (min(latitude_range) < -90) or (max(latitude_range) > 90):
			raise ValueError(f'Invalid latitude range: {latitude_range}')

		self._latitude_range = sorted(latitude_range)

		self._flat = flat
		self._height = height
		self._width = width
		self._longitude_range = longitude_range
		self._noise_coord = noise_coord

		if self._width is None:
			if self._longitude_range is not None:
				self._width = round(height * self.longitude_span / self.latitude_span)
			elif flat:
				self._width = height
			else:
				self._width = 2 * height

		if self._longitude_range is None:
			half_longitude_span = 0.5 * self.latitude_span * self._width / self._height
			self._longitude_range = (-half_longitude_span, half_longitude_span)

		assert (self._width is not None) and (self._longitude_range is not None)

	# Simple getter properties

	@property
	def flat(self) -> bool:
		return self._flat

	@property
	def height(self) -> int:
		return self._height

	@property
	def width(self) -> int:
		return self._width

	@property
	def latitude_range(self) -> tuple[float, float]:
		return self._latitude_range

	@property
	def longitude_range(self) -> tuple[float, float]:
		return self._longitude_range

	@property
	def north(self) -> float:
		return self._latitude_range[0]

	@property
	def south(self) -> float:
		return self._latitude_range[1]

	@property
	def east(self) -> float:
		return self._longitude_range[0]

	@property
	def west(self) -> float:
		return self._longitude_range[1]

	@property
	def noise_coord(self) -> Optional[NoiseCoords]:
		return self._noise_coord

	# Non-cached calculated properties

	@property
	def latitude_span(self) -> float:
		return self._latitude_range[1] - self._latitude_range[0]

	@property
	def longitude_span(self) -> float:
		return self._longitude_range[1] - self._longitude_range[0]

	# Cached calculated properties

	@cached_property
	def latitude_1d(self) -> np.ndarray:
		return linspace_midpoint(start=self.north, stop=self.south, num=self.height) 

	@cached_property
	def latitude_1d_radians(self) -> np.ndarray:
		return np.radians(self.latitude_1d)

	@cached_property
	def latitude_column(self) -> np.ndarray:
		return self.latitude_1d[:, np.newaxis]

	@cached_property
	def latitude_column_radians(self) -> np.ndarray:
		return self.latitude_1d_radians[:, np.newaxis]

	@cached_property
	def latitude_map(self) -> np.ndarray:
		return np.repeat(self.latitude_column, repeats=self.width, axis=1)

	@cached_property
	def latitude_map_radians(self) -> np.ndarray:
		return np.repeat(self.latitude_column_radians, repeats=self.width, axis=1)
