#!/usr/bin/env python


from datetime import datetime
from typing import Optional, Callable, TypeVar


T = TypeVar('T')


_start_time = None


# TODO: should just use logging instead of this
def tprint(s: str, start=None, is_start=False, **kwargs):
	global _start_time

	now = datetime.now()

	if is_start:
		_start_time = now

	if (start is None) and (_start_time is not None):
		start = _start_time

	# TODO: better formatting
	# now_str = now.strftime()
	now_str = str(now)

	if start is not None:
		delta = now - start
		delta_str = str(delta)  # TODO: better
		s = f'[{now_str}] [{delta_str}] {s}'
	else:
		s = f'[{now_str}] {s}'
	print(s)
	return now


class Parameter:

	__slots__ = ['_value', '_type', '_name', '_default', '_min', '_max', '_step', '_minlength', '_maxlength']

	def __init__(
			self,
			type: Callable[..., T],
			name: str = '',
			default: Optional[T] = None,
			min: Optional[T] = None,
			max: Optional[T] = None,
			step: Optional[T] = None,
			minlength: Optional[int] = None,
			maxlength: Optional[int] = None,
			):

		if (min is not None) and not isinstance(min, type):
			min = type(min)

		if (max is not None) and not isinstance(max, type):
			max = type(max)

		if (min is not None) and (max is not None) and (min > max):
			raise ValueError(f'min {min} > max {max}')

		if (minlength is not None) and (maxlength is not None) and (minlength > maxlength):
			raise ValueError(f'minlength {min} > maxlength {max}')

		if default is not None:
			if not isinstance(default, type):
				default = type(default)
			if ((min is not None) and (default < min)) or ((max is not None) and (default > max)):
				raise ValueError(f'Default value {default} is not in range: [{min}, {max}]')

		self._type = type
		self._name = name
		self._default = default
		self._min = min
		self._max = max
		self._step = step

		self._minlength = minlength
		self._maxlength = maxlength

		# self._value = None  # I don't think this is necessary when using slots?
		if default is not None:
			self._set(default)

	@property
	def value(self):
		return self._value

	@value.setter
	def value(self, value):
		self._set(value)

	@property
	def type(self):
		return self._type

	@property
	def name(self):
		return self._name

	@property
	def default(self):
		return self._default

	@property
	def min(self):
		return self._min

	@property
	def max(self):
		return self._max

	@property
	def step(self):
		return self._step

	@property
	def minlength(self) -> Optional[int]:
		return self._minlength

	@property
	def maxlength(self) -> Optional[int]:
		return self._maxlength

	def _set(self, new_value):

		new_value = self.type(new_value)

		if (self.maxlength is not None) and (len(new_value) > self.maxlength):
			raise ValueError('Value too short')

		if (self.minlength is not None) and (len(new_value) < self.minlength):
			raise ValueError('Value too long')

		if self.min is not None:
			new_value = max(self.min, new_value)

		if self.step is not None:
			# Round to multiple of step
			delta_min = new_value - self.min
			delta_min = round(new_value / self.step) * self.step
			new_value = self.min + delta_min

		if self.max is not None:
			new_value = min(self.max, new_value)

		self._value = new_value

	def __str__(self) -> str:
		return self.__repr__()

	def __repr__(self) -> str:
		return f'Parameter(name="{self.name}", type={self.type.__name__}, value={self.value}, default={self.default}, min={self.min}, max={self.max}, step={self.step})'

	def __eq__(self, other):
		if isinstance(other, Parameter):
			return self.value == other.value
		else:
			return self.value == other

	def __hash__(self):
		return hash((self._value, self._type, self._default, self._min, self._max, self._step))

