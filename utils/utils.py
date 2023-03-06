#!/usr/bin/env python


from datetime import datetime


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
