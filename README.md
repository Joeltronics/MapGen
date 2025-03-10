# Map generator

How to use:

(Assuming you've gotten all of your Python dependencies installed and such)

Just run `python3 main.py`. Then, open the printed URL (likely http://127.0.0.1:7860/) in a web browser to run the Gradio UI.

Right now, Python 3.10 is required - in theory this should would be compatible with 3.11 as well, but it requires Numba, which does not currently support 3.11.

## How does it work?

The current general algorithm is:

1. Generate elevation
	- Based on fractal simplex noise
	- Scale depending on water amount
2. Erode elevation map
	- Based on more noise (valley noise), not a realistic model
3. Calculate gradient from elevation
4. Generate temperature map, based on latitude + elevation + noise
5. Generate precipitation map
	- Base amount is based on latitude + noise
	- Scale for orographic precipitation & shadows based on where wind is going uphill or downhill
6. Determine color:
	- Ocean base color from temperature & depth
	- Land base color from temperature & precipitation
	- Add shading based on gradient

### Future algorithm improvements

**Improved base terrain generation**

Add new method based on tectonic plates

Also improve noise-based options:

* Experiment with more noise options, such as domain warping
* Improve existing options to be more realistic (exponentially distributed)
* More realistic continental shelf simulation

**Simulate by season**

* It currently only simulates weather once, but it would be more accurate to run simulations throughout the year, and average them to get total year-round results (as well as then having the year-round results available)
* Could run monthly, but this is probably overkill - even just running simulations for 3 seasons (using the same sim for spring & fall) would be a big improvement

**More advanced simulation**

* Current orographic precipitation & shadow model is a rough approximation, which is relatively fast to compute compared to a proper iterative model, but not as realistic
* Wind model is very basic (almost entirely just based on latitude), which leads to flawed precipitation model results as well
* Add rivers, and more realistic erosion simulation

## Real data processing

There are some basics implemented, but they're not incorporated into the generation yet

## FAQ

**Why Python? Why not something more performant?**

I'd love to write a C++ GUI for this, with 3D rendering and everything, and maybe even move a lot of the computation to the GPU.
But that would be a lot more work, and I'd rather focus on the core generation for now.

**Why the ugly pyopensimplex hacks?**

pyopensimplex doesn't have an optimized method to generate from an arbitrary set of coordinates.
I've moved to pyfastnoisesimd wherever possible, but there are still a few use cases that need pyopensimplex (4D noise, for one).

## TODO

See "future algorithm improvements" above for improvements to the core algorithm.

Other TODO items:

### UI features

* More options for generating 2D "flat maps"
	* Set width as well as height
	* Specify latitude
	* Options to specify what's on each side of the map (e.g. ocean, mountains, desert, etc)
* Option to regenerate a region of a map at higher resolution
* Better options for saving files
* Improve UI
	* Gradio is great because it was quick to implement, but the result is pretty basic
	* Would like something more akin to Google Earth
	* Not sure if the way to go is a full custom JS UI, or custom components within Gradio

### Engine & Performance

* Generate different views directly from data, instead of generating equirectangular projection and remapping
* Cache noise to reuse it
	* e.g. if re-generating the same seed with a few different parameters or higher resolution
* Use more pyfastnoisesimd features

### Known Bugs & Problems

* There seems to be a "shadow" to the right side of land
	* Most noticeable in the biome map, but I think it's in the main map as well (unless this is just gradient shading?)
	* Might be an X off by one error somewhere (ocean mask generation?)
