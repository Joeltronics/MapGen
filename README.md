# Map generator

How to use:

(Assuming you've gotten all of your Python dependencies installed and such)

`main.py run`

Then, open the printed URL (likely http://127.0.0.1:7860/) in a web browser to run the Gradio UI

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
5. Generate precipitation map, based on latitude + noise
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

**More advanced simulation**

* Simulate prevailing winds, and their effect on precipitation (orographic precipitation & rain shadows)
* Add rivers, and more realistic erosion simulation

## Real data processing

-



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

### Engine & Performance

* Generate different views directly from data, instead of generating equirectangular projection and remapping
* Some Gradio performance issues
	* By default it base64 encodes images, which adds a lot of overhead when we have lots of large images
	* Lots of possible ways to improve this - save to a file and link that instead; or use gradio Gallery objects everywhere
* Cache noise to reuse it
	* e.g. if re-generating the same seed with a few different parameters or higher resolution
* Use more pyfastnoisesimd features

### Known Bugs & Problems

* Same seed can give totally different results
	* It seems to give the same results within a run, but if you exit and rerun, it will give completely different results
	* Not sure if this is a bug in pyfastnoisesimd, or if I'm using it wrong - haven't really looked into this yet
* There seems to be a "shadow" to the right side of land
	* Most noticeable in the biome map, but I think it's in the main map as well (unless this is just gradient shading?)
	* Might be an X off by one error somewhere (ocean mask generation?)
* Gradient calculation has issues
	* This doesn't end up mattering because we just normalize to [-1, 1], but would like to fix this
