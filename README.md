bullseye
========

A multithreaded CPU & GPU-based facet imager

(C) Benjamin Hugo, James Gain, Oleg Smirnov and Cyril Tasse (2014-2015)

Dependencies:
  1. Python 2.7.6
  2. casacore
  3. numpy
  4. pyrap
  5. matplotlib
  6. pyfits
  7. gtk3.0 with engines and python dev packages and pycairo (and maybe any cairo-dev packages). Important: install package python-gi-cairo, otherwise the gtk drawing area's on draw is never called for some truely bizare reason.
  8. (to edit gui: glade for gtk 3.0 is needed)
  9. cmake
  10. The GNU C++ compiler >= 4.8
  11. libboost-all and libboost-dev
  12. OpenMP
  13. CUDA toolkit (nvcc,nvprof) >=5.0
  14. Montage (available: http://montage.ipac.caltech.edu/index.html). Montage's bin folder must be in the PATH
  15. CfitsIO
  16. WcsLib

Build instructions (outputs CPU and GPU single and double precision libraries and a python wrapper for these)
- run: python setup.py install --user (this will install into the python user directory)

Run instructions
- Navigate to the bullseye/bullseye directory
- "./bullseye.py --help" to display a full list of options
- Enjoy

Toy GUI is available (proof of concept demonstrator for targetted faceting)
- "./bullseye_frontend.py"
