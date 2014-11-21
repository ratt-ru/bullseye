bullseye
========

A GPU-based facet imager

Dependencies:
  1. casacore (must be compiled and installed from source). SKA repos are useful to resolve some of its dependencies. Rest of the dependencies   are available through the mirror at leg.uct.ac.za.
  2. numpy
  3. pyrap
  4. pyfits
  5. gtk3.0 with engines and python dev packages and pycairo (and maybe any cairo-dev packages). Important: install package python-gi-cairo, otherwise the gtk drawing area's on draw is never called for some truely bizare reason.
  6. (to edit gui: glade for gtk 3.0 is needed)
  7. cmake
  8. gcc and g++ 4.6 / 4.7 / 4.8
  9. libboost-all and libboost-dev
  10. python dev packages (required to write python C++ extensions)

Future dependencies:
  1. OpenMP
  2. nvidia-cuda-toolkit
  3. current nvidia drivers
  4. Possibly OpenCL
  5. Possibly MPI or some other RPC (thinking boost / cap-n-proto / something else)
