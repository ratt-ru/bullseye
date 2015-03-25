import ctypes
import os

def load_library(architecture,precision):
  if architecture not in ['CPU','GPU']:
    raise Exception("Invalid architecture, only CPU or GPU allowed")
  if precision not in ['single','double']:
    raise Exception("Invalid precision mode selected, only single or double allowed")
  mod_path = os.path.dirname(__file__)
  libimaging = None
  if architecture == 'CPU':
    if precision == 'single':
      libimaging = ctypes.pydll.LoadLibrary("%s/cbuild/algorithms/single/libimaging32.so" % mod_path)
    else:
      libimaging = ctypes.pydll.LoadLibrary("%s/cbuild/algorithms/double/libimaging64.so" % mod_path)
  elif architecture == 'GPU':
    if precision == 'single':
      libimaging = ctypes.pydll.LoadLibrary("%s/cbuild/gpu_algorithm/single/libgpu_imaging32.so" % mod_path)
    else:
      libimaging = ctypes.pydll.LoadLibrary("%s/cbuild/gpu_algorithm/double/libgpu_imaging64.so" % mod_path)
  return libimaging
