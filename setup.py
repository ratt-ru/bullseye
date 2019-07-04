import os
import sys
import warnings
from os.path import join as pjoin
from setuptools import setup
import subprocess
from setuptools.command.install import install
from setuptools.command.sdist import sdist
from distutils.command.build import build
pkg='bullseye'
__version__ = "0.4.1.0"
build_root=os.path.dirname(__file__)

def readme():
    with open('README.md') as f:
        return f.read()


    install.user_options = install.user_options + [
        ('compopts=', None, 'Any additional compile options passed to CMake')
    ]
    def initialize_options(self):
        install.initialize_options(self)
        self.compopts = None

def backend(compile_options):
    if compile_options is not None:
        print >> sys.stderr, "Compiling extension libraries with user defined options: '%s'"%compile_options
    path = pjoin(build_root, pkg, "mo", 'cbuild')
    try:
        subprocess.check_call(["mkdir", path])
    except:
        warnings.warn("%s already exists in your source folder. We will not create a fresh build folder, but you "
                      "may want to remove this folder if the configuration has changed significantly since the "
                      "last time you run setup.py" % path)
    subprocess.check_call(["cd %s && cmake %s .. && make" %
                           (path, compile_options if compile_options is not None else ""), ""], shell=True)

class custom_install(install):
    install.user_options = install.user_options + [
        ('compopts=', None, 'Any additional compile options passed to CMake')
    ]
    def initialize_options(self):
        install.initialize_options(self)
        self.compopts = None

    def run(self):
        backend(self.compopts)
        install.run(self)

class custom_build(build):
    build.user_options = build.user_options + [
        ('compopts=', None, 'Any additional compile options passed to CMake')
    ]
    def initialize_options(self):
        build.initialize_options(self)
        self.compopts = None

    def run(self):
        backend(self.compopts)
        build.run(self)

class custom_sdist(sdist):
    def run(self):
        bpath = pjoin(build_root, pkg, 'mo', 'cbuild')
        if os.path.isdir(bpath):
            subprocess.check_call(["rm", "-rf", bpath])
        sdist.run(self)

def define_scripts():
    #these must be relative to setup.py according to setuputils
    DDF_scripts = [os.path.join("bullseye", "bin", script_name) for script_name in ['bullseye_pipeliner.py', 'bullseye_gui.py']]
    return DDF_scripts

setup(name=pkg,
    version=__version__,
    description='Bullseye Measurement Operator',
    long_description=readme(),
    url='https://github.com/ratt-ru/bullseye.git',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    author='Benjamin Hugo',
    author_email='bennahugo@aol.com',
    license='MIT',
    cmdclass={'install': custom_install,
              'sdist': custom_sdist,
              'build': custom_build
             },
    packages=['bullseye'],
    scripts=define_scripts(),
    install_requires=['numpy','matplotlib<=1.5.0','scipy','python-casacore<=3.0.0','astropy<=3.0','pycairo','PyGObject'],
    include_package_data=True,
    zip_safe=False)
