#/usr/bin/python
'''
Bullseye:
An accelerated targeted facet imager
Category: Radio Astronomy / Widefield synthesis imaging

Authors: Benjamin Hugo, Oleg Smirnov, Cyril Tasse, James Gain
Contact: hgxben001@myuct.ac.za

Copyright (C) 2014-2015 Rhodes Centre for Radio Astronomy Techniques and Technologies
Department of Physics and Electronics
Rhodes University
Artillery Road P O Box 94
Grahamstown
6140
Eastern Cape South Africa

Copyright (C) 2014-2015 Department of Computer Science
University of Cape Town
18 University Avenue
University of Cape Town
Rondebosch
Cape Town
South Africa

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''
from pyrap.tables import table
import argparse
from pyrap.quanta import quantity
from pyrap import quanta
import numpy as np
import math
import pylab
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='UVW Coordinate generator')
  parser.add_argument('input_ms', help='Name of the measurement set to read', type=str)
  parser_args = vars(parser.parse_args())
  casa_ms_table = table(parser_args['input_ms']+'::ANTENNA',ack=False,readonly=True)
  no_antennae = casa_ms_table.nrows()
  antennae_coords = casa_ms_table.getcol("POSITION")
  antennae_names = casa_ms_table.getcol("NAME")
  casa_ms_table.close()
  casa_ms_table = table(parser_args['input_ms'],ack=False,readonly=True)
  actual_uvw_coords = casa_ms_table.getcol("UVW")
  antenna1 = casa_ms_table.getcol("ANTENNA1")
  antenna2 = casa_ms_table.getcol("ANTENNA2")
  time = casa_ms_table.getcol("TIME")
  casa_ms_table.close()
  for i in range(0,no_antennae):
      name = antennae_names[i]
      position = antennae_coords[i]
      print "\t %s has position [%f , %f , %f]" % (name,position[0],position[1],position[2])
  
  casa_ms_table = table(parser_args['input_ms']+"::FIELD",ack=False,readonly=True)
  field_centres = casa_ms_table.getcol("REFERENCE_DIR")
  field_centre_names = casa_ms_table.getcol("NAME")
  print "REFERENCE CENTRES OBSERVED:"
  for i in range(0, len(field_centres)):
      field_centres[i,0,0] = quantity(field_centres[i,0,0],"rad").get_value("arcsec")
      field_centres[i,0,1] = quantity(field_centres[i,0,1],"rad").get_value("arcsec")
      print "\tCENTRE OF %s (FIELD ID %d): RA: %s, DEC: %s" % (field_centre_names[i],i,
							       quantity(field_centres[i,0,0],"arcsec").get("deg").formatted("[+-]dd.mm.ss.t.."),
							       quantity(field_centres[i,0,1],"arcsec").get("deg").formatted("[+-]dd.mm.ss.t.."))
  field_centres = casa_ms_table.getcol("REFERENCE_DIR") # those were just to print, lets read them as radians
  casa_ms_table.close()
  casa_ms_table = table(parser_args['input_ms']+"::SPECTRAL_WINDOW",ack=False,readonly=True)
  no_channels = casa_ms_table.getcell("NUM_CHAN", 0) #Note: assuming all spectral windows will have an equal number of channels
  no_spw = casa_ms_table.nrows()
  spw_centres = casa_ms_table.getcol("REF_FREQUENCY")
  print "%d CHANNELS IN OBSERVATION" % no_channels
  print "%d SPECTRAL WINDOWS IN OBSERVATION" % no_spw
  chan_freqs = casa_ms_table.getcol("CHAN_FREQ") # this will have dimensions [number of SPWs][num channels in spectral window]
  chan_wavelengths = quanta.constants['c'].get_value("m/s") / chan_freqs # lambda = speed of light / frequency
  for spw in range(0,no_spw):
    for c in range(0,no_channels):
      print "\t Spectral window %d, channel %d has a wavelength of %f m" % (spw,c,chan_wavelengths[spw,c])
  
  '''
  u,v coordinate generation through parametric equations
  '''
  antennae_coords = antennae_coords - antennae_coords[0]
  pylab.figure()
  pylab.plot(antennae_coords[:,0],antennae_coords[:,1],'rx')
  pylab.figure()
  hrs = 24
  max_t = hrs / 24.0 * 2 * np.pi
  ra = 0
  t = np.linspace(0,max_t,1000) + ra 
  declination = -90 * np.pi / 180
  for i in range(0,no_antennae):
    for j in range(0,no_antennae):
      for wavelength in chan_wavelengths:
	Lx,Ly,Lz = antennae_coords[i] - antennae_coords[j]
	u = np.sin(t)*Lx + np.cos(t)*Ly
	v = -np.sin(declination)*np.cos(t)*Lx + np.sin(declination)*np.sin(t)*Ly + np.cos(declination)*Lz
	w = np.cos(declination)*np.cos(t)*Lx - np.cos(declination)*np.sin(t)*Ly + np.sin(declination)*Lz
	pylab.plot(u,v)
  pylab.xlabel("m")
  pylab.ylabel("m")
  pylab.show()