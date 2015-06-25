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

import pyfits
from pyrap.quanta import quantity
import numpy as np
'''
The following definition can be found in Table 28, 
Definition of the Flexible Image Transport System (FITS),   version 3.0
W. D.  Pence, L.  Chiappetti, C. G.  Page, R. A.  Shaw, E.  Stobie
A&A 524 A42 (2010)
DOI: 10.1051/0004-6361/201015362
'''
FITS_POLARIZATION_CLASSIFIERS = {"I" : 1, #Standard Stokes unpolarized
				 "Q" : 2, #Standard Stokes linear
				 "U" : 3, #Standard Stokes linear
				 "V" : 4, #Standard Stokes circular
				 "RR": -1, #Right-right circular
				 "LL": -2, #Left-left circular
				 "RL": -3, #Right-left cross-circular
				 "LR": -4, #Left-right cross-circular
				 "XX": -5, #X parallel linear
				 "YY": -6, #Y parallel linear
				 "XY": -7, #XY cross linear
				 "YX": -8}  #YX cross linear
'''
Routine to store the data in row major format as described
by the FITS standard (see citation above). We're assuming the pixels
are linearly spaced, the coordinate system is orthogonal (SIN) RA,DEC
IAU equatorial coordinates.

See:
Representations of celestial coordinates in FITS
M. R. Calabretta, E. W. Greisen
A&A 395 (3) 1077-1122 (2002)
DOI: 10.1051/0004-6361:20021327
'''
def save_to_fits_image(name,size_l,size_m,
		   cell_l,cell_m,
		   centre_px_l,centre_px_m,
		   pointing_ra,pointing_dec,
		   polarization_term,
		   ref_wavelength,
		   delta_wavelength,
		   cube_channel_dim_size,
		   data):
  if data.size != size_l*size_m*cube_channel_dim_size:
    raise Exception("Data size must be equal to size_l * size_m * no_channels_in_cube")
  if not (data.dtype == np.float32 or data.dtype == np.float64):
    raise Exception("Expected float or double typed data but got %s" % data.dtype)
  fortran_ordered_data = data.astype(data.dtype,order="F",copy=False).reshape(cube_channel_dim_size,1,size_m,size_l)
  pri_hdr = pyfits.PrimaryHDU(fortran_ordered_data,do_not_scale_image_data=True)
  pri_hdr.header.append(("CRPIX1",centre_px_l,"Pixel coordinate of reference point"))
  pri_hdr.header.append(("CDELT1",-cell_m.get_value("deg"),"step per m pixel"))
  pri_hdr.header.append(("CTYPE1","RA---SIN","Orthog projection"))
  pri_hdr.header.append(("CRVAL1",pointing_ra.get_value("deg"),"RA value"))
  pri_hdr.header.append(("CUNIT1","deg","units are always degrees"))
  pri_hdr.header.append(("CRPIX2",centre_px_m,"Pixel coordinate of reference point"))
  pri_hdr.header.append(("CDELT2",cell_l.get_value("deg"),"step per l pixel"))
  pri_hdr.header.append(("CTYPE2","DEC--SIN","Orthog projection"))
  pri_hdr.header.append(("CRVAL2",pointing_dec.get_value("deg"),"DEC value"))
  pri_hdr.header.append(("CUNIT2","deg","units are always degrees"))
  pri_hdr.header.append(("CRPIX3",1,"Pixel coordinate of reference point"))
  pri_hdr.header.append(("CDELT3",1,"dummy value"))
  pri_hdr.header.append(("CTYPE3","STOKES","Polarization"))
  pri_hdr.header.append(("CRVAL3",FITS_POLARIZATION_CLASSIFIERS[polarization_term],"Polarization identifier"))
  pri_hdr.header.append(("CUNIT3"," ","Polarization term is unitless"))
  pri_hdr.header.append(("CRPIX4",1,"Pixel coordinate of reference point"))
  pri_hdr.header.append(("CDELT4",delta_wavelength,"dummy velocity value"))
  pri_hdr.header.append(("CTYPE4","WAVELENGTH","Wavelength in metres"))
  pri_hdr.header.append(("CRVAL4",ref_wavelength,"first wavelength in the cube"))
  pri_hdr.header.append(("CUNIT4","m","wavelength in m"))
  #pri_hdr.header.append(("LONPOLE",180 if pointing_dec.get_value("deg") < 0 else 0,"Native longitude of celestial pole"))
  fits = pyfits.HDUList([pri_hdr])
  fits.writeto(name,clobber=True)