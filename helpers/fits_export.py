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
				 "XY": -8}  #YX cross linear
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
		   pointing_ra,pointing_dec,
		   polarization_term,
		   equinox,
		   data):
  if data.size != size_l*size_m:
    raise Exception("Data size must be equal to size_l * size_m")
  if not (data.dtype == np.float32 or data.dtype == np.float64):
    raise Exception("Expected float or double typed data but got %s" % data.dtype)
  fortran_ordered_data = data.copy("F").reshape(1,1,size_m,size_l)
  pri_hdr = pyfits.PrimaryHDU(fortran_ordered_data,do_not_scale_image_data=True)
  pri_hdr.header.append(("CRPIX1",size_l/2,"Pixel coordinate of reference point"))
  pri_hdr.header.append(("CDELT1",-cell_l.get_value("deg"),"step per l pixel"))
  pri_hdr.header.append(("CTYPE1","RA---SIN","Orthog projection"))
  pri_hdr.header.append(("CRVAL1",pointing_ra.get_value("deg"),"RA value"))
  pri_hdr.header.append(("CUNIT1","deg","units are always degrees"))
  pri_hdr.header.append(("CRPIX2",size_m/2,"Pixel coordinate of reference point"))
  pri_hdr.header.append(("CDELT2",cell_m.get_value("deg"),"step per m pixel"))
  pri_hdr.header.append(("CTYPE2","DEC--SIN","Orthog projection"))
  pri_hdr.header.append(("CRVAL2",pointing_dec.get_value("deg"),"DEC value"))
  pri_hdr.header.append(("CUNIT2","deg","units are always degrees"))
  pri_hdr.header.append(("CRPIX3",1,"Pixel coordinate of reference point"))
  pri_hdr.header.append(("CDELT3",0,"dummy velocity value"))
  pri_hdr.header.append(("CTYPE3","VELOCITY",""))
  pri_hdr.header.append(("CRVAL3",0,"dummy value"))
  pri_hdr.header.append(("CUNIT3","m/s","velocity in m/s"))
  pri_hdr.header.append(("CRPIX4",1,"Pixel coordinate of reference point"))
  pri_hdr.header.append(("CDELT4",1,"dummy value"))
  pri_hdr.header.append(("CTYPE4","STOKES","Polarization"))
  pri_hdr.header.append(("CRVAL4",FITS_POLARIZATION_CLASSIFIERS[polarization_term],"Polarization identifier"))
  pri_hdr.header.append(("CUNIT4"," ","Polarization term is unitless"))
  pri_hdr.header.append(("LONPOLE",180 if pointing_dec.get_value("deg") < 0 else 0,"Native longitude of celestial pole"))
  pri_hdr.header.append(("RADESYS","FK5" if equinox >= 1984 else "FK4","Mean IAU 1984 equatorial coordinates"))
  pri_hdr.header.append(("EQUINOX",equinox,"Equinox"))
  fits = pyfits.HDUList([pri_hdr])
  fits.writeto(name,clobber=True)