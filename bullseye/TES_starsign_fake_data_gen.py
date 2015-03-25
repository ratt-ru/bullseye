'''
very stupid script to make the painful task of generating test data
somewhat less painful... this is certainly not production material
but gets the job done...

Currently we're imaging the Elder Scrolls starsigns with
LOFAR ! Should make for some interresting eyecandy..
'''
import pyfits
import os
ms_name = "test_data_visibilities.ms"
npix=1024
cellsize="2arcsec"
fits="in.fits"
modded_fits="TES_sky.fits"
os.system("lwimager ms=%s npix=%d cellsize=%s fits=%s" % (ms_name,npix,cellsize,fits))

print "CREATED FITS FILE FROM MS"

ff = pyfits.open(fits)
ff[0].data[:,:,:,:] = 0

ff[0].data[0,0,486,npix-549] = 1
ff[0].data[0,0,513,npix-549] = 1
ff[0].data[0,0,481,npix-519] = 1
ff[0].data[0,0,471,npix-505] = 1
ff[0].data[0,0,471,npix-471] = 1
ff[0].data[0,0,481,npix-452] = 1
ff[0].data[0,0,511,npix-455] = 1
ff[0].data[0,0,505,npix-489] = 1
ff[0].data[0,0,536,npix-477] = 1
ff[0].data[0,0,522,npix-521] = 1

ff[0].data[0,0,281,npix-48] = 1
ff[0].data[0,0,287,npix-97] = 1
ff[0].data[0,0,289,npix-117] = 1
ff[0].data[0,0,260,npix-108] = 1
ff[0].data[0,0,267,npix-127] = 1
ff[0].data[0,0,309,npix-97] = 1
ff[0].data[0,0,311,npix-113] = 1
ff[0].data[0,0,324,npix-80] = 1
ff[0].data[0,0,340,npix-107] = 1
ff[0].data[0,0,334,npix-57] = 1

ff[0].data[0,0,694,npix-300] = 1
ff[0].data[0,0,707,npix-323] = 1
ff[0].data[0,0,714,npix-312] = 1
ff[0].data[0,0,726,npix-303] = 1
ff[0].data[0,0,720,npix-290] = 1
ff[0].data[0,0,726,npix-290] = 1
ff[0].data[0,0,739,npix-264] = 1
ff[0].data[0,0,754,npix-290] = 1
ff[0].data[0,0,781,npix-268] = 1
ff[0].data[0,0,772,npix-301] = 1
ff[0].data[0,0,751,npix-318] = 1
ff[0].data[0,0,759,npix-312] = 1
ff[0].data[0,0,758,npix-329] = 1
ff[0].data[0,0,769,npix-326] = 1
ff[0].data[0,0,781,npix-334] = 1
ff[0].data[0,0,793,npix-341] = 1
ff[0].data[0,0,718,npix-339] = 1
ff[0].data[0,0,735,npix-355] = 1

ff[0].data[0,0,875,npix-813] = 1
ff[0].data[0,0,836,npix-836] = 1
ff[0].data[0,0,867,npix-849] = 1
ff[0].data[0,0,904,npix-833] = 1
ff[0].data[0,0,842,npix-867] = 1
ff[0].data[0,0,862,npix-879] = 1
ff[0].data[0,0,822,npix-886] = 1
ff[0].data[0,0,864,npix-900] = 1

ff[0].data[0,0,105,npix-760] = 1
ff[0].data[0,0,118,npix-790] = 1
ff[0].data[0,0,96,npix-813] = 1
ff[0].data[0,0,76,npix-850] = 1
ff[0].data[0,0,123,npix-870] = 1
ff[0].data[0,0,78,npix-919] = 1
ff[0].data[0,0,133,npix-951] = 1
ff[0].data[0,0,162,npix-908] = 1
ff[0].data[0,0,164,npix-803] = 1
ff[0].data[0,0,192,npix-787] = 1
ff[0].data[0,0,183,npix-773] = 1
ff[0].data[0,0,166,npix-771] = 1
ff[0].data[0,0,163,npix-756] = 1
ff[0].data[0,0,204,npix-797] = 1
ff[0].data[0,0,165,npix-837] = 1
ff[0].data[0,0,193,npix-837] = 1
ff[0].data[0,0,202,npix-853] = 1
ff[0].data[0,0,175,npix-885] = 1
ff[0].data[0,0,191,npix-870] = 1
ff[0].data[0,0,205,npix-870] = 1
ff[0].data[0,0,247,npix-835] = 1
ff[0].data[0,0,276,npix-809] = 1
ff[0].data[0,0,284,npix-775] = 1

ff.writeto(modded_fits)

print "SAVED FAKE DATA TO FITS"
