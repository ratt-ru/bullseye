import pylab
def png_export(image,output_prefix,freq_dim_frequencies):
  i = pylab.imshow(image[::-1,:],interpolation='nearest',cmap = pylab.get_cmap('hot'))
  i.write_png(output_prefix+'.png',noscale=True)
  pylab.close('all')