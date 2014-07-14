#!/usr/bin/python
import pyfits
import argparse
import numpy as np
import pylab
import math

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="A small utility to compare two FITS images")
	parser.add_argument("input_fits_1", help="Name of the first FITS image", type=str)
	parser.add_argument("input_fits_2", help="Name of the second FITS image", type=str)
	parser.add_argument("model_image", help="Name of model FITS image",type=str)

	parser_args = vars(parser.parse_args())

	img1 = pyfits.open(parser_args["input_fits_1"])
	img2 = pyfits.open(parser_args["input_fits_2"])
	model = pyfits.open(parser_args["model_image"])
	
	print "\033[44;33mFirst image info:\033[0m"
	img1.info()
	print "\033[44;33mSecond image info:\033[0m"
	img2.info()
	print "\033[44;33mModel image info:\033[0m"
	model.info()

	assert(len(img1) == len(img2))
	assert(len(model) == len(img1))
	assert(len(img1) == 1)
	assert(type(img1[0]) is pyfits.PrimaryHDU)
	assert(type(img2[0]) is pyfits.PrimaryHDU)
	assert(type(model[0]) is pyfits.PrimaryHDU)
	assert(img1[0].data.shape == img2[0].data.shape)
	assert(img1[0].data.shape == model[0].data.shape)
	assert(img1[0].data.shape[0] == 1 and img1[0].data.shape[1] == 1)	

	print ""
	print("\033[44;33mComputing minimums and maximums... "),
	max_img1 = np.max(img1[0].data[0,0,:,:])
	min_img1 = np.min(img1[0].data[0,0,:,:])
	max_img2 = np.max(img2[0].data[0,0,:,:])
	min_img2 = np.min(img2[0].data[0,0,:,:])
	min_model = np.min(model[0].data[0,0,:,:])
	max_model = np.max(model[0].data[0,0,:,:])

	print "\033[93m<DONE>\033[0m"

	print("\033[44;33mLinearly normalizing images to unity... "),
	#formulae thanks to wikipedia :-D, http://en.wikipedia.org/wiki/Normalization_%28image_processing%29)
	normalized_img1 = ((img1[0].data[0,0,:,:] - min_img1)*((1-0)/(max_img1-min_img1))+0)[::-1,:]
	normalized_img2 = ((img2[0].data[0,0,:,:] - min_img2)*((1-0)/(max_img2-min_img2))+0)[::-1,:]
	normalized_model = ((model[0].data[0,0,:,:] - min_model)*((1-0)/(max_model-min_model))+0)[::-1,:]
	print "\033[93m<DONE>\033[0m"

	print("\033[44;33mComputing means and standard deviations... "),	
	avg_img1 = np.mean(normalized_img1)
	avg_img2 = np.mean(normalized_img2)
	avg_model = np.mean(normalized_model)

	stddev_img1 = np.std(normalized_img1)
	stddev_img2 = np.std(normalized_img2)
	stddev_model = np.std(normalized_model)
	print "\033[93m<DONE>\033[0m"

	print("\033[44;33mComputing difference map and errors... "),
	diffmap = normalized_img2 - normalized_img1
	mse = (diffmap**2).mean()
	err_img1 = normalized_model - normalized_img1
	mse_model_img1 = (err_img1**2).mean()
	err_img2 = normalized_model - normalized_img2
	mse_model_img2 = (err_img2**2).mean()

	snr_img1 = 20 * math.log10(1) - 10 * math.log10(mse_model_img1)	
	snr_img2 = 20 * math.log10(1) - 10 * math.log10(mse_model_img2)	
	snr_img1_to_img2 =  20 * math.log10(1) - 10 * math.log10(mse)
	print "\033[93m<DONE>\033[0m"

	#plot it up :-)
	pylab.figure()
	pylab.imshow(normalized_img1,interpolation='nearest',cmap=pylab.get_cmap('hot'))
	pylab.title(parser_args["input_fits_1"])
	pylab.colorbar()

	pylab.figure()
	pylab.imshow(normalized_img2,interpolation='nearest',cmap=pylab.get_cmap('hot'))
	pylab.title(parser_args["input_fits_2"])
	pylab.colorbar()

	pylab.figure()
	pylab.imshow(normalized_model,interpolation='nearest',cmap=pylab.get_cmap('hot'))
	pylab.title(parser_args["model_image"])
	pylab.colorbar()

	pylab.figure()
	pylab.imshow(diffmap,interpolation='nearest',cmap=pylab.get_cmap('hot'))
	pylab.title("Difference map (image 2 - image 1)")
	pylab.colorbar()

	print "\033[44;32mStatistics:"
	print "\033[93m%s [image 1] (normalized):" % parser_args["input_fits_1"]
	print "\033[33m\tAverage: %f" % avg_img1
	print "\033[33m\tStandard deviation: %f" % stddev_img1
	print "\033[33m\tMean Squared Error between image and model: %f" %mse_model_img1
	print "\033[33m\tPeak Signal to Noise (model to difference): %fdB" % snr_img1
	print "\033[93m%s [image 2] (normalized):" % parser_args["input_fits_2"]
	print "\033[33m\tAverage: %f" % avg_img2
	print "\033[33m\tStandard deviation: %f" % stddev_img2
	print "\033[33m\tMean Squared Error between image and model: %f" %mse_model_img2
	print "\033[33m\tPeak Signal to Noise (model to difference): %fdB" % snr_img2
	print "\033[93m%s [model image] (normalized):" % parser_args["model_image"]
	print "\033[33m\tAverage: %f" % avg_model
	print "\033[33m\tStandard deviation: %f" % stddev_model
	print "\033[93mMean Squared Error between normalized input images: %fdB" % mse
	print "\033[93mPeak Signal to Noise between normalized input images: %fdB\033[0m" % snr_img1_to_img2


	pylab.show()
