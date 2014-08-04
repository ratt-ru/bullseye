import argparse
import numpy as np
import pylab
import math
from pyrap.tables import *
from helpers import data_set_loader
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="A small utility to write out a set of jones matricies to a new casa table")
	parser.add_argument("input_ms", help="Name of MS to modify", type=str)
	parser.add_argument("no_direction", help="Name of model FITS image",type=int)
	parser_args = vars(parser.parse_args())
	data = data_set_loader.data_set_loader(parser_args['input_ms'])
	data.read_head()
	#compute set of timestamp indicies:
	casa_ms_table = table(parser_args['input_ms'],ack=False,readonly=False)
	time_window_centre = casa_ms_table.getcol("TIME")
	time_indicies = np.zeros(casa_ms_table.nrows(),dtype=np.intp)
	
	timestamp_index = 0
	last_timestamp_time = time_window_centre[0]
	print "FOUND %d ROWS IN MAIN TABLE" % casa_ms_table.nrows()
	for r in range(0,casa_ms_table.nrows()):
	  current_time = time_window_centre[r]
	  if current_time > last_timestamp_time:
	    timestamp_index+=1
	    last_timestamp_time = current_time
	  time_indicies[r] = timestamp_index
	no_timestamps_read = timestamp_index + 1  #the last timestamp may be partially read depending on memory constraints
	
	print "GENERATING FOR %d DIRECTIONS" % parser_args["no_direction"]
	print "FOUND %d TIMESTAMPS IN THIS MS" % no_timestamps_read
	#generate a set of identity elements (at a later point the user may want to hook this into a montblanc pipeline)
	jones_data = np.empty([no_timestamps_read,data._no_antennae,parser_args["no_direction"],data._no_spw,data._no_channels,4],order='C',dtype=np.complex128)
	jones_data[:,:,:,:,:] = np.array([1.0+0.0j,0.0+0.0j,0.0+0.0j,1.0+0.0j])
	spw_id_data = np.empty([no_timestamps_read,data._no_antennae,parser_args["no_direction"],data._no_spw])
	spw_id_data[:,:,:] = np.array(range(0,data._no_spw))
	direction_id_data = np.empty([no_timestamps_read,data._no_antennae,parser_args["no_direction"],data._no_spw])
	for i in range(0,parser_args["no_direction"]):
	  direction_id_data[:,:,i] = np.array([i]*data._no_spw)
	antenna_id_data = np.empty([no_timestamps_read,data._no_antennae,parser_args["no_direction"],data._no_spw])
	for i in range(0,data._no_antennae):
	  antenna_id_data[:,i] = np.array([i]*(data._no_spw*parser_args["no_direction"])).reshape(parser_args["no_direction"],data._no_spw)
	ant_col = makescacoldesc("ANTENNA_ID",0)
	dir_col = makescacoldesc("DIRECTION_ID",0)
	spw_col = makescacoldesc("SPW_ID",0)
	jones_col = makearrcoldesc("JONES", 0+0j, ndim=2,shape=[data._no_channels,4])
	td = maketabdesc([jones_col,ant_col,dir_col,spw_col])
	output_table = table(parser_args["input_ms"]+"/DDE_CALIBRATION",td,nrow=no_timestamps_read*data._no_antennae*parser_args["no_direction"]*data._no_spw)
	output_table.putcol("JONES",jones_data.reshape(no_timestamps_read*data._no_antennae*parser_args["no_direction"]*data._no_spw,data._no_channels,4))
	output_table.putcol("SPW_ID",spw_id_data.reshape(no_timestamps_read*data._no_antennae*parser_args["no_direction"]*data._no_spw))
	output_table.putcol("DIRECTION_ID",direction_id_data.reshape(no_timestamps_read*data._no_antennae*parser_args["no_direction"]*data._no_spw))
	output_table.putcol("ANTENNA_ID",antenna_id_data.reshape(no_timestamps_read*data._no_antennae*parser_args["no_direction"]*data._no_spw))
	#reference new DDE_CALIBRATION in MAIN table
	try:
	  casa_ms_table.removekeyword("DDE_CALIBRATION")
	except:
	  print "COULD NOT REMOVE PREVIOUS KEYWORD FROM MAIN TABLE, ASSUMING THIS IS THE FIRST TIME THE DDE TERMS ARE ADDED TO THE MS"
	casa_ms_table.putkeyword("DDE_CALIBRATION",output_table)
	#finally close both MAIN and DDE_CALIBRATION tables
	casa_ms_table.close()
	output_table.close()