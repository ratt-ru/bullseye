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

import numpy as np
from pyrap.tables import table
from pyrap.tables import taql
from pyrap.measures import measures
from pyrap.quanta import quantity
from pyrap import quanta
import math
import time
import datetime
import pylab
import bullseye.mo.base_types as base_types 
import traceback
from bullseye.helpers.timer import timer
class data_set_loader(object):
    time_to_load_chunks = timer()
    '''
    classdocs
    '''
    def __init__(self, MSName,read_jones_terms = True):
        '''
        Constructor
        '''
        self._MSName = MSName
        self._should_read_jones_terms = read_jones_terms    
    '''
        Read some stats about the MS
        Assume this method only gets called from __init__
        (This can be checked against listobs in CASAPY)
    '''
    def read_head(self):
	print "READING MEASUREMENT SET HEAD OF '%s'" % self._MSName 
        casa_ms_table = table(self._MSName+'::OBSERVATION',ack=False,readonly=True)
        self._observer_name = casa_ms_table.getcell("OBSERVER", 0)
        self._telescope_name = casa_ms_table.getcell("TELESCOPE_NAME", 0)
        self._observation_start = casa_ms_table.getcell("TIME_RANGE", 0)[0]
        self._observation_end = casa_ms_table.getcell("TIME_RANGE", 0)[1]
        casa_ms_table.close()
        casa_ms_table = table(self._MSName+"::POINTING",ack=False,readonly=True)
        self._epoch = casa_ms_table.getcolkeyword("DIRECTION","MEASINFO")["Ref"]
        casa_ms_table.close()
        fmt = '%Y-%m-%d %H:%M:%S %Z'
        print "OBSERVED BY %s ON %s FROM %s TO %s SINCE EPOCH %s" % (self._observer_name,self._telescope_name,
                                                      self._observation_start,
                                                      self._observation_end,
                                                      self._epoch,
                                                     )
        casa_ms_table.close()
        casa_ms_table = table(self._MSName+'::ANTENNA',ack=False,readonly=True)
        self._no_antennae = casa_ms_table.nrows()
        self._no_baselines = self._no_antennae*(self._no_antennae-1)/2 + self._no_antennae
        print "FOUND %d ANTENNAE:" % (self._no_antennae) 
        for i in range(0,self._no_antennae):
            name = casa_ms_table.getcell("NAME", i)
            position = casa_ms_table.getcell("POSITION", i)
            print "\t %s has position [%f , %f , %f]" % (name,position[0],position[1],position[2])
        self._antennas = casa_ms_table.getcol("POSITION")
        self._maximum_baseline_length = -1
        for p in range(0,self._no_antennae):
	  for q in range(p,self._no_antennae):
	    b = self._antennas[p,:] - self._antennas[q,:]
	    self._maximum_baseline_length = max(np.linalg.norm(b),self._maximum_baseline_length)
        casa_ms_table.close()
        print "%d UNIQUE BASELINES" % (self._no_baselines)
        casa_ms_table = table(self._MSName+"::POLARIZATION",ack=False,readonly=True)
        self._no_polarization_correlations = casa_ms_table.getcell("NUM_CORR", 0)
        self._polarization_correlations = casa_ms_table.getcell("CORR_TYPE", 0)
        print "%d CORRELATIONS DUE TO POLARIZATION" % self._no_polarization_correlations
        casa_ms_table.close()        
        casa_ms_table = table(self._MSName+"::SPECTRAL_WINDOW",ack=False,readonly=True)
        self._no_channels = casa_ms_table.getcell("NUM_CHAN", 0) #Note: assuming all spectral windows will have an equal number of channels
        self._no_spw = casa_ms_table.nrows()
        self._spw_centres = casa_ms_table.getcol("REF_FREQUENCY")
        print "%d CHANNELS IN OBSERVATION" % self._no_channels
        print "%d SPECTRAL WINDOWS IN OBSERVATION" % self._no_spw
        self._chan_freqs = casa_ms_table.getcol("CHAN_FREQ") # this will have dimensions [number of SPWs][num channels in spectral window]
        self._chan_wavelengths = (quanta.constants['c'].get_value("m/s") / self._chan_freqs).astype(base_types.reference_wavelength_type) # lambda = speed of light / frequency
        for spw in range(0,self._no_spw):
	  for c in range(0,self._no_channels):
            print "\t Spectral window %d, channel %d has a wavelength of %f m" % (spw,c,self._chan_wavelengths[spw,c])
        casa_ms_table.close()
        casa_ms_table = table(self._MSName+"::DATA_DESCRIPTION",ack=False,readonly=True)
        spw_indexes = casa_ms_table.getcol("SPECTRAL_WINDOW_ID")
        for i in range(0,self._no_spw):
	  temp = self._chan_wavelengths[i] #deep copy
	  #swap the rows so that we index by the column in DATA_DESCRIPTION
	  self._chan_wavelengths[i] = self._chan_wavelengths[spw_indexes[i]]
	  self._chan_wavelengths[spw_indexes[i]] = temp 
	casa_ms_table.close()
        casa_ms_table = table(self._MSName+"::FIELD",ack=False,readonly=True)
        self._field_centres = casa_ms_table.getcol("REFERENCE_DIR")
        self._field_centre_names = casa_ms_table.getcol("NAME")
	print "REFERENCE CENTRES OBSERVED:"
        for i in range(0, len(self._field_centres)):
	  self._field_centres[i,0,0] = quantity(self._field_centres[i,0,0],"rad").get_value("arcsec")
	  self._field_centres[i,0,1] = quantity(self._field_centres[i,0,1],"rad").get_value("arcsec")
	  print "\tCENTRE OF %s (FIELD ID %d): RA: %s, DEC: %s" % (self._field_centre_names[i],i,
								 quantity(self._field_centres[i,0,0],"arcsec").get("deg").formatted("[+-]dd.mm.ss.t.."),
								 quantity(self._field_centres[i,0,1],"arcsec").get("deg").formatted("[+-]dd.mm.ss.t.."))
        casa_ms_table.close()
        casa_ms_table = table(self._MSName,ack=False,readonly=True)
        #Number of timestamps is NOT this (cannot deduce it from baselines and number of rows, since there may be 0 <= x < no baselines in each timestep): 
        #self._no_timestamps = casa_ms_table.nrows() / self._no_baselines
        #print "%d TIMESTAMPS IN OBSERVATION" % self._no_timestamps
        self._no_rows = casa_ms_table.nrows()
        casa_ms_table.close()
        self._cal_no_dirs = 0
	self._cal_no_antennae = 0
	self._cal_no_spw = 0
	self._cal_no_channels = 0
	self._cal_no_timestamps = 0
	self._dde_cal_info_desc_exists = False
	self._dde_cal_info_exists = False
        if self._should_read_jones_terms:  
	  try:
	    casa_ms_table = table(self._MSName+"/DDE_CALIBRATION",ack=False,readonly=True)
	    self._dde_cal_info_exists = True
	    self._cal_no_rows = casa_ms_table.nrows()
	    casa_ms_table.close()
	    casa_ms_table = table(self._MSName+"/DDE_CALIBRATION_INFO",ack=False,readonly=True)
	    self._dde_cal_info_desc_exists = True
	    self._cal_no_dirs = casa_ms_table.getcell("DIRECTION_COUNT",0)
	    self._cal_no_antennae = casa_ms_table.getcell("ANTENNA_COUNT",0)
	    self._cal_no_spw = casa_ms_table.getcell("SPW_COUNT",0)
	    self._cal_no_channels = casa_ms_table.getcell("CHANNEL_COUNT",0)
	    self._cal_no_timestamps = casa_ms_table.getcell("TIMESTAMP_COUNT",0)
	    casa_ms_table.close()
	  except:
	    pass
	  if self._dde_cal_info_exists != self._dde_cal_info_desc_exists:
	    raise Exception("Missing one of 'DDE_CALIBRATION' or 'DDE_CALIBRATION_INFO' tables. The following must be valid: both tables exist or neither is present in the MS.") 
	
	  if self._dde_cal_info_desc_exists:
	    if (self._cal_no_antennae != self._no_antennae or self._cal_no_spw != self._no_spw or self._cal_no_channels != self._no_channels or
		self._cal_no_rows != self._cal_no_timestamps*self._cal_no_antennae*self._cal_no_dirs*self._cal_no_spw):
		  raise Exception("Calibration data dimensions does not correspond to measurement set. Ensure calibration data has dimensions [no_timestamps x no_antennae x no_directions x no_spw x no_channel]")
    
    '''
      Read data from the MS
      Arguments:
      start_row moves the reading cursor in the primary table
      no_rows specifies the number of rows to read (-1 == "read all")
      Assumes read_head has been called prior to this call
    '''
    def read_data(self,start_row=0,no_rows=-1,data_column = "DATA",do_romein_baseline_ordering = False):
	print "READING UVW VALUES, DATA, WEIGHTS AND FLAGS"
        casa_ms_table = table(self._MSName,ack=False,readonly=True)        
        time_ordered_ms_table = None
        if self._should_read_jones_terms or do_romein_baseline_ordering:
	  with data_set_loader.time_to_load_chunks:
	    try:
	      chopped_ms_table = taql("SELECT UVW,"+data_column+",WEIGHT_SPECTRUM,FLAG,FLAG_ROW,DATA_DESC_ID,ANTENNA1,ANTENNA2,FIELD_ID,TIME FROM $casa_ms_table LIMIT $no_rows OFFSET $start_row")
	      time_ordered_ms_table = taql("SELECT UVW,"+data_column+",WEIGHT_SPECTRUM,FLAG,FLAG_ROW,DATA_DESC_ID,ANTENNA1,ANTENNA2,FIELD_ID,TIME FROM $chopped_ms_table ORDERBY TIME")
	      c = time_ordered_ms_table.getcol("WEIGHT_SPECTRUM")
	      chopped_ms_table.close()
	    except:
	      chopped_ms_table = taql("SELECT UVW,"+data_column+",WEIGHT,FLAG,FLAG_ROW,DATA_DESC_ID,ANTENNA1,ANTENNA2,FIELD_ID,TIME FROM $casa_ms_table LIMIT $no_rows OFFSET $start_row")
	      time_ordered_ms_table = taql("SELECT UVW,"+data_column+",WEIGHT,FLAG,FLAG_ROW,DATA_DESC_ID,ANTENNA1,ANTENNA2,FIELD_ID,TIME FROM $chopped_ms_table ORDERBY TIME")
	      chopped_ms_table.close()
	else: #since we're not doing jones corrections or gpu imaging let's leave the table unordered (shaves a lot of time off the data loading)
	  with data_set_loader.time_to_load_chunks:
	    try:
	      time_ordered_ms_table = taql("SELECT UVW,"+data_column+",WEIGHT_SPECTRUM,FLAG,FLAG_ROW,DATA_DESC_ID,ANTENNA1,ANTENNA2,FIELD_ID,TIME FROM $casa_ms_table LIMIT $no_rows OFFSET $start_row")
	      c = time_ordered_ms_table.getcol("WEIGHT_SPECTRUM")
	    except:
	      time_ordered_ms_table = taql("SELECT UVW,"+data_column+",WEIGHT,FLAG,FLAG_ROW,DATA_DESC_ID,ANTENNA1,ANTENNA2,FIELD_ID,TIME FROM $casa_ms_table LIMIT $no_rows OFFSET $start_row")
	no_rows = casa_ms_table.nrows() if no_rows==-1 else no_rows
        assert(no_rows > 0) #table is the empty set
        '''
        Grab the uvw coordinates (these are not yet measured in terms of wavelength!)
        This should have dimensions [0...time * baseline -1][0...num_channels-1][0...num_correlations-1][3]
        '''
        with data_set_loader.time_to_load_chunks:
	  self._arr_uvw = time_ordered_ms_table.getcol("UVW").astype(base_types.uvw_type,copy = False)
          
        '''
            the data variable has dimensions: [0...obs_time_range*baselines-1][0...num_channels-1][0...num_correlations-1] 
        '''
        with data_set_loader.time_to_load_chunks:
	  self._arr_data = time_ordered_ms_table.getcol(data_column).astype(base_types.visibility_type)
        '''
            the weights column has dimensions: [0...obs_time_range*baselines-1][0...num_correlations-1]
            However this column only contains the averages accross all channels. The weight_spectrum column
            is actually preferred and has dimensions of [0...obs_time_range*baselines-1][0...channels-1][0...num_correlations-1]
            
            If weight_spectrum is not available then each average should be duplicated accross all the channels. 
            This column can be used to store the estimated rms thermal noise, tapering, sampling weight (uniform, normal, etc.) and
            possibly and prior knowledge about the antenna gains. See for instance the imaging chapter of Synthesis Imaging II.
            
            The jones terms (Smirnov I, 2011) should probably not be added into the weight_spectrum array because the matricies **won't**
            necessarily commute to form something like: 
            sum_over_all_sources(Tapering_weights*sampling_weights*RMS_est*Jones_DDE(time,channel,baseline,direction)*V(time,channel,baseline,direction)) 
            as we would expect the weighting to be applied. Instead we expect the weighted visibilty to be corrected for using something like 
            this in normal imaging (without faceting):
            sum_over_all_sources dde_p^-1 * V_weighted * (dde_q^H)^-1
            With faceting the directional dependent effects can move out of the all-sky integral (Smirnov II, 2011)
        '''
	try:
	  with data_set_loader.time_to_load_chunks:
	    self._arr_weights = time_ordered_ms_table.getcol("WEIGHT_SPECTRUM").astype(base_types.weight_type,copy = False)
	  print "THIS MEASUREMENT SET HAS VISIBILITY WEIGHTS PER CHANNEL, LOADING [WEIGHT_SPECTRUM] INSTEAD OF [WEIGHT]" 
	except:
	  print "WARNING: THIS MEASUREMENT SET ONLY HAS AVERAGED VISIBILITY WEIGHTS (PER BASELINE), LOADING [WEIGHT]"
	  with data_set_loader.time_to_load_chunks:
	    self._arr_weights = np.zeros([no_rows,self._no_channels,self._no_polarization_correlations]).astype(base_types.weight_type,copy = False)
        
	    self._arr_weights[:,0:1,:] = time_ordered_ms_table.getcol("WEIGHT").reshape([no_rows,1,self._no_polarization_correlations])
        
	    for c in range(1,self._no_channels):
	      self._arr_weights[:,c:c+1,:] = self._arr_weights[:,0:1,:] #apply the same average to each channel per baseline

        '''
            the flag column has dimensions: [0...obs_time_range*baselines-1][0...num_channels-1][0...num_correlations-1]
            of type boolean
        '''
        with data_set_loader.time_to_load_chunks:
	  self._arr_flaged = time_ordered_ms_table.getcol("FLAG")
        '''
	    the flag row column has dimensions [0...obs_time_range*baselines-1] and must be taken into account
	    even though there is a more fine-grained flag column
        '''
        with data_set_loader.time_to_load_chunks:
	  self._arr_flagged_rows = time_ordered_ms_table.getcol("FLAG_ROW")
        '''
	  Grab the DATA_DESCRIPTION_ID column (which will describe which reference frequency (per baseline to use)
        '''
        with data_set_loader.time_to_load_chunks:
	  self._description_col = time_ordered_ms_table.getcol("DATA_DESC_ID")
        '''
	  Grab the two antenna id arrays specifying the two antennas defining each baseline (in uvw space)
	'''
	with data_set_loader.time_to_load_chunks:
	  self._arr_antenna_1 = time_ordered_ms_table.getcol("ANTENNA1")
	  self._arr_antenna_2 = time_ordered_ms_table.getcol("ANTENNA2")
	'''
	  Grab the FIELD_ID column in case there is more than a single pointing
	'''
	with data_set_loader.time_to_load_chunks:
	  self._row_field_id = time_ordered_ms_table.getcol("FIELD_ID")
        
	'''
	  Compute how many timeslices this slice of the MAIN table actually contains
	  This cannot be predicted without breaking general MS support, because
	  the number of baselines per timeslice can be anywhere in 0 <= x < no_baselines
	'''
	with data_set_loader.time_to_load_chunks:
	  time_window_centre = time_ordered_ms_table.getcol("TIME")
	casa_ms_table.close()
	time_ordered_ms_table.close()
	with data_set_loader.time_to_load_chunks:
	  self._time_indicies = np.zeros(no_rows,dtype=np.intp)
	  self._baseline_counts = np.zeros(self._no_baselines,dtype=np.intp)
	
	'''
	  Compute a timestep index array from the given integration centres:
	'''
	if self._should_read_jones_terms:
	  timestamp_index = 0 #this index always refers to the index numbers of the timesteps in the current data slice
	  try:
	    self._last_timestamp_time
	    self._last_starting_point
	    if time_window_centre[0] == self._last_timestamp_time: 
	      print "PREVIOUS SLICE DID NOT READ TIMESTAMP COMPLETELY, CONTINUEING FROM PREVIOUS TIMESTAMP... "
	  except:
	    self._last_timestamp_time = time_window_centre[0] #first slice being read, set equal to the first window centre
	    self._last_starting_point = 0
	  
	  '''
	  Now generate timestamp index array
	  '''
	  with data_set_loader.time_to_load_chunks:
	    epsilon = 0.00000001
	    for r in range(0,no_rows):
	      current_time = time_window_centre[r]
	      if current_time - self._last_timestamp_time > epsilon:
		timestamp_index+=1
		self._last_timestamp_time = current_time
	      #else is not possible because we've explicitly sorted the measurement set with the TAQL statement above (when jones inversion is enabled)
	   
	      self._time_indicies[r] = timestamp_index
	    self._no_timestamps_read = timestamp_index + 1  #the last timestamp may be partially read depending on memory constraints
	
	'''
	Read set of jones matricies from disk
	'''
	if self._should_read_jones_terms:
	  try:
	    casa_ms_table = table(self._MSName+"/DDE_CALIBRATION",ack=False,readonly=True)
	    jones_timestamp_size = self._no_antennae*self._cal_no_dirs*self._no_spw
	    print "READING JONES TERMS FROM TIMESTAMP %d TO %d" % (self._last_starting_point,self._last_starting_point+self._no_timestamps_read-1)
	    with data_set_loader.time_to_load_chunks:
	      self._jones_terms = casa_ms_table.getcol("JONES",startrow=self._last_starting_point*jones_timestamp_size,nrow=self._no_timestamps_read*jones_timestamp_size).astype(base_types.visibility_type,copy = False)
	    casa_ms_table.close()
	  except:
	    print "WARNING: MEASUREMENT SET DOES NOT CONTAIN OPTIONAL SUBTABLE 'DDE_CALIBRATION'"
	    print traceback.format_exc()
	  self._last_starting_point += timestamp_index #finally set the last timestamp read, this will be the starting point in the DDE_CALIBRATION table
      
