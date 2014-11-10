'''
Created on Mar 20, 2014

@author: bhugo
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
import base_types
import traceback
from timer import timer
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
      Computes the number of rows to read, given memory constraints (in bytes)
      Assumes read_head has been called prior to this call
    '''
    def compute_number_of_rows_to_read_from_mem_requirements(self,max_bytes_available):
	return int(max_bytes_available / ((3*8) + #uvw data
				      (self._no_channels*self._no_polarization_correlations*2*8) + #complex visibilities
				      (self._no_channels*self._no_polarization_correlations*8) + #weight
				      (self._no_channels*self._no_polarization_correlations*np.dtype(np.bool_).itemsize) + #visibility flags
				      (np.dtype(np.bool_).itemsize) + #row flags
				      (4*np.dtype(np.intc).itemsize) + #antenna 1 & 2, FIELD_ID and DATA_DESCRIPTION_ID
				      (np.dtype(np.float64).itemsize) + #timestamp window centre (MAIN/TIME)
				      (1*np.dtype(np.intp).itemsize) + #timestamp id (computed when data is read)
				      (self._no_antennae*self._cal_no_dirs*self._no_spw*self._no_channels*4*8*2/float(self._no_baselines)) #average jones contibution per row if all baselines are present per timestamp
				     ) / 2.0) #we're using double the amount of memory in order to buffer IO while doing compute
    '''
      Computes the number of iterations required to read entire file, given memory constraints (in bytes)
      Assumes read_head has been called prior to this call
    '''
    def number_of_read_iterations_required_from_mem_requirements(self,max_bytes_available):
	return int(math.ceil(self._no_rows / float(self.compute_number_of_rows_to_read_from_mem_requirements(max_bytes_available))))
    
    '''
      Read data from the MS
      Arguements:
      start_row moves the reading cursor in the primary table
      no_rows specifies the number of rows to read (-1 == "read all")
      Assumes read_head has been called prior to this call
    '''
    def read_data(self,start_row=0,no_rows=-1,data_column = "DATA"):
	print "READING UVW VALUES, DATA, WEIGHTS AND FLAGS"
        casa_ms_table = table(self._MSName,ack=False,readonly=True)
        time_ordered_ms_table = None
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
	  for r in range(0,no_rows):
	    current_time = time_window_centre[r]
	    epsilon = 0.00000001
	    if current_time - self._last_timestamp_time > epsilon:
	      timestamp_index+=1
	      self._last_timestamp_time = current_time
	    #else is not possible because we've explicitly sorted the measurement set with the TAQL statement above
	   
	    self._time_indicies[r] = timestamp_index
	  self._no_timestamps_read = timestamp_index + 1  #the last timestamp may be partially read depending on memory constraints
	
	'''
	Romein's gridding algorithm requires ordered baselines. We'll therefore count the number of timestamps per baseline
	(some baselines may not be present in one or more timestamps) and compute prescan of this count in order to get the starting
	positions per baseline.
	'''
	def baseline_index(a1,a2):
		slow_changing_antenna_index = min(self._arr_antenna_1[r],self._arr_antenna_2[r]) + 1
		#the unique index per baseline is given by a quadratic series on the slow-varying index plus the fast varying index...
		baseline_flat_index = (slow_changing_antenna_index*(-slow_changing_antenna_index + (2*self._no_antennae + 3)) - 2 * (self._no_antennae + 1)) // 2 + abs(self._arr_antenna_2[r] - self._arr_antenna_1[r])
		return baseline_flat_index

	self._baseline_timestamp_count = np.zeros([self._no_baselines],dtype=np.intp)
	current_baseline_timestamp_index = np.zeros([self._no_baselines],dtype=np.intp) 
	with data_set_loader.time_to_load_chunks:
		for r in range(0,no_rows): #bin the data according to index
			bi = baseline_index(self._arr_antenna_1[r],self._arr_antenna_2[r])
			self._baseline_timestamp_count[bi] += 1
		self._starting_indexes = np.cumsum(self._baseline_timestamp_count) - self._baseline_timestamp_count[0] #we want the prescan operator here
	
	tmp_uvw = np.zeros([no_rows,3],dtype=base_types.uvw_type)
	tmp_data = np.empty([no_rows,self._no_channels,self._no_polarization_correlations],dtype=base_types.visibility_type)
	tmp_weights = np.empty([no_rows,self._no_channels,self._no_polarization_correlations],dtype=base_types.weight_type)
	tmp_flags = np.empty([no_rows,self._no_channels,self._no_polarization_correlations],dtype=np.bool_)
	tmp_flag_rows = np.empty([no_rows],dtype=np.bool_)
	tmp_data_desc = np.empty([no_rows],dtype=np.int_)
	tmp_ant_1 = np.empty([no_rows],dtype=np.int_)
	tmp_ant_2 = np.empty([no_rows],dtype=np.int_)
	tmp_field = np.empty([no_rows],dtype=np.int_)
	tmp_time = np.empty([no_rows],dtype=np.intp)
	with data_set_loader.time_to_load_chunks:
	  for r in range(0,no_rows):
			bi = baseline_index(self._arr_antenna_1[r],self._arr_antenna_2[r])
			rearanged_index = current_baseline_timestamp_index[bi] + self._starting_indexes[bi]
			current_baseline_timestamp_index[bi] += 1
			tmp_uvw[rearanged_index] = self._arr_uvw[r]
			tmp_data[rearanged_index] = self._arr_data[r]
			tmp_weights[rearanged_index] = self._arr_weights[r]
			tmp_flags[rearanged_index] = self._arr_flaged[r]
			tmp_flag_rows[rearanged_index] = self._arr_flagged_rows[r]
			tmp_data_desc[rearanged_index] = self._description_col[r]
			tmp_ant_1[rearanged_index] = self._arr_antenna_1[r]
			tmp_ant_2[rearanged_index] = self._arr_antenna_2[r]
			tmp_field[rearanged_index] = self._row_field_id[r]
			tmp_time[rearanged_index] = self._time_indicies[r]
	self._arr_uvw = tmp_uvw
	self._arr_data = tmp_data
	self._arr_weights = tmp_weights
	self._arr_flaged = tmp_flags
	self._arr_flagged_rows = tmp_flag_rows
	self._description_col = tmp_data_desc
	self._arr_antenna_1 = tmp_ant_1
	self._arr_antenna_2 = tmp_ant_2
	self._row_field_id = tmp_field
	self._time_indicies = tmp_time
  
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
      
