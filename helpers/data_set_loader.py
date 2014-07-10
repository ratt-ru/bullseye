'''
Created on Mar 20, 2014

@author: bhugo
'''
import numpy as np
from pyrap.tables import table
from pyrap.measures import measures
from pyrap.quanta import quantity
from pyrap import quanta
import math
import time
import datetime
import pylab

class data_set_loader(object):
    '''
    classdocs
    '''
    def __init__(self, MSName, data_column="DATA",load_only_header=False):
        '''
        Constructor
        '''
        print "READING MEASUREMENT SET '%s'" % MSName 
        self._MSName = MSName
        self.__read_head()
        if not load_only_header:
	  print "READING UVW VALUES AND DATA[TIME][POLARIZATION_CORRELATION][FREQUENCY]"
	  self.__read_data(data_column)
         
    '''
        Read some stats about the MS
        Assume this method only gets called from __init__
        (This can be checked against listobs in CASAPY)
    '''
    def __read_head(self):
        casa_ms_table = table(self._MSName+'/OBSERVATION',ack=False,readonly=True)
        self._observer_name = casa_ms_table.getcell("OBSERVER", 0)
        self._telescope_name = casa_ms_table.getcell("TELESCOPE_NAME", 0)
        self._observation_start = casa_ms_table.getcell("TIME_RANGE", 0)[0]
        self._observation_end = casa_ms_table.getcell("TIME_RANGE", 0)[1]
        casa_ms_table.close()
        casa_ms_table = table(self._MSName+"/POINTING",ack=False,readonly=True)
        self._epoch = casa_ms_table.getcolkeyword("DIRECTION","MEASINFO")["Ref"]
        casa_ms_table.close()
        fmt = '%Y-%m-%d %H:%M:%S %Z'
        print "OBSERVED BY %s ON %s FROM %s TO %s SINCE EPOCH %s" % (self._observer_name,self._telescope_name,
                                                      self._observation_start,
                                                      self._observation_end,
                                                      self._epoch,
                                                     )
        
        
        casa_ms_table.close()
        casa_ms_table = table(self._MSName+'/ANTENNA',ack=False,readonly=True)
        self._no_antennae = casa_ms_table.nrows()
        self._no_baselines = self._no_antennae*(self._no_antennae-1)/2 + self._no_antennae
        print "FOUND %d ANTENNAE:" % (self._no_antennae) 
        for i in range(0,self._no_antennae):
            name = casa_ms_table.getcell("NAME", i)
            position = casa_ms_table.getcell("POSITION", i)
            print "\t %s has position [%f , %f , %f]" % (name,position[0],position[1],position[2])
        casa_ms_table.close()
        print "%d UNIQUE BASELINES" % (self._no_baselines)
        casa_ms_table = table(self._MSName+"/POLARIZATION",ack=False,readonly=True)
        self._no_polarization_correlations = casa_ms_table.getcell("NUM_CORR", 0)
        print "%d CORRELATIONS DUE TO POLARIZATION" % self._no_polarization_correlations
        casa_ms_table.close()
        casa_ms_table = table(self._MSName+"/FEED",ack=False,readonly=True)
        self._no_receptors = casa_ms_table.getcell("NUM_RECEPTORS", 0)
        assert(self._no_polarization_correlations == self._no_receptors**2) #number of ways to correlate two feeds
        self._polarization_type = casa_ms_table.getcell("POLARIZATION_TYPE", 0) #should be something like [X,Y] or [R,L]
        casa_ms_table.close()
        casa_ms_table = table(self._MSName+"/SPECTRAL_WINDOW",ack=False,readonly=True)
        self._no_channels = casa_ms_table.getcell("NUM_CHAN", 0)
        print "%d CHANNELS IN OBSERVATION" % self._no_channels
        self._chan_freqs = casa_ms_table.getcell("CHAN_FREQ",0)
        self._chan_wavelengths = (quanta.constants['c'].get_value("m/s") /self._chan_freqs).astype(np.float32) # lambda = speed of light / frequency
        for i,lamb in enumerate(self._chan_wavelengths):
            print "\t Channel %d has a wavelength of %f m" % (i,lamb)
        self._ref_frequency = casa_ms_table.getcell("REF_FREQUENCY", 0)
        self._ref_wavelength = quanta.constants['c'].get_value("m/s") / float(self._ref_frequency) # lambda = speed of light / frequency
        print "REFERENCE WAVELENGTH: %f m" % (self._ref_wavelength)
        casa_ms_table.close()
        casa_ms_table = table(self._MSName+"/FIELD",ack=False,readonly=True)
        self._phase_centre = casa_ms_table.getcell("PHASE_DIR", 0)
	self._phase_centre[0,0] = quantity(self._phase_centre[0,0],"rad").get_value("arcsec")
	self._phase_centre[0,1] = quantity(self._phase_centre[0,1],"rad").get_value("arcsec")
	print "PHASE CENTRE: (RA: %s, DEC: %s)" % (quantity(self._phase_centre[0,0],"arcsec").get("deg").formatted("[+-]dd.mm.ss.t.."),
						   quantity(self._phase_centre[0,1],"arcsec").get("deg").formatted("[+-]dd.mm.ss.t.."))
        casa_ms_table.close()
        casa_ms_table = table(self._MSName,ack=False,readonly=True)
        self._no_timestamps = casa_ms_table.nrows() / self._no_baselines
        print "%d TIMESTAMPS IN OBSERVATION" % self._no_timestamps
        casa_ms_table.close()
    '''
        Read data from the MS
        Assume this method only gets called from __init__
    '''
    def __read_data(self,data_column):
        casa_ms_table = table(self._MSName,ack=False,readonly=True)
        '''
        Grab the uvw coordinates (these are not yet measured in terms of wavelength!)
        '''
        self._arr_uvw = casa_ms_table.getcol("UVW").astype(np.float32)
        '''
        self._min_u = min(self._arr_uvw,key=lambda p: p[0])[0]
        self._max_u = max(self._arr_uvw,key=lambda p: p[0])[0]
        self._min_v = min(self._arr_uvw,key=lambda p: p[1])[1]
        self._max_v = max(self._arr_uvw,key=lambda p: p[1])[1]
        self._min_w = min(self._arr_uvw,key=lambda p: p[2])[2]
        self._max_w = max(self._arr_uvw,key=lambda p: p[2])[2]
        '''
        
        '''
            the data variable has dimensions: [0...obs_time_range*baselines-1][0...num_channels-1][0...num_correlations-1] 
        '''
        self._arr_data = casa_ms_table.getcol(data_column).astype(np.complex64)
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
	if "WEIGHT_SPECTRUM" in casa_ms_table.colnames():
	  self._arr_weights = casa_ms_table.getcol("WEIGHT_SPECTRUM").astype(np.float32)
	  print "THIS MEASUREMENT SET HAS VISIBILITY WEIGHTS PER CHANNEL, LOADING [WEIGHT_SPECTRUM] INSTEAD OF [WEIGHT]" 
	else:
	  print "THIS MEASUREMENT SET ONLY HAS AVERAGED VISIBILITY WEIGHTS (PER BASELINE), LOADING [WEIGHT]"
	  self._arr_weights = np.zeros([self._no_baselines*self._no_timestamps,self._no_channels,self._no_polarization_correlations]).astype(np.float32)
        
	  self._arr_weights[:,0:1,:] = casa_ms_table.getcol("WEIGHT").reshape([self._no_baselines*self._no_timestamps,1,self._no_polarization_correlations])
        
	  for c in range(1,self._no_channels):
	    self._arr_weights[:,c:c+1,:] = self._arr_weights[:,0:1,:] #apply the same average to each channel per baseline

        '''
            the flag column has dimensions: [0...obs_time_range*baselines-1][0...num_channels-1][0...num_correlations-1]
            of type boolean
        '''
        self._arr_flaged = casa_ms_table.getcol("FLAG")
        
        '''
	    the flag row column has dimensions [0...obs_time_range*baselines-1] and must be taken into account
	    even though there is a more fine-grained flag column
        '''
        self._arr_flagged_rows = casa_ms_table.getcol("FLAG_ROW")
        
        '''
        Grab the two antenna id arrays defining the two antennas defining each baseline (in uvw space)
	'''
	self._arr_antenna_1 = casa_ms_table.getcol("ANTENNA1")
	self._arr_antenna_1 = casa_ms_table.getcol("ANTENNA2")
        casa_ms_table.close()
        '''
        print "MIN UVW = (%f,%f,%f), MAX UVW = (%f,%f,%f)" % (self._min_u,self._min_v,self._min_w,self._max_u,self._max_v,self._max_w)
	'''