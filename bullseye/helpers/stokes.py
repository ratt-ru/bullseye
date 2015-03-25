pol_options = {'I' : 1, 'Q' : 2, 'U' : 3, 'V' : 4, 'RR' : 5, 'RL' : 6, 'LR' : 7, 'LL' : 8, 'XX' : 9, 'XY' : 10, 'YX' : 11, 'YY' : 12} # as per Stokes.h in casacore, the rest is left unimplemented
'''
See Smirnov I (2011) for description on conversion between correlation terms and stokes params for linearly polarized feeds
See Synthesis Imaging II (1999) Pg. 9 for a description on conversion between correlation terms and stokes params for circularly polarized feeds
'''
pol_dependencies = {
    'I'  : [[pol_options['I']],[pol_options['RR'],pol_options['LL']],[pol_options['XX'],pol_options['YY']]],
    'V'  : [[pol_options['V']],[pol_options['RR'],pol_options['LL']],[pol_options['XY'],pol_options['YX']]],
    'U'  : [[pol_options['U']],[pol_options['RL'],pol_options['LR']],[pol_options['XY'],pol_options['YX']]],
    'Q'  : [[pol_options['Q']],[pol_options['RL'],pol_options['LR']],[pol_options['XX'],pol_options['YY']]],
    'RR' : [[pol_options['RR']]],
    'RL' : [[pol_options['RL']]],
    'LR' : [[pol_options['LR']]],
    'LL' : [[pol_options['LL']]],
    'XX' : [[pol_options['XX']]],
    'XY' : [[pol_options['XY']]],
    'YX' : [[pol_options['YX']]],
    'YY' : [[pol_options['YY']]]
}
feed_descriptions = {
    'I'  : ["stokes","circular","linear"],
    'V'  : ["stokes","circular","linear"],
    'U'  : ["stokes","circular","linear"],
    'Q'  : ["stokes","circular","linear"],
    'RR' : ["circular"],
    'RL' : ["circular"],
    'LR' : ["circular"],
    'LL' : ["circular"],
    'XX' : ["linear"],
    'XY' : ["linear"],
    'YX' : ["linear"],
    'YY' : ["linear"]
}

def find_necessary_correlations_indexes(should_do_jones_corrections,required_polarization,data):
  correlations_to_grid = None
  feeds_in_use = None
  if should_do_jones_corrections:
    correlations_to_grid = data._polarization_correlations.tolist() #check below if this list is supported
  else:
    for feed_index,req in enumerate(pol_dependencies[required_polarization]):
      if set(req).issubset(data._polarization_correlations.tolist()):
	correlations_to_grid = req #we found a subset that must be gridded
	feeds_in_use = feed_descriptions[required_polarization][feed_index]
  return (correlations_to_grid, feeds_in_use)

'''
See Smirnov I (2011) for description on conversion between correlation terms and stokes params for linearly polarized feeds
See Synthesis Imaging II (1999) Pg. 9 for a description on conversion between correlation terms and stokes params for circularly polarized feeds
'''
def create_stokes_term_from_gridded_vis(should_do_jones_corrections,data,correlations_to_grid,feeds_in_use,gridded_vis,wanted_polarization):
  if should_do_jones_corrections:
    if feeds_in_use == 'circular':		#circular correlation products
      if wanted_polarization == "I":
	IpV = data._polarization_correlations.tolist().index(pol_options['RR'])
	ImV = data._polarization_correlations.tolist().index(pol_options['LL'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpV,:,:] + gridded_vis[:,:,ImV,:,:])/2)
      elif wanted_polarization == "V":
	IpV = data._polarization_correlations.tolist().index(pol_options['RR'])
	ImV = data._polarization_correlations.tolist().index(pol_options['LL'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpV,:,:] - gridded_vis[:,:,ImV,:,:])/2)
      elif wanted_polarization == "Q":
	QpiU = data._polarization_correlations.tolist().index(pol_options['RL'])
	QmiU = data._polarization_correlations.tolist().index(pol_options['LR'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,QpiU,:,:] + gridded_vis[:,:,QmiU,:,:])/2)
      elif wanted_polarization == "U":
	QpiU = data._polarization_correlations.tolist().index(pol_options['RL'])
	QmiU = data._polarization_correlations.tolist().index(pol_options['LR'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,QpiU,:,:] - gridded_vis[:,:,QmiU,:,:])/2)
      elif wanted_polarization in ['RR','RL','LR','LL']:
	pass #already stored in gridded_vis[:,0,:,:]
    elif feeds_in_use == 'linear':		#linear correlation products
      if wanted_polarization == "I":
	IpQ = data._polarization_correlations.tolist().index(pol_options['XX'])
	ImQ = data._polarization_correlations.tolist().index(pol_options['YY'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpQ,:,:] + gridded_vis[:,:,ImQ,:,:])/2)
      elif wanted_polarization == "Q":
	IpQ = data._polarization_correlations.tolist().index(pol_options['XX'])
	ImQ = data._polarization_correlations.tolist().index(pol_options['YY'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpQ,:,:] - gridded_vis[:,:,ImQ,:,:])/2)
      elif wanted_polarization == "U":
	UpiV = data._polarization_correlations.tolist().index(pol_options['XY'])
	UmiV = data._polarization_correlations.tolist().index(pol_options['YX'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,UpiV,:,:] + gridded_vis[:,:,UmiV,:,:])/2)
      elif wanted_polarization == "V":
	UpiV = data._polarization_correlations.tolist().index(pol_options['XY'])
	UmiV = data._polarization_correlations.tolist().index(pol_options['YX'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,UpiV,:,:] - gridded_vis[:,:,UmiV,:,:])/2)
      elif wanted_polarization in ['XX','XY','YX','YY']:
	pass #already stored in gridded_vis[:,0,:,:]
    else:
      pass #any cases not stated here should be flagged by sanity checks on the program arguement list
  else:
    if feeds_in_use == 'circular':		#circular correlation products
      if wanted_polarization == "I":
	IpV = correlations_to_grid.index(pol_options['RR'])
	ImV = correlations_to_grid.index(pol_options['LL'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpV,:,:] + gridded_vis[:,:,ImV,:,:])/2)
      elif wanted_polarization == "V":
	IpV = correlations_to_grid.index(pol_options['RR'])
	ImV = correlations_to_grid.index(pol_options['LL'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpV,:,:] - gridded_vis[:,:,ImV,:,:])/2)
      elif wanted_polarization == "Q":
	QpiU = correlations_to_grid.index(pol_options['RL'])
	QmiU = correlations_to_grid.index(pol_options['LR'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,QpiU,:,:] + gridded_vis[:,:,QmiU,:,:])/2)
      elif wanted_polarization == "U":
	QpiU = correlations_to_grid.index(pol_options['RL'])
	QmiU = correlations_to_grid.index(pol_options['LR'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,QpiU,:,:] - gridded_vis[:,:,QmiU,:,:])/2)
      elif wanted_polarization in ['RR','RL','LR','LL']:
	pass #already stored in gridded_vis[:,0,:,:]
    elif feeds_in_use == 'linear':		#linear correlation products
      if wanted_polarization == "I":
	IpQ = correlations_to_grid.index(pol_options['XX'])
	ImQ = correlations_to_grid.index(pol_options['YY'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpQ,:,:] + gridded_vis[:,:,ImQ,:,:])/2)
      elif wanted_polarization == "Q":
	IpQ = correlations_to_grid.index(pol_options['XX'])
	ImQ = correlations_to_grid.index(pol_options['YY'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,IpQ,:,:] - gridded_vis[:,:,ImQ,:,:])/2)
      elif wanted_polarization == "U":
	UpiV = correlations_to_grid.index(pol_options['XY'])
	UmiV = correlations_to_grid.index(pol_options['YX'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,UpiV,:,:] + gridded_vis[:,:,UmiV,:,:])/2)
      elif wanted_polarization == "V":
	UpiV = correlations_to_grid.index(pol_options['XY'])
	UmiV = correlations_to_grid.index(pol_options['YX'])
	gridded_vis[:,:,0,:,:] = ((gridded_vis[:,:,UpiV,:,:] - gridded_vis[:,:,UmiV,:,:])/2)
      elif wanted_polarization in ['XX','XY','YX','YY']:
	pass #already stored in gridded_vis[:,0,:,:]
    else:
      pass #any cases not stated here should be flagged by sanity checks on the program arguement list