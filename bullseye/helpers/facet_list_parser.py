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

import bullseye_mo.base_types as base_types
from pyrap.quanta import quantity
import numpy as np
import sys
import os
from os.path import *
import shutil

def compute_number_of_facet_centres(parser_args):
  num_facet_centres = parser_args['n_facets_l'] * parser_args['n_facets_m']
  if (parser_args['facet_centres'] != None):
    num_facet_centres += len(parser_args['facet_centres'])
  return num_facet_centres

def create_facet_centre_list(parser_args,data,num_facet_centres):
  #remember fastest changing to the right (C ordering):l,m,n is written n,m,l
  facet_centres = np.empty([parser_args['n_facets_m'],parser_args['n_facets_l'],2],dtype=base_types.uvw_type)
  offset_l = np.ceil(parser_args['n_facets_l']/2.0)
  offset_m = np.ceil(parser_args['n_facets_m']/2.0)
  field_centre_l = data._field_centres[parser_args['field_id'],0,0]
  field_centre_m = data._field_centres[parser_args['field_id'],0,1]
  size_in_arcsec_l = parser_args['npix_l']*parser_args['cell_l']
  size_in_arcsec_m = parser_args['npix_m']*parser_args['cell_m']
  range_of_l_coords = (np.arange(1,parser_args['n_facets_l']+1) - offset_l)*size_in_arcsec_l + field_centre_l
  range_of_m_coords = (np.arange(1,parser_args['n_facets_m']+1) - offset_m)*size_in_arcsec_m + field_centre_m
  #each element in the range of l coordinates repeat m times in the l dim
  facet_centres[:parser_args['n_facets_m'],:parser_args['n_facets_l'],0] = np.repeat(range_of_l_coords,parser_args['n_facets_m']).reshape(parser_args['n_facets_m'],parser_args['n_facets_l'])
  #the range of m coordinates repeat l times in the m dim
  facet_centres[:parser_args['n_facets_m'],:parser_args['n_facets_l'],1] = np.tile(range_of_m_coords,parser_args['n_facets_l']).reshape(parser_args['n_facets_m'],parser_args['n_facets_l'])

  #create a flat list of coordinates
  facet_centres = facet_centres.reshape(parser_args['n_facets_l']*parser_args['n_facets_m'],2)
  if (parser_args['facet_centres'] != None):
      facet_centres = np.append(facet_centres,np.array(parser_args['facet_centres']).astype(base_types.uvw_type)).reshape(num_facet_centres,2)
  return facet_centres

def print_facet_centre_list(facet_centres,num_facet_centres):
  if num_facet_centres != 0:
      print "REQUESTED FACET CENTRES:"
      for i,c in enumerate(facet_centres):
	print "\tFACET %d RA: %s DEC: %s" % (i,quantity(c[0],'arcsec').get('deg'),quantity(c[1],'arcsec').get('deg'))

'''
This uses Montage:
Montage: a grid portal and software toolkit for science-grade astronomical image mosaicking
Jacob, Katz, Berriman, Good, Laity, Deelman, Kasselman, Singh, Su Prince, Williams
Int. J. Computational Science and Engineering (2006)
'''
def output_mosaic(output_prefix,num_facet_centres):
  file_names = [basename(output_prefix + '_facet' + str(i) + '.fits') for i in range(0,num_facet_centres)]
  facet_image_list_filename = dirname(output_prefix) + '/facets.lst'
  f_file_list = open(facet_image_list_filename,'w')
  f_file_list.writelines(["|%sfname|\n" % (" "*(max([len(item) for item in file_names])-5)),
			  "|%schar|\n" % (" "*(max([len(item) for item in file_names])-4))
			 ])
  f_file_list.writelines([" %s\n" % item for item in file_names])
  f_file_list.close()
  #Closely follow Montage's recipy for creating a background-corrected mosaic
  #http://montage.ipac.caltech.edu/docs/how.html
  montage_unprojected_img_table = dirname(output_prefix) + '/facets.montage.tbl'
  os.system('mImgtbl -t %s %s %s' % (facet_image_list_filename,
				     dirname(output_prefix),
				     montage_unprojected_img_table
				    )
	   )
  montage_proj_template_hdr = dirname(output_prefix) + '/projected_template.hdr'
  os.system('mMakeHdr %s %s' % (montage_unprojected_img_table,
				montage_proj_template_hdr
			       )
	   )
  proj_dir = dirname(output_prefix) + '/projected_facets'
  if exists(proj_dir):
    shutil.rmtree(proj_dir)
  os.makedirs(proj_dir)
  montage_stats_file = dirname(output_prefix) + '/stats.tbl'
  os.system('mProjExec -p %s %s %s %s %s' % (dirname(output_prefix),
					     montage_unprojected_img_table,
					     montage_proj_template_hdr,
					     proj_dir,
					     montage_stats_file
					    )
	   )
  montage_projected_img_table = dirname(output_prefix) + '/facets.montage.proj.tbl'
  os.system('mImgtbl %s %s' % (proj_dir,
			       montage_projected_img_table
			      )
	   )
  montage_diffs_table = dirname(output_prefix) + '/overlap.tbl'
  os.system('mOverlaps -e %s %s' % (montage_projected_img_table,
				 montage_diffs_table
				)
	   )
  diffs_dir = dirname(output_prefix) + '/overlaps'
  if exists(diffs_dir):
    shutil.rmtree(diffs_dir)
  os.makedirs(diffs_dir)
  os.system('mDiffExec -p %s %s %s %s' % (proj_dir,
					  montage_diffs_table,
					  montage_proj_template_hdr,
					  diffs_dir
					 )
	   )
  montage_fit_table = dirname(output_prefix) + '/fit.tbl'
  os.system('mFitExec %s %s %s' % (montage_diffs_table,
				   montage_fit_table,
				   diffs_dir,
				   )
	   )
  montage_corrections_table = dirname(output_prefix) + '/combine.tbl'
  os.system('mBgModel %s %s %s' % (montage_projected_img_table,
				   montage_fit_table,
				   montage_corrections_table
				  )
	   )
  corrections_dir = dirname(output_prefix) + '/corrections'
  if exists(corrections_dir):
    shutil.rmtree(corrections_dir)
  os.makedirs(corrections_dir)
  os.system('mBgExec -p %s %s %s %s' % (proj_dir,
					montage_projected_img_table,
					montage_corrections_table,
					corrections_dir
				       )
	   )
  montage_combined_img = output_prefix + '.combined.fits'
  os.system('mAdd -a mean -p %s %s %s %s' % (corrections_dir,
					     montage_projected_img_table,
					     montage_proj_template_hdr,
					     montage_combined_img
					    )
	   )
