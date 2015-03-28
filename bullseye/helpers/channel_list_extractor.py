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

import re
def channel_range(s):
    sT = s.strip()
    if re.match('[0-9]+:[0-9]+(~[0-9]+)?(,[0-9]+(~[0-9]+)?)*$',sT) != None:
	spw,selections = s.split(':')
	ranges = selections.split(',')
	channels = []
	for r in ranges:
	  if re.match('[0-9]+~[0-9]+$',r) != None:
	    start,finish = r.split('~')
	    channels = channels + range(int(start),int(finish)+1)
	  else:
	    channels.append(int(r))
	return int(spw),sorted(set(channels))
    else:
	raise argparse.ArgumentTypeError("Channel ranges should be specified as 'spw index':'comma seperated ranges of channels', for example 0:0,2~5,7 will select channels 0,2,3,4,5,7 from spw 0")