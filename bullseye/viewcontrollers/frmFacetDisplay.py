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

from gi.repository import Gtk
from gi.repository import Gdk
import os
import bullseye
BULLSEYE_PATH = os.path.dirname(bullseye.__file__)

class frmFacetDisplay:
  def __init__(self,image_name):
	self._builder = Gtk.Builder()
	self._builder.add_from_file(os.path.join(BULLSEYE_PATH, "viewcontrollers/facetDisplay.glade"))
	self._builder.connect_signals(self)
	self._builder.get_object("frmFacetDisplay").show_all()
	self._builder.get_object("imgFacetDisplay").set_from_file(image_name)
	self._builder.get_object("imgFacetDisplay").show()