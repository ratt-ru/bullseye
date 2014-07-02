from gi.repository import Gtk
from gi.repository import Gdk
class frmFacetDisplay:
  def __init__(self,image_name):
	self._builder = Gtk.Builder()
	self._builder.add_from_file("viewcontrollers/facetDisplay.glade")
	self._builder.connect_signals(self)
	self._builder.get_object("frmFacetDisplay").show_all()
	self._builder.get_object("imgFacetDisplay").set_from_file(image_name)
	self._builder.get_object("imgFacetDisplay").show()