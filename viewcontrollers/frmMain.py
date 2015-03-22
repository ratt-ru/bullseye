import sys
import os
import traceback
from pyrap.tables import table
from pyrap.measures import measures
from pyrap.quanta import quantity
from pyrap import quanta
import math
import time
import datetime
import pylab
import numpy as np
import cairo
from gi.repository import Gtk
from gi.repository import Gdk
from multiprocessing import Process

from helpers import data_set_loader
from viewcontrollers import frmFacetDisplay
class frmMain:
	IMAGE_TMP_FILE_NAME = "/tmp/bullseye_temp"
	FACET_TMP_FILE_NAME = "/tmp/bullseye_facet_temp"
	
	def __change_visibilities(self):
		if self._low_res_image != None:
			self._builder.get_object("tvwLowResProperties").set_sensitive(True)
			self._builder.get_object("btnMakeLowRes").set_sensitive(True)
			self._builder.get_object("tvwFacetingProperties").set_sensitive(True)	
		else:
			self._builder.get_object("tvwLowResProperties").set_sensitive(False)
			self._builder.get_object("btnMakeLowRes").set_sensitive(False)
			self._builder.get_object("tvwFacetingProperties").set_sensitive(False)

	def __reset_view(self):
		self._ms_name = None
		self._data_set = None
		self._low_res_image = None
		self.__change_visibilities()

	def on_btnMakeLowRes_clicked(self,widget):
		model = self._builder.get_object("lstLowResProperties")
		itr = model.get_iter(0)
		self._img_size_l = int(model.get_value(itr,1))
		itr = model.iter_next(itr)
		self._img_size_m = int(model.get_value(itr,1))
		itr = model.iter_next(itr)
		self._img_cell_l = float(model.get_value(itr,1))
		itr = model.iter_next(itr)
		self._img_cell_m = float(model.get_value(itr,1))
		itr = model.iter_next(itr)
		conv_support = int(model.get_value(itr,1))
		itr = model.iter_next(itr)
		conv_oversample = int(model.get_value(itr,1))
		itr = model.iter_next(itr)
		self._polarization = model.get_value(itr,1)
		itr = model.iter_next(itr)
		self._field_id = int(model.get_value(itr,1))
		if (not (os.system("python bullseye.py \"%s\" --output_prefix \"%s\" --output_format png --npix_l %d --npix_m %d --cell_l %d --cell_m %d"
				   " --pol %s --conv_sup %d --conv_oversamp %d --field_id %d  --average_all 1" % (self._ms_name,
				   self.IMAGE_TMP_FILE_NAME,self._img_size_l,self._img_size_m,self._img_cell_l,self._img_cell_m,
				   self._polarization,conv_support,conv_oversample,self._field_id)) == 0)):
		    raise Exception("Invalid parameters for imager")
		self._low_res_image = cairo.ImageSurface.create_from_png(self.IMAGE_TMP_FILE_NAME+".png")
		self._builder.get_object("cvsLowRes").queue_draw()
		self.__change_visibilities()
		
	def on_cvsLowRes_draw(self,widget,cr):
		cairo.Context.set_source_rgb(cr, 0, 0, 0);
		if self._low_res_image != None:
			rect = self._builder.get_object("cvsLowRes").get_allocation()
			img_height = self._low_res_image.get_height()
			img_width = self._low_res_image.get_width()
			cr.scale(rect.width/float(img_width),rect.height/float(img_height))
			cr.set_source_surface(self._low_res_image,0,0)	
			cr.paint()
			
			
			model = self._builder.get_object("lstFacetingProperties")
			itr = model.get_iter(0)
			facet_size_l = int(model.get_value(itr,1))
			itr = model.iter_next(itr)
			facet_size_m = int(model.get_value(itr,1))
			itr = model.iter_next(itr)
			facet_cell_l = float(model.get_value(itr,1))
			itr = model.iter_next(itr)
			facet_cell_m = float(model.get_value(itr,1))
		
			selection_box_height = int(self._img_cell_l / facet_cell_l * facet_size_l)
			selection_box_width = int(self._img_cell_m / facet_cell_m * facet_size_m)
		
			cairo.Context.set_source_rgb(cr, 0, 0.75, 0)
			cairo.Context.set_line_width(cr, 5)
			cairo.Context.move_to(cr, 0, 0)
			cairo.Context.rectangle(cr, self._l_pos-(selection_box_width//2), 
						self._m_pos-(selection_box_height//2), 
						selection_box_width, selection_box_height)
			cairo.Context.stroke_preserve(cr)
		else:	
			text = "Create low resolution image first before faceting"
			cairo.Context.select_font_face (cr, "Sans",cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
			
			cairo.Context.set_font_size (cr, 12.0)
			(x_bearing,y_bearing,width,height,x_advance,y_advance) = cairo.Context.text_extents(cr,text)
			rect = self._builder.get_object("cvsLowRes").get_allocation()
			x = rect.width/2-(width/2 + x_bearing)
			y = rect.height/2-(height/2 + y_bearing)
			cairo.Context.move_to(cr, x, y)
			cairo.Context.show_text (cr,text)
		
	def on_mitExit_click(self,widget):
		Gtk.main_quit()

	def on_mitOpen_click(self,widget):
		dialog = Gtk.FileChooserDialog("Open Measurement Set",self._builder.get_object("frmMain"),Gtk.FileChooserAction.SELECT_FOLDER,
					       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK))
		filter = Gtk.FileFilter()
		filter.set_name("Measurement Set")
		filter.add_mime_type("ms")
		filter.add_pattern("*")
		dialog.add_filter(filter)
		response = dialog.run()
		if response == Gtk.ResponseType.OK:
			try:
			  self.__reset_view()
			  self._ms_name = dialog.get_filename()
			  self.on_btnMakeLowRes_clicked(self._builder.get_object("btnMakeLowRes"))
			  casa_ms_table = table(self._ms_name+"/FIELD",ack=False,readonly=True)
			  self._phase_centres = casa_ms_table.getcol("REFERENCE_DIR")
			  casa_ms_table.close()
			  self.__change_visibilities() 
			except:
			  message = Gtk.MessageDialog(self._builder.get_object("frmMain"),0,Gtk.MessageType.ERROR,Gtk.ButtonsType.OK,"Not a valid MS 2.0 Measurement Set directory!\n"+traceback.format_exc())
			  message.run()
			  message.destroy()
			  self.__reset_view()
		dialog.destroy()

	def on_mitExport_click(self,widget):
		print "EXPORT STUB"

	def on_tvwLowResProperties_cell_edited(self,cell,path_string,new_text):
		model = self._builder.get_object("lstLowResProperties")
		it = model.get_iter(path_string)
		model.set_value(it, 1, new_text)
	
	def on_tvwFacetingProperties_cell_edited(self,cell,path_string,new_text):
		model = self._builder.get_object("lstFacetingProperties")
                it = model.get_iter(path_string)
                model.set_value(it, 1, new_text)

	def on_click(self,widget,event):
		if self._low_res_image != None:
			model = self._builder.get_object("lstFacetingProperties")
			itr = model.get_iter(0)
			facet_size_l = int(model.get_value(itr,1))
			itr = model.iter_next(itr)
			facet_size_m = int(model.get_value(itr,1))
			itr = model.iter_next(itr)
			facet_cell_l = float(model.get_value(itr,1))
			itr = model.iter_next(itr)
			facet_cell_m = float(model.get_value(itr,1))
			itr = model.iter_next(itr)
			conv_support = int(model.get_value(itr,1))
			itr = model.iter_next(itr)
			conv_oversample = int(model.get_value(itr,1))
			
			rect = self._builder.get_object("cvsLowRes").get_allocation()
                        img_height = self._low_res_image.get_height()
                        img_width = self._low_res_image.get_width()
                        facet_ra = quantity(self._phase_centres[self._field_id,0,0],"rad").get_value("arcsec") + (-int(event.x/float(rect.width) * img_width) + img_width/2)*self._img_cell_l
                        facet_dec = quantity(self._phase_centres[self._field_id,0,1],"rad").get_value("arcsec") + (-int(event.y/float(rect.height) * img_height) + img_height/2)*self._img_cell_m	
			facet_centres = np.array([[facet_ra,facet_dec]],dtype=np.float32)
			
			if (not (os.system("python bullseye.py \"%s\" --output_prefix \"%s\" --output_format png --npix_l %d --npix_m %d --cell_l %d"
					   " --cell_m %d --pol %s --conv_sup %d --conv_oversamp %d --facet_centres \(%f,%f\) --field_id %d --average_all 1" % (
					    self._ms_name,self.FACET_TMP_FILE_NAME,facet_size_l,facet_size_m,facet_cell_l,
					    facet_cell_m,self._polarization,conv_support,conv_oversample,
					    int(facet_ra),int(facet_dec),self._field_id)) == 0)):
			  raise Exception("Invalid parameters for imager")
			
			frmFacetDisplay.frmFacetDisplay(self.FACET_TMP_FILE_NAME+"_facet0.png")
			

	def on_mouse_move(self,widget,event):
		if self._low_res_image != None: 
			sbrMain = self._builder.get_object("sbrMain")
			sbrMain.remove_all(sbrMain.get_context_id("CursorPos"))
			handle = self._builder.get_object("cvsLowRes")
			handle.queue_draw()
			rect = handle.get_allocation()
                        img_height = self._low_res_image.get_height()
                        img_width = self._low_res_image.get_width()
                        
                        self._l_pos = int(event.x/float(rect.width) * img_width) 
			self._m_pos = int(event.y/float(rect.height) * img_height) 
			
			sbrMain.push(sbrMain.get_context_id("CursorPos"),"Pixel position: (l,m) = (%d,%d), (ra,dec) = (%s,%s)" % (self._l_pos, self._m_pos,
							    		 quantity(quantity(self._phase_centres[self._field_id,0,0],"rad").get_value("arcsec") + (-self._l_pos + img_width/2)*self._img_cell_l,"arcsec").get("deg").formatted("[+-]dd.mm.ss.t.."),
							    		 quantity(quantity(self._phase_centres[self._field_id,0,1],"rad").get_value("arcsec") + (-self._m_pos + img_height/2)*self._img_cell_m,"arcsec").get("deg").formatted("[+-]dd.mm.ss.t..")))
	def __init__(self):
		self._builder = Gtk.Builder()
		self._builder.add_from_file("viewcontrollers/main.glade")
		self._builder.connect_signals(self)
		self._builder.get_object("frmMain").connect("destroy", Gtk.main_quit)
		self._builder.get_object("frmMain").show_all()
		self._builder.get_object("cvsLowRes").add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.BUTTON_RELEASE_MASK | Gdk.EventMask.POINTER_MOTION_MASK)
		self._data_set = None
		self._low_res_image = None
		self.__change_visibilities()
		handle = self._builder.get_object("cvsLowRes")
		rect = handle.get_allocation()
		self._l_pos = rect.width // 2
		self._m_pos = rect.height // 2