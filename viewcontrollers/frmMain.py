import sys
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

from models import data_set_loader
from models import convolution_filter
from models import fft_utils
sys.path.append("build/algorithms")
import libimaging

class frmMain:
	IMAGE_TMP_FILE_NAME = "/tmp/bullseye_temp.png"
	
	def __change_visibilities(self):
		if self._data_set != None:
			self._builder.get_object("tvwLowResProperties").set_sensitive(True)
			self._builder.get_object("btnMakeLowRes").set_sensitive(True)
			if self._low_res_image != None:
				self._builder.get_object("tvwFacetingProperties").set_sensitive(True)
			else:
				self._builder.get_object("tvwFacetingProperties").set_sensitive(False)
		else:
			self._builder.get_object("tvwLowResProperties").set_sensitive(False)
			self._builder.get_object("btnMakeLowRes").set_sensitive(False)
			self._builder.get_object("tvwFacetingProperties").set_sensitive(False)

	def __reset_view(self):
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
		self._img_cell_l = model.get_value(itr,1)
		itr = model.iter_next(itr)
		self._img_cell_m = model.get_value(itr,1)
		itr = model.iter_next(itr)
		conv_support = int(model.get_value(itr,1))
		itr = model.iter_next(itr)
		conv_oversample = int(model.get_value(itr,1))
		itr = model.iter_next(itr)
		self._polarization = int(model.get_value(itr,1))

		pol_label = ["XX","XY","YX","YY"]
		conv = convolution_filter.convolution_filter(conv_support,conv_support,conv_oversample,self._img_size_l,self._img_size_m)
		print("GRIDDING POLARIZATION %s..." % pol_label[self._polarization]),
                g = libimaging.grid(self._data_set._arr_data,self._data_set._arr_uvw,
                                    conv._conv_FIR.astype(np.float32),conv_support,conv_oversample,
                                    self._data_set._no_timestamps,self._data_set._no_baselines,self._data_set._no_channels,self._data_set._no_polarization_correlations,
                                    self._polarization,self._data_set._chan_wavelengths,self._data_set._arr_flaged,self._data_set._arr_flagged_rows,self._data_set._arr_weights,
                                    self._data_set._phase_centre[0,0],self._data_set._phase_centre[0,1],
                                    None,self._img_size_l,self._img_size_m,self._img_cell_l,self._img_cell_m)
                print " <DONE>"
                g = np.real(fft_utils.ifft2(g[0,:,:]))/conv._F_detaper
                i = pylab.imshow(g,interpolation="nearest",cmap = pylab.get_cmap('hot'),extent=[0, self._img_size_l-1, 0, self._img_size_m-1])
		pylab.close("all")
		i.write_png(self.IMAGE_TMP_FILE_NAME,noscale=True)		
		self._low_res_image = cairo.ImageSurface.create_from_png(self.IMAGE_TMP_FILE_NAME)
	 	self._builder.get_object("cvsLowRes").queue_draw()
		self.__change_visibilities()
		
		pylab.figure()
		pylab.title("CONVOLUTION F")
		pylab.imshow(conv._conv_FIR,cmap=pylab.get_cmap("hot"))
		pylab.colorbar()
		pylab.figure()
		pylab.title("DETAPERER F")
		pylab.imshow(conv._F_detaper,cmap=pylab.get_cmap("hot"))
		pylab.colorbar()
		pylab.show(block=True)
		

	def on_cvsLowRes_draw(self,widget,cr):
		if self._low_res_image != None:
			rect = self._builder.get_object("cvsLowRes").get_allocation()
			img_height = self._low_res_image.get_height()
			img_width = self._low_res_image.get_width()
			cr.scale(rect.width/float(img_width),rect.height/float(img_height))
			cr.set_source_surface(self._low_res_image,0,0)	
			cr.paint()
		else:	
			text = "Create low resolution image first before faceting"
			cairo.Context.select_font_face (cr, "Sans",cairo.FONT_SLANT_NORMAL,cairo.FONT_WEIGHT_NORMAL)
			
			cairo.Context.set_font_size (cr, 12.0)
			(x_bearing,y_bearing,width,height,x_advance,y_advance) = cairo.Context.text_extents(cr,text)
			rect = self._builder.get_object("cvsLowRes").get_allocation()
			x = rect.width/2-(width/2 + x_bearing);
			y = rect.height/2-(height/2 + y_bearing);
			cairo.Context.move_to(cr, x, y);
			cairo.Context.show_text (cr,text);

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
			  msname = dialog.get_filename()
			  self._data_set = data_set_loader.data_set_loader(msname)
			  self.__change_visibilities() 
			except:
			  message = Gtk.MessageDialog(self._builder.get_object("frmMain"),0,Gtk.MessageType.ERROR,Gtk.ButtonsType.OK,"Not a valid MS 2.0 Measurement Set directory!")
			  message.run()
			  message.destroy()
			  self.__reset_view()
			
		dialog.destroy()

	def on_mitExport_click(self,widget):
		print "EXPORT STUB"

	def on_tvwLowResProperties_cell_edited(self,cell,path_string,new_text):
		model = self._builder.get_object("lstLowResProperties")
		it = model.get_iter(path_string)
		val = math.floor(float(new_text)) if int(path_string) <= 1 or int(path_string) > 3 else float(new_text)
		if path_string == "6":
			val = max(0,min(3,val))
		model.set_value(it, 1, val)
	
	def on_tvwFacetingProperties_cell_edited(self,cell,path_string,new_text):
		model = self._builder.get_object("lstFacetingProperties")
                it = model.get_iter(path_string)
                val = math.floor(float(new_text)) if int(path_string) <= 1 or int(path_string) > 3 else float(new_text)
                model.set_value(it, 1, val)

	def on_click(self,widget,event):
		if self._low_res_image != None:
			model = self._builder.get_object("lstFacetingProperties")
			itr = model.get_iter(0)
			facet_size_l = int(model.get_value(itr,1))
			itr = model.iter_next(itr)
			facet_size_m = int(model.get_value(itr,1))
			itr = model.iter_next(itr)
			facet_cell_l = model.get_value(itr,1)
			itr = model.iter_next(itr)
			facet_cell_m = model.get_value(itr,1)
			itr = model.iter_next(itr)
			conv_support = int(model.get_value(itr,1))
			itr = model.iter_next(itr)
			conv_oversample = int(model.get_value(itr,1))
			
			rect = self._builder.get_object("cvsLowRes").get_allocation()
                        img_height = self._low_res_image.get_height()
                        img_width = self._low_res_image.get_width()
                        facet_ra = self._data_set._phase_centre[0,0] + (-int(event.x/float(rect.width) * img_width) + img_width/2)*self._img_cell_l
                        facet_dec = self._data_set._phase_centre[0,1] + (-int(event.y/float(rect.height) * img_height) + img_height/2)*self._img_cell_m
				
			facet_centres = np.array([[facet_ra,facet_dec]],dtype=np.float32)
			pol_label = ["XX","XY","YX","YY"]
			conv = convolution_filter.convolution_filter(conv_support,conv_support,conv_oversample,facet_size_l,facet_size_m)
                	g = libimaging.grid(self._data_set._arr_data,self._data_set._arr_uvw,
                                    conv._conv_FIR.astype(np.float32),conv_support,conv_oversample,
                                    self._data_set._no_timestamps,self._data_set._no_baselines,self._data_set._no_channels,self._data_set._no_polarization_correlations,
                                    self._polarization,self._data_set._chan_wavelengths,self._data_set._arr_flaged,self._data_set._arr_weights,
                                    self._data_set._phase_centre[0,0],self._data_set._phase_centre[0,1],
                                    facet_centres,facet_size_l,facet_size_m,facet_cell_l,facet_cell_m)
                	g = fft_utils.ifft2(g[0,:,:])/conv._F_detaper
			
	        	pylab.figure()
			pylab.imshow(np.real(g),interpolation="nearest",cmap = pylab.get_cmap('hot'),extent=[0, facet_size_l-1, 0, facet_size_m-1])
			pylab.colorbar()
			pylab.show(block=True)

	def on_mouse_move(self,widget,event):
		if self._low_res_image != None: 
			sbrMain = self._builder.get_object("sbrMain")
			sbrMain.remove_all(sbrMain.get_context_id("CursorPos"))
			
			rect = self._builder.get_object("cvsLowRes").get_allocation()
                        img_height = self._low_res_image.get_height()
                        img_width = self._low_res_image.get_width()
                        l_pos = int(event.x/float(rect.width) * img_width) 
			m_pos = int(event.y/float(rect.height) * img_height) 
			
			sbrMain.push(sbrMain.get_context_id("CursorPos"),"Pixel position: (l,m) = (%d,%d), (ra,dec) = (%s,%s)" % (l_pos, m_pos,
							    		 quantity(self._data_set._phase_centre[0,0] + (l_pos - img_width/2)*self._img_cell_l,"arcsec").get("deg").formatted("[+-]dd.mm.ss.t.."),
							    		 quantity(self._data_set._phase_centre[0,1] + (-m_pos + img_height/2)*self._img_cell_m,"arcsec").get("deg").formatted("[+-]dd.mm.ss.t..")))
				

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
