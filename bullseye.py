#!/usr/bin/python
import matplotlib
import pylab
pylab.ion() #don't start closing other GTK windows!!
from gi.repository import Gtk
from viewcontrollers import frmMain

if __name__ == "__main__":
	wnd = frmMain.frmMain()
	Gtk.main()
		
