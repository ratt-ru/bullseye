#!/usr/bin/python

from gi.repository import Gtk
from viewcontrollers import frmMain

if __name__ == "__main__":
  wnd = frmMain.frmMain()
  Gtk.main()