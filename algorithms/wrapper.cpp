#include "gridding.h"
#include <string>
static PyMethodDef libimaging_methods[] = {
	{"grid", grid, METH_VARARGS, "Grid visibilities."},
	{nullptr,nullptr,0,nullptr}
};
PyMODINIT_FUNC initlibimaging(){
        PyObject *m;
        m = Py_InitModule("libimaging",libimaging_methods);
	if (m == nullptr) 
	      return;
	gridding_error = PyErr_NewException((char*)std::string("gridding.error").c_str(),nullptr,nullptr);
	Py_INCREF(gridding_error);
	PyModule_AddObject(m,"error",gridding_error);
        import_array();  // Must be present for NumPy.  Called first after above line.
}
