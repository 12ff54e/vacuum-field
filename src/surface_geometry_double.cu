// surface_geometry_double.cu — explicit double instantiation of the surface
// geometry operator.
#include "kernels/surface_geometry_impl.cuh"

template class vfield::SurfaceGeometryOperator<double>;
