// surface_geometry_float.cu — explicit float instantiation of the surface
// geometry operator.
#include "kernels/surface_geometry_impl.cuh"

template class vfield::SurfaceGeometryOperator<float>;
