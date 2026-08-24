// external_field_float.cu — explicit float instantiation of the external
// field operator.
#include "kernels/external_field_impl.cuh"

template class vfield::ExternalFieldOperator<float>;
