// external_field_double.cu — explicit double instantiation of the external
// field operator.
#include "kernels/external_field_impl.cuh"

template class vfield::ExternalFieldOperator<double>;
