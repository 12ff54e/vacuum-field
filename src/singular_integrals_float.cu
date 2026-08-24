// singular_integrals_float.cu — explicit float instantiation of the
// singular integrals operator.
#include "kernels/singular_integrals_impl.cuh"

template class vfield::SingularIntegralsOperator<float>;
