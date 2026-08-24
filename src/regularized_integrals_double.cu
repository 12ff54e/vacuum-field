// regularized_integrals_double.cu — explicit double instantiation of the
// regularized integrals operator.
#include "kernels/regularized_integrals_impl.cuh"

template class vfield::RegularizedIntegralsOperator<double>;
