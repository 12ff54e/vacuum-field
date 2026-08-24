// singular_integrals_double.cu — explicit double instantiation of the
// singular integrals operator.
#include "kernels/singular_integrals_impl.cuh"

template class vfield::SingularIntegralsOperator<double>;
