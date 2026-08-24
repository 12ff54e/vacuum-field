// laplace_solver_float.cu — explicit float instantiation of the Laplace
// solver operator.
#include "kernels/laplace_solver_impl.cuh"

template class vfield::LaplaceSolverOperator<float>;
