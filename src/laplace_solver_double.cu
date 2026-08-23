// laplace_solver_double.cu — explicit double instantiation of the Laplace
// solver operator.
#include "laplace_solver_impl.cuh"

template class vfield::LaplaceSolverOperator<double>;
