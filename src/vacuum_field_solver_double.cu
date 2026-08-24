// vacuum_field_solver_double.cu — explicit double instantiation of the
// vacuum-field driver.
#include "kernels/vacuum_field_solver_impl.cuh"

template class vfield::VacuumFieldSolver<double>;
