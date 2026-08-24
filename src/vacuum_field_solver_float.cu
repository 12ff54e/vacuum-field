// vacuum_field_solver_float.cu — explicit float instantiation of the
// vacuum-field driver.
#include "kernels/vacuum_field_solver_impl.cuh"

template class vfield::VacuumFieldSolver<float>;
