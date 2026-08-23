// vfield_types.h — compile-time scalar switch and shared aliases.
//
// Every computation in the library is template<typename T> (double or float);
// `Real` names the build's preferred instantiation, switched by the
// VFIELD_USE_FLOAT compile definition (set by the build system on the
// selected scalar library). The host dense solve always runs in double.
#ifndef VFIELD_VFIELD_TYPES_H_
#define VFIELD_VFIELD_TYPES_H_

#ifdef VFIELD_USE_FLOAT
using Real = float;
#else
using Real = double;
#endif

#endif  // VFIELD_VFIELD_TYPES_H_
