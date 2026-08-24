// kernels/external_field_impl.cuh — mgrid interpolation + axis-current kernels.
//
// Included once per scalar type by external_field_double.cu /
// external_field_float.cu; see the explicit-instantiation split. Every output
// element is written by exactly one thread, which replays vmecpp's
// accumulation order (source_index -> kl for the axis filament), so the
// results are summation-order-identical to the CPU reference.
#ifndef VFIELD_SRC_EXTERNAL_FIELD_IMPL_CUH_
#define VFIELD_SRC_EXTERNAL_FIELD_IMPL_CUH_

#include "vfield/free_boundary/external_field_operator.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numbers>
#include <vector>

namespace vfield {

namespace {

constexpr int BLOCK_SIZE = 256;

inline int gridSize(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Checked 1D launch of a kernel taking a single param struct by value.
// cudaLaunchKernel copies the params out of the args array at launch time, so
// a stack array is safe.
template <class KernelParams>
void launchChecked(const void* func,
                   int n,
                   int block,
                   const KernelParams& params,
                   cudaStream_t stream,
                   const char* tag) {
    void* kargs[] = {const_cast<KernelParams*>(&params)};
    check_cuda(cudaLaunchKernel(func, dim3(gridSize(n)), dim3(block), kargs, 0,
                                stream),
               tag);
}

// double -> T conversion for uploads (a raw cudaMemcpy would reinterpret the
// doubles in float builds).
template <class T>
void uploadConverted(const std::vector<double>& src, DeviceBuffer<T>* dst) {
    std::vector<T> converted(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        converted[i] = static_cast<T>(src[i]);
    }
    dst->allocate(src.size());
    dst->upload(converted.data(), converted.size());
}

// Device -> host download into a vector (used by the out-of-bounds warning
// path).
template <class T>
std::vector<T> toHostT(const T* d, std::size_t n) {
    std::vector<T> h(n);
    check_cuda(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost),
               "toHostT");
    return h;
}

}  // namespace

template <class T>
struct ExternalKernelParams {
    // Surface geometry.
    const T* r1b;
    const T* z1b;
    const T* rub;
    const T* rvb;
    const T* zub;
    const T* zvb;
    const T* snr;
    const T* snv;
    const T* snz;
    const T* cosPhi;
    const T* sinPhi;
    // mgrid grid + metadata.
    const T* bR;
    const T* bP;
    const T* bZ;
    const T* fixedBr;
    const T* fixedBp;
    const T* fixedBz;
    double minR;
    double maxR;
    double deltaR;
    double minZ;
    double maxZ;
    double deltaZ;
    int numR;
    int numZ;
    int numPhi;
    bool hasFixedField;
    // Axis geometry + current.
    const T* rAxis;
    const T* zAxis;
    T* axisXyz;
    T netToroidalCurrent;
    // Sizes.
    int nZeta;
    int nvper;
    int nZnT;
    // Outputs.
    T* interpBr;
    T* interpBp;
    T* interpBz;
    T* curtorBr;
    T* curtorBp;
    T* curtorBz;
    T* bSubU;
    T* bSubV;
    T* bDotN;
    int* outOfBounds;
};

// Interpolate the mgrid field at the LCFS points (bilinear in R,Z at fixed
// phi; points outside the grid are cropped and flagged).
template <class T>
__global__ void mgridInterpKernel(ExternalKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;

    if (p.hasFixedField) {
        // quick return: just copy into target storage
        p.interpBr[kl] = p.fixedBr[kl];
        p.interpBp[kl] = p.fixedBp[kl];
        p.interpBz[kl] = p.fixedBz[kl];
        return;
    }

    const int k = kl % p.nZeta;

    // check if plasma boundary exceeds pre-computed grid
    if (p.r1b[kl] < T(p.minR) || p.r1b[kl] > T(p.maxR) ||
        p.z1b[kl] < T(p.minZ) || p.z1b[kl] > T(p.maxZ)) {
        atomicOr(p.outOfBounds, 1);
    }

    // crop to available grid
    T r = max(T(p.minR), min(p.r1b[kl], T(p.maxR)));
    T z = max(T(p.minZ), min(p.z1b[kl], T(p.maxZ)));

    // DETERMINE INTEGER INDICES (IR,JZ) FOR LOWER LEFT R, Z CORNER GRID POINT
    int ir = int(floor((r - T(p.minR)) / T(p.deltaR)));
    int jz = int(floor((z - T(p.minZ)) / T(p.deltaZ)));
    int ir1 = min(p.numR - 1, ir + 1);
    int jz1 = min(p.numZ - 1, jz + 1);

    // COMPUTE RI, ZJ AND PR, QZ AT GRID POINT (IR , JZ)
    T ri = T(p.minR) + ir * T(p.deltaR);
    T zj = T(p.minZ) + jz * T(p.deltaZ);
    T pr = (r - ri) / T(p.deltaR);
    T qz = (z - zj) / T(p.deltaZ);

    // COMPUTE WEIGHTS WIJ FOR 4 CORNER GRID POINTS
    T w22 = pr * qz;              //    p *   q
    T w21 = pr - w22;             //    p *(1-q) = p - p*q
    T w12 = qz - w22;             // (1-p)*   q  = q - p*q
    T w11 = 1 + w22 - (pr + qz);  // (1-p)*(1-q) = 1 + p*q - (p + q)

    // COMPUTE B FIELD AT R, PHI, Z BY INTERPOLATION
    int kj_i_ = (k * p.numZ + jz) * p.numR + ir;
    int kj1i_ = (k * p.numZ + jz1) * p.numR + ir;
    int kj_i1 = (k * p.numZ + jz) * p.numR + ir1;
    int kj1i1 = (k * p.numZ + jz1) * p.numR + ir1;

    p.interpBr[kl] = w11 * p.bR[kj_i_] + w12 * p.bR[kj1i_] + w21 * p.bR[kj_i1] +
                     w22 * p.bR[kj1i1];
    p.interpBp[kl] = w11 * p.bP[kj_i_] + w12 * p.bP[kj1i_] + w21 * p.bP[kj_i1] +
                     w22 * p.bP[kj1i1];
    p.interpBz[kl] = w11 * p.bZ[kj_i_] + w12 * p.bZ[kj1i_] + w21 * p.bZ[kj_i1] +
                     w22 * p.bZ[kj1i1];
}

// Build the axis polygon in Cartesian coordinates: the axis of the first
// field period rotated into the other toroidal replications, closed. The
// closure entry equals the first point and is written directly (no
// cross-thread read of axisXyz[0..2]).
template <class T>
__global__ void axisXyzKernel(ExternalKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nAxis = p.nZeta * p.nvper;
    if (idx > nAxis) return;

    if (idx == nAxis) {
        // close the loop: the first point is (rAxis[0], 0, zAxis[0]) since
        // cos_phi[0] == 1 and sin_phi[0] == 0.
        p.axisXyz[3 * nAxis + 0] = p.rAxis[0];
        p.axisXyz[3 * nAxis + 1] = 0;
        p.axisXyz[3 * nAxis + 2] = p.zAxis[0];
        return;
    }

    const int k = idx % p.nZeta;
    const T x0 = p.rAxis[k] * p.cosPhi[k];
    const T y0 = p.rAxis[k] * p.sinPhi[k];

    if (idx < p.nZeta) {
        // first field-period module
        p.axisXyz[3 * idx + 0] = x0;
        p.axisXyz[3 * idx + 1] = y0;
        p.axisXyz[3 * idx + 2] = p.zAxis[k];
        return;
    }

    // rotated into the other toroidal replications
    const int per = idx / p.nZeta;
    const T omega_per = T(2.0 * std::numbers::pi) / T(p.nvper);
    const T cos_per = cos(omega_per * per);
    const T sin_per = sin(omega_per * per);

    p.axisXyz[3 * idx + 0] = cos_per * x0 - sin_per * y0;
    p.axisXyz[3 * idx + 1] = sin_per * x0 + cos_per * y0;
    p.axisXyz[3 * idx + 2] = p.zAxis[k];
}

// Magnetic field of the net toroidal current along the magnetic axis
// (Hanson-Hirshman polygon-filament Biot-Savart, vmecpp's
// AddAxisCurrentFieldSimple), converted to cylindrical components.
template <class T>
__global__ void axisCurrentKernel(ExternalKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;
    const int k = kl % p.nZeta;

    const T surface_x = p.r1b[kl] * p.cosPhi[k];
    const T surface_y = p.r1b[kl] * p.sinPhi[k];
    const T surface_z = p.z1b[kl];

    // 1.0e-7 == mu0/4 pi
    // NOTE: The factor of 2 comes from the Hanson-Hirshman Biot-Savart
    // formula, which is Eqn. (8) in Hanson & Hirshman (2002)
    // [Physics of Plasmas 9, 4410].
    const T magnetic_field_scale = T(1.0e-7) * p.netToroidalCurrent * T(2.0);

    // Accumulate the segment contributions in vmecpp's order; registers start
    // at zero, so the bCoilsXYZ.setZero() has no kernel equivalent.
    T b_x = 0;
    T b_y = 0;
    T b_z = 0;
    for (int source_index = 0; source_index < p.nZeta * p.nvper;
         ++source_index) {
        const T segment_dx = p.axisXyz[(source_index + 1) * 3 + 0] -
                             p.axisXyz[source_index * 3 + 0];
        const T segment_dy = p.axisXyz[(source_index + 1) * 3 + 1] -
                             p.axisXyz[source_index * 3 + 1];
        const T segment_dz = p.axisXyz[(source_index + 1) * 3 + 2] -
                             p.axisXyz[source_index * 3 + 2];

        const T segment_length =
            sqrt(segment_dx * segment_dx + segment_dy * segment_dy +
                 segment_dz * segment_dz);

        const T r_i_x = surface_x - p.axisXyz[source_index * 3 + 0];
        const T r_i_y = surface_y - p.axisXyz[source_index * 3 + 1];
        const T r_i_z = surface_z - p.axisXyz[source_index * 3 + 2];
        const T r_i = sqrt(r_i_x * r_i_x + r_i_y * r_i_y + r_i_z * r_i_z);

        const T r_f_x = surface_x - p.axisXyz[(source_index + 1) * 3 + 0];
        const T r_f_y = surface_y - p.axisXyz[(source_index + 1) * 3 + 1];
        const T r_f_z = surface_z - p.axisXyz[(source_index + 1) * 3 + 2];
        const T r_f = sqrt(r_f_x * r_f_x + r_f_y * r_f_y + r_f_z * r_f_z);

        const T r_i_plus_r_f = r_i + r_f;

        const T magnetic_field_magnitude =
            magnetic_field_scale * r_i_plus_r_f /
            (r_i * r_f *
             (r_i_plus_r_f * r_i_plus_r_f - segment_length * segment_length));

        // cross product of L*hat(eps)==dvec with Ri_vec,
        // scaled by magnetic field magnitude
        b_x += magnetic_field_magnitude *
               (segment_dy * r_i_z - segment_dz * r_i_y);
        b_y += magnetic_field_magnitude *
               (segment_dz * r_i_x - segment_dx * r_i_z);
        b_z += magnetic_field_magnitude *
               (segment_dx * r_i_y - segment_dy * r_i_x);
    }  // source_index

    // transform bCoilsXYZ into cylindrical coordinates
    p.curtorBr[kl] = p.cosPhi[k] * b_x + p.sinPhi[k] * b_y;
    p.curtorBp[kl] = p.cosPhi[k] * b_y - p.sinPhi[k] * b_x;
    p.curtorBz[kl] = b_z;
}

// Compute bSubU, bSubV: covariant components of external magnetic field
// and bDotN: normal component of external magnetic field.
template <class T>
__global__ void covariantAndNormalKernel(ExternalKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;

    // add contributions together
    // --> helps in debugging to have them separate until here
    T fullBr = p.interpBr[kl] + p.curtorBr[kl];
    T fullBp = p.interpBp[kl] + p.curtorBp[kl];
    T fullBz = p.interpBz[kl] + p.curtorBz[kl];

    // covariant components
    p.bSubU[kl] = fullBr * p.rub[kl] + fullBz * p.zub[kl];
    p.bSubV[kl] = fullBr * p.rvb[kl] + fullBz * p.zvb[kl] + fullBp * p.r1b[kl];

    // normal component
    p.bDotN[kl] =
        -(fullBr * p.snr[kl] + fullBp * p.snv[kl] + fullBz * p.snz[kl]);
}

template <class T>
ExternalFieldOperator<T>::ExternalFieldOperator(
    const Sizes& sizes,
    const SurfaceGeometryOperator<T>& sg,
    const MgridProvider& mgrid)
    : sizes_(sizes), sg_(sg), stream_(nullptr) {
    if (!mgrid.isLoaded()) {
        throw VfieldError("ExternalFieldOperator: no mgrid loaded");
    }

    nvper_ = (sizes_.nZeta == 1) ? 64 : sizes_.nfp;

    min_r_ = mgrid.minR;
    max_r_ = mgrid.maxR;
    delta_r_ = mgrid.deltaR;
    min_z_ = mgrid.minZ;
    max_z_ = mgrid.maxZ;
    delta_z_ = mgrid.deltaZ;
    num_r_ = mgrid.numR;
    num_z_ = mgrid.numZ;
    num_phi_ = mgrid.numPhi;
    has_fixed_field_ = mgrid.hasFixedField();

    if (has_fixed_field_) {
        uploadConverted(mgrid.fixedBr(), &fixed_br_);
        uploadConverted(mgrid.fixedBp(), &fixed_bp_);
        uploadConverted(mgrid.fixedBz(), &fixed_bz_);
    } else {
        uploadConverted(mgrid.bR, &b_r_);
        uploadConverted(mgrid.bP, &b_p_);
        uploadConverted(mgrid.bZ, &b_z_);
    }

    axis_xyz_.allocate(3 * (sizes_.nZeta * nvper_ + 1));

    interp_br_.allocate(sizes_.nZnT);
    interp_bp_.allocate(sizes_.nZnT);
    interp_bz_.allocate(sizes_.nZnT);
    curtor_br_.allocate(sizes_.nZnT);
    curtor_bp_.allocate(sizes_.nZnT);
    curtor_bz_.allocate(sizes_.nZnT);
    bsubu_.allocate(sizes_.nZnT);
    bsubv_.allocate(sizes_.nZnT);
    bdotn_.allocate(sizes_.nZnT);
    out_of_bounds_.allocate(1);
    out_of_bounds_.zero();
}

template <class T>
void ExternalFieldOperator<T>::update(const T* d_r_axis,
                                      const T* d_z_axis,
                                      T net_toroidal_current) {
    ExternalKernelParams<T> p{};
    p.r1b = sg_.r1b();
    p.z1b = sg_.z1b();
    p.rub = sg_.rub();
    p.rvb = sg_.rvb();
    p.zub = sg_.zub();
    p.zvb = sg_.zvb();
    p.snr = sg_.snr();
    p.snv = sg_.snv();
    p.snz = sg_.snz();
    p.cosPhi = sg_.cosPhi();
    p.sinPhi = sg_.sinPhi();
    p.bR = b_r_.data();
    p.bP = b_p_.data();
    p.bZ = b_z_.data();
    p.fixedBr = fixed_br_.data();
    p.fixedBp = fixed_bp_.data();
    p.fixedBz = fixed_bz_.data();
    p.minR = min_r_;
    p.maxR = max_r_;
    p.deltaR = delta_r_;
    p.minZ = min_z_;
    p.maxZ = max_z_;
    p.deltaZ = delta_z_;
    p.numR = num_r_;
    p.numZ = num_z_;
    p.numPhi = num_phi_;
    p.hasFixedField = has_fixed_field_;
    p.rAxis = d_r_axis;
    p.zAxis = d_z_axis;
    p.axisXyz = axis_xyz_.data();
    p.netToroidalCurrent = net_toroidal_current;
    p.nZeta = sizes_.nZeta;
    p.nvper = nvper_;
    p.nZnT = sizes_.nZnT;
    p.interpBr = interp_br_.data();
    p.interpBp = interp_bp_.data();
    p.interpBz = interp_bz_.data();
    p.curtorBr = curtor_br_.data();
    p.curtorBp = curtor_bp_.data();
    p.curtorBz = curtor_bz_.data();
    p.bSubU = bsubu_.data();
    p.bSubV = bsubv_.data();
    p.bDotN = bdotn_.data();
    p.outOfBounds = out_of_bounds_.data();

    check_cuda(cudaMemsetAsync(p.outOfBounds, 0, sizeof(int), stream_),
               "ExternalFieldOperator::update: flag reset");
    launchChecked(reinterpret_cast<const void*>(&mgridInterpKernel<T>),
                  sizes_.nZnT, BLOCK_SIZE, p, stream_,
                  "ExternalFieldOperator::update: mgrid interpolation");
    launchChecked(reinterpret_cast<const void*>(&axisXyzKernel<T>),
                  sizes_.nZeta * nvper_ + 1, BLOCK_SIZE, p, stream_,
                  "ExternalFieldOperator::update: axis polygon");
    launchChecked(reinterpret_cast<const void*>(&axisCurrentKernel<T>),
                  sizes_.nZnT, BLOCK_SIZE, p, stream_,
                  "ExternalFieldOperator::update: axis current");
    launchChecked(reinterpret_cast<const void*>(&covariantAndNormalKernel<T>),
                  sizes_.nZnT, BLOCK_SIZE, p, stream_,
                  "ExternalFieldOperator::update: covariant/normal");

    // Out-of-bounds check for the grid-cropping warning (the flag copy
    // synchronizes the stream; the warning fires only in pathological cases).
    int out_of_bounds = 0;
    out_of_bounds_.download(&out_of_bounds, 1);
    if (out_of_bounds != 0 && !has_fixed_field_) {
        // Recompute the raw min/max of the surface points for the message.
        std::vector<T> r = toHostT(sg_.r1b(), sizes_.nZnT);
        std::vector<T> z = toHostT(sg_.z1b(), sizes_.nZnT);
        double min_r = r[0], max_r = r[0], min_z = z[0], max_z = z[0];
        for (int kl = 1; kl < sizes_.nZnT; ++kl) {
            min_r = std::min(min_r, static_cast<double>(r[kl]));
            max_r = std::max(max_r, static_cast<double>(r[kl]));
            min_z = std::min(min_z, static_cast<double>(z[kl]));
            max_z = std::max(max_z, static_cast<double>(z[kl]));
        }
        std::cerr << "WARNING: Plasma Boundary exceeded Vacuum Grid Size\n";
        if (min_r < min_r_ || max_r > max_r_) {
            std::cerr << "  R: min = " << min_r << "  max = " << max_r << "\n";
        }
        if (min_z < min_z_ || max_z > max_z_) {
            std::cerr << "  Z: min = " << min_z << "  max = " << max_z << "\n";
        }
    }
}

}  // namespace vfield

#endif  // VFIELD_SRC_EXTERNAL_FIELD_IMPL_CUH_
