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

inline int grid_size(int n) {
    return (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Checked 1D launch of a kernel taking a single param struct by value.
// cudaLaunchKernel copies the params out of the args array at launch time, so
// a stack array is safe.
template <class KernelParams>
void launch_checked(const void* func,
                   int n,
                   int block,
                   const KernelParams& params,
                   cudaStream_t stream,
                   const char* tag) {
    void* kargs[] = {const_cast<KernelParams*>(&params)};
    check_cuda(cudaLaunchKernel(func, dim3(grid_size(n)), dim3(block), kargs, 0,
                                stream),
               tag);
}

// double -> T conversion for uploads (a raw cudaMemcpy would reinterpret the
// doubles in float builds).
template <class T>
void upload_converted(const std::vector<double>& src, DeviceBuffer<T>* dst) {
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
std::vector<T> to_host_t(const T* d, std::size_t n) {
    std::vector<T> h(n);
    check_cuda(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost),
               "to_host_t");
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
    const T* cos_phi;
    const T* sin_phi;
    // mgrid grid + metadata.
    const T* b_r;
    const T* b_p;
    const T* b_z;
    const T* fixed_br;
    const T* fixed_bp;
    const T* fixed_bz;
    double min_r;
    double max_r;
    double delta_r;
    double min_z;
    double max_z;
    double delta_z;
    int num_r;
    int num_z;
    int num_phi;
    bool has_fixed_field;
    // Axis geometry + current.
    const T* r_axis;
    const T* z_axis;
    T* axis_xyz;
    T net_toroidal_current;
    // Sizes.
    int nZeta;
    int nvper;
    int nZnT;
    // Outputs.
    T* interp_br;
    T* interp_bp;
    T* interp_bz;
    T* curtor_br;
    T* curtor_bp;
    T* curtor_bz;
    T* b_sub_u;
    T* b_sub_v;
    T* b_dot_n;
    int* out_of_bounds;
};

// Interpolate the mgrid field at the LCFS points (bilinear in R,Z at fixed
// phi; points outside the grid are cropped and flagged).
template <class T>
__global__ void mgrid_interp_kernel(ExternalKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;

    if (p.has_fixed_field) {
        // quick return: just copy into target storage
        p.interp_br[kl] = p.fixed_br[kl];
        p.interp_bp[kl] = p.fixed_bp[kl];
        p.interp_bz[kl] = p.fixed_bz[kl];
        return;
    }

    const int k = kl % p.nZeta;

    // check if plasma boundary exceeds pre-computed grid
    if (p.r1b[kl] < T(p.min_r) || p.r1b[kl] > T(p.max_r) ||
        p.z1b[kl] < T(p.min_z) || p.z1b[kl] > T(p.max_z)) {
        atomicOr(p.out_of_bounds, 1);
    }

    // crop to available grid
    T r = max(T(p.min_r), min(p.r1b[kl], T(p.max_r)));
    T z = max(T(p.min_z), min(p.z1b[kl], T(p.max_z)));

    // DETERMINE INTEGER INDICES (IR,JZ) FOR LOWER LEFT R, Z CORNER GRID POINT
    int ir = int(floor((r - T(p.min_r)) / T(p.delta_r)));
    int jz = int(floor((z - T(p.min_z)) / T(p.delta_z)));
    int ir1 = min(p.num_r - 1, ir + 1);
    int jz1 = min(p.num_z - 1, jz + 1);

    // COMPUTE RI, ZJ AND PR, QZ AT GRID POINT (IR , JZ)
    T ri = T(p.min_r) + ir * T(p.delta_r);
    T zj = T(p.min_z) + jz * T(p.delta_z);
    T pr = (r - ri) / T(p.delta_r);
    T qz = (z - zj) / T(p.delta_z);

    // COMPUTE WEIGHTS WIJ FOR 4 CORNER GRID POINTS
    T w22 = pr * qz;              //    p *   q
    T w21 = pr - w22;             //    p *(1-q) = p - p*q
    T w12 = qz - w22;             // (1-p)*   q  = q - p*q
    T w11 = 1 + w22 - (pr + qz);  // (1-p)*(1-q) = 1 + p*q - (p + q)

    // COMPUTE B FIELD AT R, PHI, Z BY INTERPOLATION
    int kj_i_ = (k * p.num_z + jz) * p.num_r + ir;
    int kj1i_ = (k * p.num_z + jz1) * p.num_r + ir;
    int kj_i1 = (k * p.num_z + jz) * p.num_r + ir1;
    int kj1i1 = (k * p.num_z + jz1) * p.num_r + ir1;

    p.interp_br[kl] = w11 * p.b_r[kj_i_] + w12 * p.b_r[kj1i_] + w21 * p.b_r[kj_i1] +
                     w22 * p.b_r[kj1i1];
    p.interp_bp[kl] = w11 * p.b_p[kj_i_] + w12 * p.b_p[kj1i_] + w21 * p.b_p[kj_i1] +
                     w22 * p.b_p[kj1i1];
    p.interp_bz[kl] = w11 * p.b_z[kj_i_] + w12 * p.b_z[kj1i_] + w21 * p.b_z[kj_i1] +
                     w22 * p.b_z[kj1i1];
}

// Build the axis polygon in Cartesian coordinates: the axis of the first
// field period rotated into the other toroidal replications, closed. The
// closure entry equals the first point and is written directly (no
// cross-thread read of axis_xyz[0..2]).
template <class T>
__global__ void axis_xyz_kernel(ExternalKernelParams<T> p) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nAxis = p.nZeta * p.nvper;
    if (idx > nAxis) return;

    if (idx == nAxis) {
        // close the loop: the first point is (r_axis[0], 0, z_axis[0]) since
        // cos_phi[0] == 1 and sin_phi[0] == 0.
        p.axis_xyz[3 * nAxis + 0] = p.r_axis[0];
        p.axis_xyz[3 * nAxis + 1] = 0;
        p.axis_xyz[3 * nAxis + 2] = p.z_axis[0];
        return;
    }

    const int k = idx % p.nZeta;
    const T x0 = p.r_axis[k] * p.cos_phi[k];
    const T y0 = p.r_axis[k] * p.sin_phi[k];

    if (idx < p.nZeta) {
        // first field-period module
        p.axis_xyz[3 * idx + 0] = x0;
        p.axis_xyz[3 * idx + 1] = y0;
        p.axis_xyz[3 * idx + 2] = p.z_axis[k];
        return;
    }

    // rotated into the other toroidal replications
    const int per = idx / p.nZeta;
    const T omega_per = T(2.0 * std::numbers::pi) / T(p.nvper);
    const T cos_per = cos(omega_per * per);
    const T sin_per = sin(omega_per * per);

    p.axis_xyz[3 * idx + 0] = cos_per * x0 - sin_per * y0;
    p.axis_xyz[3 * idx + 1] = sin_per * x0 + cos_per * y0;
    p.axis_xyz[3 * idx + 2] = p.z_axis[k];
}

// Magnetic field of the net toroidal current along the magnetic axis
// (Hanson-Hirshman polygon-filament Biot-Savart, vmecpp's
// AddAxisCurrentFieldSimple), converted to cylindrical components.
template <class T>
__global__ void axis_current_kernel(ExternalKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;
    const int k = kl % p.nZeta;

    const T surface_x = p.r1b[kl] * p.cos_phi[k];
    const T surface_y = p.r1b[kl] * p.sin_phi[k];
    const T surface_z = p.z1b[kl];

    // 1.0e-7 == mu0/4 pi
    // NOTE: The factor of 2 comes from the Hanson-Hirshman Biot-Savart
    // formula, which is Eqn. (8) in Hanson & Hirshman (2002)
    // [Physics of Plasmas 9, 4410].
    const T magnetic_field_scale = T(1.0e-7) * p.net_toroidal_current * T(2.0);

    // Accumulate the segment contributions in vmecpp's order; registers start
    // at zero, so the b_coils_xyz.set_zero() has no kernel equivalent.
    T b_x = 0;
    T b_y = 0;
    T b_z = 0;
    for (int source_index = 0; source_index < p.nZeta * p.nvper;
         ++source_index) {
        const T segment_dx = p.axis_xyz[(source_index + 1) * 3 + 0] -
                             p.axis_xyz[source_index * 3 + 0];
        const T segment_dy = p.axis_xyz[(source_index + 1) * 3 + 1] -
                             p.axis_xyz[source_index * 3 + 1];
        const T segment_dz = p.axis_xyz[(source_index + 1) * 3 + 2] -
                             p.axis_xyz[source_index * 3 + 2];

        const T segment_length =
            sqrt(segment_dx * segment_dx + segment_dy * segment_dy +
                 segment_dz * segment_dz);

        const T r_i_x = surface_x - p.axis_xyz[source_index * 3 + 0];
        const T r_i_y = surface_y - p.axis_xyz[source_index * 3 + 1];
        const T r_i_z = surface_z - p.axis_xyz[source_index * 3 + 2];
        const T r_i = sqrt(r_i_x * r_i_x + r_i_y * r_i_y + r_i_z * r_i_z);

        const T r_f_x = surface_x - p.axis_xyz[(source_index + 1) * 3 + 0];
        const T r_f_y = surface_y - p.axis_xyz[(source_index + 1) * 3 + 1];
        const T r_f_z = surface_z - p.axis_xyz[(source_index + 1) * 3 + 2];
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

    // transform b_coils_xyz into cylindrical coordinates
    p.curtor_br[kl] = p.cos_phi[k] * b_x + p.sin_phi[k] * b_y;
    p.curtor_bp[kl] = p.cos_phi[k] * b_y - p.sin_phi[k] * b_x;
    p.curtor_bz[kl] = b_z;
}

// Compute b_sub_u, b_sub_v: covariant components of external magnetic field
// and b_dot_n: normal component of external magnetic field.
template <class T>
__global__ void covariant_and_normal_kernel(ExternalKernelParams<T> p) {
    const int kl = blockIdx.x * blockDim.x + threadIdx.x;
    if (kl >= p.nZnT) return;

    // add contributions together
    // --> helps in debugging to have them separate until here
    T full_br = p.interp_br[kl] + p.curtor_br[kl];
    T full_bp = p.interp_bp[kl] + p.curtor_bp[kl];
    T full_bz = p.interp_bz[kl] + p.curtor_bz[kl];

    // covariant components
    p.b_sub_u[kl] = full_br * p.rub[kl] + full_bz * p.zub[kl];
    p.b_sub_v[kl] = full_br * p.rvb[kl] + full_bz * p.zvb[kl] + full_bp * p.r1b[kl];

    // normal component
    p.b_dot_n[kl] =
        -(full_br * p.snr[kl] + full_bp * p.snv[kl] + full_bz * p.snz[kl]);
}

template <class T>
ExternalFieldOperator<T>::ExternalFieldOperator(
    const Sizes& sizes,
    const SurfaceGeometryOperator<T>& sg,
    const MgridProvider& mgrid)
    : sizes_(sizes), sg_(sg), stream_(nullptr) {
    if (!mgrid.is_loaded()) {
        throw VfieldError("ExternalFieldOperator: no mgrid loaded");
    }

    nvper_ = (sizes_.nZeta == 1) ? 64 : sizes_.nfp;

    min_r_ = mgrid.min_r;
    max_r_ = mgrid.max_r;
    delta_r_ = mgrid.delta_r;
    min_z_ = mgrid.min_z;
    max_z_ = mgrid.max_z;
    delta_z_ = mgrid.delta_z;
    num_r_ = mgrid.num_r;
    num_z_ = mgrid.num_z;
    num_phi_ = mgrid.num_phi;
    has_fixed_field_ = mgrid.has_fixed_field();

    if (has_fixed_field_) {
        upload_converted(mgrid.fixed_br(), &fixed_br_);
        upload_converted(mgrid.fixed_bp(), &fixed_bp_);
        upload_converted(mgrid.fixed_bz(), &fixed_bz_);
    } else {
        upload_converted(mgrid.b_r, &b_r_);
        upload_converted(mgrid.b_p, &b_p_);
        upload_converted(mgrid.b_z, &b_z_);
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
    p.cos_phi = sg_.cos_phi();
    p.sin_phi = sg_.sin_phi();
    p.b_r = b_r_.data();
    p.b_p = b_p_.data();
    p.b_z = b_z_.data();
    p.fixed_br = fixed_br_.data();
    p.fixed_bp = fixed_bp_.data();
    p.fixed_bz = fixed_bz_.data();
    p.min_r = min_r_;
    p.max_r = max_r_;
    p.delta_r = delta_r_;
    p.min_z = min_z_;
    p.max_z = max_z_;
    p.delta_z = delta_z_;
    p.num_r = num_r_;
    p.num_z = num_z_;
    p.num_phi = num_phi_;
    p.has_fixed_field = has_fixed_field_;
    p.r_axis = d_r_axis;
    p.z_axis = d_z_axis;
    p.axis_xyz = axis_xyz_.data();
    p.net_toroidal_current = net_toroidal_current;
    p.nZeta = sizes_.nZeta;
    p.nvper = nvper_;
    p.nZnT = sizes_.nZnT;
    p.interp_br = interp_br_.data();
    p.interp_bp = interp_bp_.data();
    p.interp_bz = interp_bz_.data();
    p.curtor_br = curtor_br_.data();
    p.curtor_bp = curtor_bp_.data();
    p.curtor_bz = curtor_bz_.data();
    p.b_sub_u = bsubu_.data();
    p.b_sub_v = bsubv_.data();
    p.b_dot_n = bdotn_.data();
    p.out_of_bounds = out_of_bounds_.data();

    check_cuda(cudaMemsetAsync(p.out_of_bounds, 0, sizeof(int), stream_),
               "ExternalFieldOperator::update: flag reset");
    launch_checked(reinterpret_cast<const void*>(&mgrid_interp_kernel<T>),
                  sizes_.nZnT, BLOCK_SIZE, p, stream_,
                  "ExternalFieldOperator::update: mgrid interpolation");
    launch_checked(reinterpret_cast<const void*>(&axis_xyz_kernel<T>),
                  sizes_.nZeta * nvper_ + 1, BLOCK_SIZE, p, stream_,
                  "ExternalFieldOperator::update: axis polygon");
    launch_checked(reinterpret_cast<const void*>(&axis_current_kernel<T>),
                  sizes_.nZnT, BLOCK_SIZE, p, stream_,
                  "ExternalFieldOperator::update: axis current");
    launch_checked(reinterpret_cast<const void*>(&covariant_and_normal_kernel<T>),
                  sizes_.nZnT, BLOCK_SIZE, p, stream_,
                  "ExternalFieldOperator::update: covariant/normal");

    // Out-of-bounds check for the grid-cropping warning (the flag copy
    // synchronizes the stream; the warning fires only in pathological cases).
    int out_of_bounds = 0;
    out_of_bounds_.download(&out_of_bounds, 1);
    if (out_of_bounds != 0 && !has_fixed_field_) {
        // Recompute the raw min/max of the surface points for the message.
        std::vector<T> r = to_host_t(sg_.r1b(), sizes_.nZnT);
        std::vector<T> z = to_host_t(sg_.z1b(), sizes_.nZnT);
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
