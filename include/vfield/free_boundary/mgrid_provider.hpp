// mgrid_provider.hpp — host-side mgrid (external coil field) provider.
//
// Port of vmecpp's MGridProvider (free_boundary/mgrid_provider): reads a
// MAKEGRID-style NetCDF file (br_iii/bp_iii/bz_iii per coil group, field per
// ampere, scaled by extcur on load) or takes the field arrays in memory, and
// exposes the summed R-Z-phi grids that the external-field operator uploads
// to the device. NetCDF support is optional (VFIELD_HAVE_NETCDF); the
// response-table and fixed-field paths need no NetCDF. Errors throw
// std::runtime_error instead of returning absl::Status.
#ifndef VFIELD_FREE_BOUNDARY_MGRID_PROVIDER_HPP_
#define VFIELD_FREE_BOUNDARY_MGRID_PROVIDER_HPP_

#include <cstddef>
#include <string>
#include <vector>

namespace vfield {

class MgridProvider {
   public:
    // Response-table input: coil-field arrays with one row per coil group,
    // columns over the (phi, z, r) grid points (row-major over the grid).
    struct ResponseTable {
        int nfp;
        int num_r;
        double min_r;
        double max_r;
        int num_z;
        double min_z;
        double max_z;
        int num_phi;
        bool normalize_by_currents;
        // One row per coil group, each of length num_phi * num_z * num_r.
        std::vector<std::vector<double>> b_r;
        std::vector<std::vector<double>> b_p;
        std::vector<std::vector<double>> b_z;
    };

    MgridProvider() = default;

    // Reads an mgrid NetCDF file and accumulates the per-ampere coil fields
    // weighted by coil_currents (A). Throws on missing/invalid input;
    // throws "not compiled in" when NetCDF support is absent.
    void loadFile(const std::string& filename,
                  const std::vector<double>& coil_currents);

    // In-memory variant of loadFile (no NetCDF needed).
    void loadFields(const ResponseTable& table,
                    const std::vector<double>& coil_currents);

    // Bypasses the grid entirely: the external-field operator copies the
    // per-point fields verbatim instead of interpolating.
    void setFixedMagneticField(const std::vector<double>& fixed_br,
                               const std::vector<double>& fixed_bp,
                               const std::vector<double>& fixed_bz);

    // Summed coil field on the R-Z-phi grid, [numPhi * numZ * numR], linear
    // index (phi * numZ + z) * numR + r.
    std::vector<double> bR;
    std::vector<double> bP;
    std::vector<double> bZ;

    int nfp = -1;

    int numR = -1;
    double minR = 0.0;
    double maxR = 0.0;
    double deltaR = 0.0;

    int numZ = -1;
    double minZ = 0.0;
    double maxZ = 0.0;
    double deltaZ = 0.0;

    int numPhi = -1;

    int nextcur = -1;

    std::string mgrid_mode;

    bool isLoaded() const { return has_mgrid_loaded_; }
    bool hasFixedField() const { return has_fixed_field_; }
    const std::vector<double>& fixedBr() const { return fixed_br_; }
    const std::vector<double>& fixedBp() const { return fixed_bp_; }
    const std::vector<double>& fixedBz() const { return fixed_bz_; }

   private:
    bool has_mgrid_loaded_ = false;
    bool has_fixed_field_ = false;
    std::vector<double> fixed_br_;
    std::vector<double> fixed_bp_;
    std::vector<double> fixed_bz_;
};

}  // namespace vfield

#endif  // VFIELD_FREE_BOUNDARY_MGRID_PROVIDER_HPP_
