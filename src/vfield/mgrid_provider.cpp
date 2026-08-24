// mgrid_provider.cpp — mgrid NetCDF reading and in-memory loading.
//
// Direct port of vmecpp's MGridProvider (free_boundary/mgrid_provider) with
// the NetCDF C API used directly (no netcdf_io helper), absl::Status replaced
// by exceptions, and Eigen::VectorXd by std::vector<double>. The NetCDF
// section is guarded by VFIELD_HAVE_NETCDF.
#include "vfield/free_boundary/mgrid_provider.hpp"

#ifdef VFIELD_HAVE_NETCDF
#include <netcdf.h>
#endif  // VFIELD_HAVE_NETCDF

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace vfield {

namespace {

#ifdef VFIELD_HAVE_NETCDF

// Helpers over the NetCDF C API: each throws a std::runtime_error with a
// descriptive message on failure.

int open_variable(int ncid, const std::string& name) {
    int varid = -1;
    if (nc_inq_varid(ncid, name.c_str(), &varid) != NC_NOERR) {
        throw std::runtime_error("mgrid file has no variable '" + name + "'");
    }
    return varid;
}

void require_rank(int ncid, const std::string& name, int varid, int rank) {
    int ndims = -1;
    nc_inq_varndims(ncid, varid, &ndims);
    if (ndims != rank) {
        throw std::runtime_error("variable '" + name + "' has rank " +
                                 std::to_string(ndims) + ", expected " +
                                 std::to_string(rank));
    }
}

int read_int_scalar(int ncid, const std::string& name) {
    int varid = open_variable(ncid, name);
    require_rank(ncid, name, varid, 0);
    int value = 0;
    if (nc_get_var_int(ncid, varid, &value) != NC_NOERR) {
        throw std::runtime_error("failed to read variable '" + name + "'");
    }
    return value;
}

double read_double_scalar(int ncid, const std::string& name) {
    int varid = open_variable(ncid, name);
    require_rank(ncid, name, varid, 0);
    double value = 0.0;
    if (nc_get_var_double(ncid, varid, &value) != NC_NOERR) {
        throw std::runtime_error("failed to read variable '" + name + "'");
    }
    return value;
}

std::string read_string_scalar(int ncid, const std::string& name) {
    int varid = open_variable(ncid, name);
    require_rank(ncid, name, varid, 1);
    // The string variable's own dimension id comes from nc_inq_vardimid.
    int dimids[1] = {0};
    if (nc_inq_vardimid(ncid, varid, dimids) != NC_NOERR) {
        throw std::runtime_error("failed to query variable '" + name + "'");
    }
    std::size_t len = 0;
    nc_inq_dimlen(ncid, dimids[0], &len);
    std::string value(len, '\0');
    if (nc_get_var_text(ncid, varid, value.data()) != NC_NOERR) {
        throw std::runtime_error("failed to read variable '" + name + "'");
    }
    return value;
}

std::vector<double> read_array_3d(int ncid,
                                const std::string& name,
                                int num_phi,
                                int num_z,
                                int num_r) {
    int varid = open_variable(ncid, name);
    require_rank(ncid, name, varid, 3);
    int dimids[3] = {0, 0, 0};
    nc_inq_vardimid(ncid, varid, dimids);
    // The mgrid convention: dims (phi, zee, rad) — R fastest.
    std::size_t phi_len = 0, z_len = 0, r_len = 0;
    nc_inq_dimlen(ncid, dimids[0], &phi_len);
    nc_inq_dimlen(ncid, dimids[1], &z_len);
    nc_inq_dimlen(ncid, dimids[2], &r_len);
    if (phi_len != static_cast<std::size_t>(num_phi) ||
        z_len != static_cast<std::size_t>(num_z) ||
        r_len != static_cast<std::size_t>(num_r)) {
        throw std::runtime_error(
            "variable '" + name + "' has shape (" + std::to_string(phi_len) +
            ", " + std::to_string(z_len) + ", " + std::to_string(r_len) +
            "), expected (" + std::to_string(num_phi) + ", " +
            std::to_string(num_z) + ", " + std::to_string(num_r) + ")");
    }
    std::vector<double> out(num_phi * num_z * num_r);
    if (nc_get_var_double(ncid, varid, out.data()) != NC_NOERR) {
        throw std::runtime_error("failed to read variable '" + name + "'");
    }
    return out;
}

#endif  // VFIELD_HAVE_NETCDF

std::string three_digit_field(int i) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%03d", i + 1);
    return std::string(buf);
}

void validate_coil_count(std::size_t n_currents, int nextcur) {
    if (static_cast<int>(n_currents) != nextcur) {
        throw std::runtime_error(
            "Number of currents " + std::to_string(n_currents) +
            " does not match number of mgrid coil fields nextcur=" +
            std::to_string(nextcur) + ".");
    }
}

}  // namespace

void MgridProvider::load_file(const std::string& filename,
                             const std::vector<double>& coil_currents) {
#ifdef VFIELD_HAVE_NETCDF
    {  // try to open file in order to check if it is accessible
        std::ifstream fp(filename);
        if (!fp.is_open()) {
            throw std::runtime_error("Could not find '" + filename + "'.");
        }
    }

    int ncid = 0;
    if (nc_open(filename.c_str(), NC_NOWRITE, &ncid) != NC_NOERR) {
        throw std::runtime_error(
            "NetCDF couldn't open '" + filename +
            "', despite passing preconditions. The file may be corrupted.");
    }

    // Reads below throw std::runtime_error on failure (e.g. a missing
    // variable), which we propagate to the caller instead of aborting.
    auto with_context = [&filename](const std::runtime_error& e) {
        return std::runtime_error("While reading mgrid file '" + filename +
                                  "': " + e.what());
    };

    try {
        nfp = read_int_scalar(ncid, "nfp");

        num_r = read_int_scalar(ncid, "ir");
        min_r = read_double_scalar(ncid, "rmin");
        max_r = read_double_scalar(ncid, "rmax");
        delta_r = (max_r - min_r) / (num_r - 1.0);

        num_z = read_int_scalar(ncid, "jz");
        min_z = read_double_scalar(ncid, "zmin");
        max_z = read_double_scalar(ncid, "zmax");
        delta_z = (max_z - min_z) / (num_z - 1.0);

        num_phi = read_int_scalar(ncid, "kp");

        nextcur = read_int_scalar(ncid, "nextcur");
        validate_coil_count(coil_currents.size(), nextcur);

        mgrid_mode = read_string_scalar(ncid, "mgrid_mode");

        // Resize and make sure that the accumulation arrays are reset to zeros
        // if they contained previous contents from an earlier call.
        b_r.assign(num_phi * num_z * num_r, 0.0);
        b_p.assign(num_phi * num_z * num_r, 0.0);
        b_z.assign(num_phi * num_z * num_r, 0.0);

        // combine coil contributions, weighted by coil currents
        for (int i = 0; i < nextcur; ++i) {
            // for each coil group:
            // get 3d double array "br_%03d", "bp_%03d", "bz_%03d"
            // from i=1, 2, ..., nextcur
            const std::vector<double> br_contrib = read_array_3d(
                ncid, "br_" + three_digit_field(i), num_phi, num_z, num_r);
            const std::vector<double> bp_contrib = read_array_3d(
                ncid, "bp_" + three_digit_field(i), num_phi, num_z, num_r);
            const std::vector<double> bz_contrib = read_array_3d(
                ncid, "bz_" + three_digit_field(i), num_phi, num_z, num_r);

            for (int index_phi = 0; index_phi < num_phi; ++index_phi) {
                for (int index_z = 0; index_z < num_z; ++index_z) {
                    for (int index_r = 0; index_r < num_r; ++index_r) {
                        const int linear_index =
                            (index_phi * num_z + index_z) * num_r + index_r;

                        b_r[linear_index] +=
                            br_contrib[linear_index] * coil_currents[i];
                        b_p[linear_index] +=
                            bp_contrib[linear_index] * coil_currents[i];
                        b_z[linear_index] +=
                            bz_contrib[linear_index] * coil_currents[i];
                    }  // index_r
                }  // index_z
            }  // index_phi
        }  // nextcur
    } catch (const std::runtime_error& e) {
        nc_close(ncid);
        throw with_context(e);
    }

    if (nc_close(ncid) != NC_NOERR) {
        throw std::runtime_error("Failed to close NetCDF file.");
    }

    has_mgrid_loaded_ = true;
    has_fixed_field_ = false;
#else
    (void)filename;
    (void)coil_currents;
    throw std::runtime_error(
        "MgridProvider::load_file: NetCDF support is not compiled in "
        "(build with NetCDF available or pass -DVFIELD_USE_NETCDF=ON).");
#endif  // VFIELD_HAVE_NETCDF
}

void MgridProvider::load_fields(const ResponseTable& table,
                               const std::vector<double>& coil_currents) {
    if (coil_currents.size() != table.b_r.size()) {
        throw std::runtime_error(
            "Number of currents " + std::to_string(coil_currents.size()) +
            " does not match number of coil fields in the response table " +
            std::to_string(table.b_r.size()) + ".");
    }

    nfp = table.nfp;

    num_r = table.num_r;
    min_r = table.min_r;
    max_r = table.max_r;
    delta_r = (max_r - min_r) / (num_r - 1.0);

    num_z = table.num_z;
    min_z = table.min_z;
    max_z = table.max_z;
    delta_z = (max_z - min_z) / (num_z - 1.0);

    num_phi = table.num_phi;

    nextcur = static_cast<int>(coil_currents.size());

    if (table.normalize_by_currents) {
        mgrid_mode = "S";
    } else {
        mgrid_mode = "R";
    }

    const int num_grid_points = num_phi * num_z * num_r;
    b_r.assign(num_grid_points, 0.0);
    b_p.assign(num_grid_points, 0.0);
    b_z.assign(num_grid_points, 0.0);

    // combine coil contributions, weighted by coil currents
    for (int i = 0; i < nextcur; ++i) {
        for (int linear_index = 0; linear_index < num_grid_points;
             ++linear_index) {
            b_r[linear_index] += table.b_r[i][linear_index] * coil_currents[i];
            b_p[linear_index] += table.b_p[i][linear_index] * coil_currents[i];
            b_z[linear_index] += table.b_z[i][linear_index] * coil_currents[i];
        }  // linear_index
    }  // nextcur

    has_mgrid_loaded_ = true;
    has_fixed_field_ = false;
}

void MgridProvider::set_fixed_magnetic_field(const std::vector<double>& fixed_br,
                                          const std::vector<double>& fixed_bp,
                                          const std::vector<double>& fixed_bz) {
    // copy into local storage
    fixed_br_ = fixed_br;
    fixed_bp_ = fixed_bp;
    fixed_bz_ = fixed_bz;

    has_mgrid_loaded_ = true;
    has_fixed_field_ = true;
}

}  // namespace vfield
