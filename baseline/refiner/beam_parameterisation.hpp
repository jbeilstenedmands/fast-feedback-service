#ifndef BASELINE_REFINER_BEAM_PARAM
#define BASELINE_REFINER_BEAM_PARAM
#include <dx2/beam.hpp>
#include <dx2/goniometer.hpp>
#include <Eigen/Dense>

using Eigen::Vector3d;

class BeamParameterisation {
public:
  BeamParameterisation(
    const MonochromaticBeam& beam, const Goniometer& goniometer,
    bool fix_in_spindle_plane=true, bool fix_out_spindle_plane=false, bool fix_wavelength=true);
  std::vector<double> get_params() const;
  void set_params(std::vector<double> p);
  Vector3d get_state() const;
  std::vector<Vector3d> get_dS_dp() const;
  bool in_spindle_plane_fixed() const;
  bool out_spindle_plane_fixed() const;
  bool wavelength_fixed() const;

private:
  std::vector<double> params_ = {0.0,0.0,0.0}; //mu1, mu2, nu
  void compose();
  Vector3d istate_s0{};
  Vector3d istate_pol_norm{};
  Vector3d s0{};
  Vector3d pn{};
  Vector3d s0_plane_dir1{};
  Vector3d s0_plane_dir2{};
  std::vector<Vector3d> dS_dp{
    3,
    Vector3d(0.0, 0, 0)};
  bool _fix_in_spindle_plane{true};
  bool _fix_out_spindle_plane{false};
  bool _fix_wavelength{true};
};

#endif  // BASELINE_REFINER_BEAM_PARAM