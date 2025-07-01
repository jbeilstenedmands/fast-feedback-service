#ifndef BASELINE_REFINER_DETECTOR_PARAM
#define BASELINE_REFINER_DETECTOR_PARAM
#include <dx2/detector.hpp>
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;

class DetectorParameterisation {
public:
  DetectorParameterisation(
    const Panel &panel,
    bool fix_dist=false,
    bool fix_shift1=false,
    bool fix_shift2=false,
    bool fix_tau1=false,
    bool fix_tau2=false,
    bool fix_tau3=false);
  std::vector<double> get_params() const;
  void set_params(std::vector<double> p);
  Matrix3d get_state() const;
  std::vector<Matrix3d> get_dS_dp() const;
  bool dist_fixed() const;
  bool shift1_fixed() const;
  bool shift2_fixed() const;
  bool tau1_fixed() const;
  bool tau2_fixed() const;
  bool tau3_fixed() const;

private:
  std::vector<double> params_ = {0.0,0.0,0.0,0.0,0.0,0.0}; //
  void compose();
  std::vector<Matrix3d> dS_dp{
    6,
    Matrix3d {{0, 0, 0}, {0, 0, 0}, {0,0,0}}};
  Vector3d initial_offset{{0.0,0.0,0.0}};
  Vector3d initial_d1{{0.0,0.0,0.0}};
  Vector3d initial_d2{{0.0,0.0,0.0}};
  Vector3d initial_dn{{0.0,0.0,0.0}};
  Vector3d initial_origin{{0.0,0.0,0.0}};
  Vector3d current_origin{{0.0,0.0,0.0}};
  Vector3d current_d1{{0.0,0.0,0.0}};
  Vector3d current_d2{{0.0,0.0,0.0}};
  bool _fix_dist{true};
  bool _fix_shift1{false};
  bool _fix_shift2{true};
  bool _fix_tau1{true};
  bool _fix_tau2{true};
  bool _fix_tau3{true};
};


#endif  // BASELINE_REFINER_DETECTOR_PARAM