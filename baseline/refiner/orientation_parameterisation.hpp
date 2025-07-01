#ifndef BASELINE_REFINER_ORIENTATION_PARAM
#define BASELINE_REFINER_ORIENTATION_PARAM
#include <dx2/crystal.hpp>
#include <dx2/goniometer.hpp>
#include <Eigen/Dense>

using Eigen::Matrix3d;
using Eigen::Vector3d;

class CrystalOrientationCompose {
public:
    CrystalOrientationCompose(const Matrix3d &U0,
                              double phi1,
                              const Vector3d &phi1_axis,
                              double phi2,
                              const Vector3d &phi2_axis,
                              double phi3,
                              const Vector3d &phi3_axis);
    Matrix3d U() const;

    Matrix3d dU_dphi1() const;

    Matrix3d dU_dphi2() const;

    Matrix3d dU_dphi3() const;
private:
  Matrix3d U_;
  Matrix3d dU_dphi1_;
  Matrix3d dU_dphi2_;
  Matrix3d dU_dphi3_;
};




class OrientationParameterisation {
public:
  OrientationParameterisation(const Crystal& crystal);
  std::vector<double> get_params() const;
  void set_params(std::vector<double> p);
  Matrix3d get_state() const;
  std::vector<Matrix3d> get_dS_dp() const;

private:
  std::vector<double> params = {0.0, 0.0, 0.0};
  std::vector<Vector3d> axes{3, Vector3d(1.0, 0.0, 0.0)};
  void compose();
  Matrix3d istate{};
  Matrix3d U_{};
  std::vector<Matrix3d> dS_dp{
    3,
    Matrix3d{{1.0, 0, 0}, {0, 1.0, 0}, {0, 0, 1.0}}};
};

#endif  // BASELINE_REFINER_ORIENTATION_PARAM