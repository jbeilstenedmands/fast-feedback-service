#ifndef DIALS_RESEARCH_BPARAM
#define DIALS_RESEARCH_BPARAM


#include "helpers.h"
#include <gemmi/math.hpp> // for symmetric 3x3 matrix SMat33
#include "g_gradients.h"
#include <Eigen/Dense>
#include <gemmi/symmetry.hpp>
#include <gemmi/cellred.hpp>
#include <gemmi/scaling.hpp> // for constraints
#include <dx2/crystal.h>
using Eigen::Matrix3d;
using Eigen::Vector3d;

class SymmetrizeReduceEnlarge {
public:
  SymmetrizeReduceEnlarge(gemmi::SpaceGroup space_group);
  void set_orientation(Matrix3d B);
  std::array<double, 6> forward_independent_parameters();
  crystal_orientation backward_orientation(std::array<double, 6> independent);
  std::vector<Matrix3d> forward_gradients();

private:
  gemmi::SpaceGroup space_group_;
  Constraints constraints_;
  crystal_orientation orientation_;
  AG Bconverter;
};

class SimpleBParameterisation {
public:
  SimpleBParameterisation(const Crystal &crystal);
  std::vector<double> get_params() const;
  void set_params(std::vector<double>);
  Matrix3d get_state() const;
  std::vector<Matrix3d> get_dS_dp() const;

private:
  std::vector<double> params;
  void compose();
  Matrix3d B_{};
  std::vector<Matrix3d> dS_dp{};
  SymmetrizeReduceEnlarge SRE;
};

SymmetrizeReduceEnlarge::SymmetrizeReduceEnlarge(gemmi::SpaceGroup space_group)
    : space_group_(space_group), constraints_(space_group) {}

void SymmetrizeReduceEnlarge::set_orientation(Matrix3d B) {
  orientation_ = crystal_orientation(B);
}

std::array<double, 6> SymmetrizeReduceEnlarge::forward_independent_parameters() {
  Bconverter.forward(orientation_);
  return constraints_.independent_params(Bconverter.G);
}

crystal_orientation SymmetrizeReduceEnlarge::backward_orientation(
  std::array<double, 6> independent){
  gemmi::SMat33<double> ustar = constraints_.all_params(independent);
  Bconverter.validate_and_setG(ustar);
  orientation_ = Bconverter.back_as_orientation();
  return orientation_;
}

std::vector<Matrix3d> SymmetrizeReduceEnlarge::forward_gradients() {
  return dB_dp(Bconverter, constraints_);
}

void SimpleBParameterisation::compose() {
  std::array<double, 6> vals{0}; // can be less than 6 long?
  for (int i = 0; i < params.size(); ++i) {
    vals[i] = 1E-5 * params[i];
  }
  SRE.set_orientation(B_);
  SRE.forward_independent_parameters();
  B_ = SRE.backward_orientation(vals).reciprocal_matrix();
  dS_dp = SRE.forward_gradients();
  for (int i = 0; i < dS_dp.size(); ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k=0;k<3;k++){
        dS_dp[i](j,k) *= 1E-5;
      }
    }
  }
}

SimpleBParameterisation::SimpleBParameterisation(const Crystal &crystal)
    : B_(crystal.get_B_matrix()), SRE(crystal.get_space_group()) {
  // first get params
  SRE.set_orientation(B_);
  std::array<double, 6> X = SRE.forward_independent_parameters();
  params = std::vector<double>(X.size());
  for (int i = 0; i < X.size(); ++i) {
    params[i] = 1E5 * X[i];
  }
  compose();
}

std::vector<double> SimpleBParameterisation::get_params() const {
  return params;
}
Matrix3d SimpleBParameterisation::get_state() const {
  return B_;
}
void SimpleBParameterisation::set_params(std::vector<double> p) {
  params = p;
  compose();
}
std::vector<Matrix3d> SimpleBParameterisation::get_dS_dp() const {
  return dS_dp;
}

#endif  // DIALS_RESEARCH_BPARAM