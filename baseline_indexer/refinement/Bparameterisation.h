#ifndef DIALS_RESEARCH_BPARAM
#define DIALS_RESEARCH_BPARAM

#include <cctbx/sgtbx/tensor_rank_2.h>


#include <gemmi/math.hpp> // for symmetric 3x3 matrix SMat33
#include <g_gradients.h>
#include <Eigen/Dense>
#include <gemmi/symmetry.hpp>
#include <gemmi/scaling.hpp> // for constraints
#include <dx2/crystal.h>
using Eigen::Matrix3d;
using Eigen::Vector3d;

// Define a crystal orientation class - based on cctbx/crystal_orientation.h
class crystal_orientation {
public:
  crystal_orientation(Matrix3d const&, bool const&);
  Matrix3d reciprocal_matrix() const;
}

// Define the AG converter
struct AG {// convert orientation matrix A to metrical matrix g & reverse
  /*The general orientation matrix A is re-expressed in terms of the
    upper-triangular fractionalization matrix F by means of the following
    transformation:
                           F = (D * C * B * A).transpose()
    where D,C,B are three rotation matrices.
  */
  crystal_orientation orientation;
  double phi,psi,theta; //in radians
  Matrix3d B,C,D,F;
  SMat33<double> G;
  void forward(crystal_orientation const& ori){}
  void validate_and_setG(SMat33<double> const& g){}
  Matrix3d back() const {
    cctbx::uctbx::unit_cell ersatz_uc ( cctbx::uctbx::unit_cell(G).reciprocal() );
    // ersatz F is the fract. matrix, PDB Convention, compatible with CCTBX
    Matrix3d ersatzF (ersatz_uc.fractionalization_matrix());
    // Fback is the lower-triangular matrix compatible with the Rsymop
    // paper, equation (3) = {{a*_x,0,0},{a*_y,b*_y,0},{a*_z,b*_z,c*_z}}
    Matrix3d Fback (ersatzF.transpose());
    return (B.inverse() * C.inverse() * D.inverse() * Fback);
  }
  crystal_orientation back_as_orientation() const {
    return orientation(back());
  }
}

class Constraints {
public:
  Constraints(gemmi::SpaceGroup space_group){//note working in reciprocal space
    constraint_matrix(adp_symmetry_constraints(space_group));
  }
  std::array<double, 6> independent_params(SMat33 G){

  }
  SMat33<double> all_params(std::array<double, 6> independent){

  }
private:
  std::vector<Vec6> constraint_matrix;
}

class SymmetrizeReduceEnlarge {
public:
  SymmetrizeReduceEnlarge(gemmi::SpaceGroup space_group);
  void set_orientation(Matrix3d B);
  std::array<double, 6> forward_independent_parameters();
  crystal_orientation backward_orientation(std::array<double, 6> independent);
  std::vector<Matrix3d> forward_gradients();

private:
  gemmi::SpaceGroup space_group_;
  cctbx::sgtbx::tensor_rank_2::constraints<double> constraints_;
  crystal_orientation orientation_{};
  AG Bconverter{};
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
    : space_group_(space_group), constraints_(space_group, true) {}

void SymmetrizeReduceEnlarge::set_orientation(Matrix3d B) {
  orientation_ = crystal_orientation(B, true);
}

std::array<double, 6> SymmetrizeReduceEnlarge::forward_independent_parameters() {
  Bconverter.forward(orientation_);
  return constraints_.independent_params(Bconverter.G);
}

crystal_orientation SymmetrizeReduceEnlarge::backward_orientation(
  std::array<double, 6> independent){
  SMat33<double> ustar = constraints_.all_params(independent);
  Bconverter.validate_and_setG(ustar);
  orientation_ = Bconverter.back_as_orientation();
  return orientation_;
}

std::vector<Matrix3d> SymmetrizeReduceEnlarge::forward_gradients() {
  return dB_dp(Bconverter, constraints_);
}

void SimpleBParameterisation::compose() {
  std::array<double, 6> vals(params.size());
  for (int i = 0; i < params.size(); ++i) {
    vals[i] = 1E-5 * params[i];
  }
  SRE.set_orientation(B_);
  SRE.forward_independent_parameters();
  B_ = SRE.backward_orientation(vals).reciprocal_matrix();
  dS_dp = SRE.forward_gradients();
  for (int i = 0; i < dS_dp.size(); ++i) {
    for (int j = 0; j < 9; ++j) {
      dS_dp[i][j] *= 1E-5;
    }
  }
}

SimpleBParameterisation::SimpleBParameterisation(const Crystal &crystal)
    : B_(crystal.get_B()), SRE(crystal.get_space_group()) {
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