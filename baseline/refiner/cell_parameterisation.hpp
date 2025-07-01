#ifndef REFINE_BPARAM
#define REFINE_BPARAM

#include <gemmi/math.hpp> // for symmetric 3x3 matrix SMat33
#include <dx2/crystal.hpp>
#include <Eigen/Dense>
#include <math.h>
using Eigen::Matrix3d;
using Eigen::Vector3d;


// Define the BG converter
struct BG {
  // convert orientation matrix B (called A internally here) to metrical
  // matrix g & reverse
  /*The general orientation matrix A is re-expressed in terms of the
    upper-triangular fractionalization matrix F by means of the following
    transformation:
                           F = (D * C * B * A).transpose()
    where D,C,B are three rotation matrices.
  */
  Matrix3d orientation;
  double phi,psi,theta; //in radians
  Matrix3d B,C,D,F;
  gemmi::SMat33<double> G;
  /// Convert orientation matrix to metrical matrix G
  void forward(Matrix3d const& ori);
  void validate_and_setG(gemmi::SMat33<double> const& g);
  Matrix3d back() const;
  Matrix3d back_as_orientation() const;
};


std::vector<Matrix3d> calc_dB_dg(BG Bconverter);

std::vector<Matrix3d> dB_dp(BG Bconverter);

// A class to manage the translation from B to G and back,
// plus any symmetry constraints (we are sticking to P1 here though.)
class SymmetrizeReduceEnlarge {
public:
  SymmetrizeReduceEnlarge();
  void set_orientation(Matrix3d B);
  std::vector<double> forward_independent_parameters();
  Matrix3d backward_orientation(std::vector<double> independent);
  std::vector<Matrix3d> forward_gradients();

private:
  Matrix3d orientation_{};
  BG Bconverter{};
};


class CellParameterisation {
public:
  CellParameterisation(const Crystal& crystal);
  std::vector<double> get_params() const;
  void set_params(std::vector<double>);
  Matrix3d get_state() const;
  std::vector<Matrix3d> get_dS_dp() const;

private:
  std::vector<double> params_ = {0.0,0.0,0.0,0.0,0.0,0.0};
  void compose();
  Matrix3d B_{};
  std::vector<Matrix3d> dS_dp{};
  SymmetrizeReduceEnlarge SRE;
};

#endif  // REFINE_BPARAM