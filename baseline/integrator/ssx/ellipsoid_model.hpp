#pragma once

#include <Eigen/Core>
#include <Eigen/Cholesky>

#include <array>
#include <stdexcept>

#include "ellipsoid_parameterisation.hpp"
using DerivativeMatrices = std::array<Eigen::Matrix3d, 6>;

class Simple6ProfileModel {
public:

  using Vector6d = Eigen::Matrix<double,6,1>;
  using Matrix3d = Eigen::Matrix3d;

  static constexpr const char* name = "simple6";

  Simple6ProfileModel()
      : params_(Vector6d::Zero()),
        n_obs_(0)
  {}

  explicit Simple6ProfileModel(
      const Vector6d& params)
      : params_(params),
        n_obs_(0)
  {}

  //--------------------------------------------------------------------------
  // Factory methods
  //--------------------------------------------------------------------------

  static Simple6ProfileModel from_sigma_d(
      double sigma_d)
  {
    Vector6d p;

    p <<
      sigma_d,
      0.0,
      sigma_d,
      0.0,
      0.0,
      sigma_d;

    return Simple6ProfileModel(p);
  }

  static Simple6ProfileModel from_sigma(
      const Matrix3d& sigma)
  {
    Eigen::LLT<Matrix3d> llt(sigma);

    if (llt.info() != Eigen::Success) {
      throw std::runtime_error(
          "Sigma matrix is not positive definite");
    }

    Matrix3d L = llt.matrixL();

    Vector6d p;

    p <<
      L(0,0),
      L(1,0),
      L(1,1),
      L(2,0),
      L(2,1),
      L(2,2);

    return Simple6ProfileModel(p);
  }

  //--------------------------------------------------------------------------
  // Accessors
  //--------------------------------------------------------------------------

  const Vector6d& parameters() const {
    return params_;
  }

  void set_parameters(
      const Vector6d& p)
  {
    params_ = p;
  }

  std::size_t n_obs() const {
    return n_obs_;
  }

  void set_n_obs(
      std::size_t n)
  {
    n_obs_ = n;
  }

  //--------------------------------------------------------------------------
  // Parameterisation
  //--------------------------------------------------------------------------

  Simple6MosaicityParameterisation parameterisation() const {
    return Simple6MosaicityParameterisation(params_);
  }

  //--------------------------------------------------------------------------
  // Covariance matrix
  //--------------------------------------------------------------------------

  Matrix3d sigma() const {
    return parameterisation().sigma();
  }

  //--------------------------------------------------------------------------
  // First derivatives
  //--------------------------------------------------------------------------

  DerivativeMatrices first_derivatives() const {
    return parameterisation().first_derivatives();
  }

  //--------------------------------------------------------------------------
  // Mosaicity summary
  //--------------------------------------------------------------------------

  Simple6MosaicityParameterisation::Mosaicity
  mosaicity() const
  {
    return parameterisation().mosaicity();
  }

  //--------------------------------------------------------------------------
  // Model-state interactions
  //--------------------------------------------------------------------------

  template <typename ModelState>
  void update_model_state_parameters(
      ModelState& state) const
  {
    state.set_M_params(params_);
  }

  template <typename ModelState>
  void update_model(
      ModelState& state)
  {
    Matrix3d sigma =
        state.mosaicity_covariance_matrix();

    Eigen::SelfAdjointEigenSolver<Matrix3d>
        solver(sigma);

    if (solver.info() != Eigen::Success) {
      throw std::runtime_error(
          "Failed eigendecomposition of mosaicity matrix");
    }

    constexpr double mosaicity_max_limit = 0.004;

    double max_eigenvalue =
        solver.eigenvalues().maxCoeff();

    double min_eigenvalue =
        solver.eigenvalues().minCoeff();

    if (max_eigenvalue >
        mosaicity_max_limit * mosaicity_max_limit)
    {
      throw std::runtime_error(
          "Mosaicity matrix is unphysically large");
    }

    if (min_eigenvalue < 1e-12) {
      throw std::runtime_error(
          "Mosaicity matrix is unphysically small");
    }

    params_ = state.M_params();
  }

private:

  Vector6d params_;

  std::size_t n_obs_;
};