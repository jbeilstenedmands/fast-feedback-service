#pragma once

#include <Eigen/Core>

#include <string>
#include <vector>

#include "ellipsoid_model.hpp"
#include "ellipsoid_parameterisation.hpp"

using Vector6d = Simple6MosaicityParameterisation::Vector6d;
using DerivativeMatrices = std::array<Eigen::Matrix3d, 6>;

class ModelState {
public:

  using VectorXd = Eigen::VectorXd;
  using Matrix3d = Eigen::Matrix3d;

  ModelState(const Simple6ProfileModel& profile_model): 
    profile_model_(profile_model),
    mosaicity_parameterisation_(profile_model.parameterisation()){}
  // --------------------------------------------------------------------------
  // Access to profile model
  // --------------------------------------------------------------------------

  const Simple6ProfileModel& profile_model() const {
    return profile_model_;
  }

  /*auto unit_cell() const {
    return profile_model_.crystal().get_unit_cell();
  }

  Matrix3d A_matrix() const {

    const auto A = profile_model_.crystal().get_A();

    Matrix3d result;

    result <<
      A[0], A[1], A[2],
      A[3], A[4], A[5],
      A[6], A[7], A[8];

    return result;
  }*/

  // --------------------------------------------------------------------------
  // Mosaicity parameters
  // --------------------------------------------------------------------------

  Vector6d M_params() const {
    return mosaicity_parameterisation_.parameters();
  }

  void set_M_params(const Vector6d& params) {
    mosaicity_parameterisation_.set_parameters(params);
  }

  // --------------------------------------------------------------------------
  // Covariance matrix
  // --------------------------------------------------------------------------

  Matrix3d mosaicity_covariance_matrix() const {
    return mosaicity_parameterisation_.sigma();
  }

  // --------------------------------------------------------------------------
  // First derivatives
  // --------------------------------------------------------------------------

  DerivativeMatrices dM_dp() const {
    return mosaicity_parameterisation_.first_derivatives();
  }

  // --------------------------------------------------------------------------
  // Active parameters
  // --------------------------------------------------------------------------

  Vector6d active_parameters() const {
    return M_params();
  }

  void set_active_parameters(const Vector6d& params) {
    set_M_params(params);
  }

  std::size_t num_active_parameters() const {
    return static_cast<std::size_t>(
        mosaicity_parameterisation_.parameters().size());
  }

  // --------------------------------------------------------------------------
  // Parameter labels
  // --------------------------------------------------------------------------

  std::vector<std::string> parameter_labels() const {

    std::vector<std::string> labels = {
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6"
    };

    return labels;
  }

private:

  const Simple6ProfileModel& profile_model_;

  Simple6MosaicityParameterisation mosaicity_parameterisation_;

};