#ifndef BASELINE_REFINER_GRADIENTS_CALCULATOR_H
#define BASELINE_REFINER_GRADIENTS_CALCULATOR_H

#include <dx2/goniometer.hpp>
#include "detector_parameterisation.hpp"
#include "beam_parameterisation.hpp"
#include "orientation_parameterisation.hpp"
#include "cell_parameterisation.hpp"
#include <dx2/reflection.hpp>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;

class GradientsCalculator {
public:
    GradientsCalculator(
        OrientationParameterisation &uparam,
        CellParameterisation &bparam,
        const Goniometer &goniometer,
        BeamParameterisation& beamparam,
        DetectorParameterisation& Dparam) ;
    std::vector<std::vector<double>> get_gradients(const ReflectionTable &obs) const;

private:
  OrientationParameterisation uparam;
  CellParameterisation bparam;
  Goniometer goniometer;
  BeamParameterisation beamparam;
  DetectorParameterisation& Dparam;
};

#endif // BASELINE_REFINER_GRADIENTS_CALCULATOR_H