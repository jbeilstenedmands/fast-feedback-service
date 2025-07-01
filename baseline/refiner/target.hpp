#ifndef REFINE_TARGET_H
#define REFINE_TARGET_H
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include "detector_parameterisation.hpp"
#include "beam_parameterisation.hpp"
#include "orientation_parameterisation.hpp"
#include "cell_parameterisation.hpp"
#include <dx2/reflection.hpp>
#include "gradients_calculator.hpp"
#include <cmath>


class Target {
public:
    Target(
        Crystal &crystal,
        const Goniometer &goniometer,
        MonochromaticBeam& beam,
        Panel& panel,
        ReflectionTable& obs);
    std::vector<std::vector<double>> gradients() const;
    std::vector<double> residuals(std::vector<double> x);
    int nref() const;
    int nparams() const;
    CellParameterisation cell_parameterisation() const;
    OrientationParameterisation orientation_parameterisation() const;
    DetectorParameterisation detector_parameterisation() const;
    BeamParameterisation beam_parameterisation() const;
    std::vector<double> rmsds() const;

private:
    Crystal crystal;
    Goniometer goniometer;
    MonochromaticBeam beam;
    Panel& panel_;
    ReflectionTable& obs;
    CellParameterisation cellparam;
    OrientationParameterisation orientationparam;
    DetectorParameterisation detectorparam;
    BeamParameterisation beamparam;
    GradientsCalculator calculator;
    int n_ref;
    int n_params;
    std::vector<double> rmsds_ = {0.0,0.0,0.0};
};

#endif //REFINE_TARGET_H