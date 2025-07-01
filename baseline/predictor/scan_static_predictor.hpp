#ifndef DIALS_STATIC_PREDICTOR
#define DIALS_STATIC_PREDICTOR
#include <Eigen/Dense>
#include <dx2/beam.hpp>
#include <dx2/detector.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>

using Eigen::Matrix3d;
using Eigen::Vector3d;

void simple_reflection_predictor(const MonochromaticBeam beam,
                                 const Goniometer gonio,
                                 const Matrix3d UB,
                                 const Panel& panel,
                                 ReflectionTable& reflections);

Vector3d unit_rotate_around_origin(Vector3d vec, Vector3d unit, double angle);

#endif  // DIALS_STATIC_PREDICTOR