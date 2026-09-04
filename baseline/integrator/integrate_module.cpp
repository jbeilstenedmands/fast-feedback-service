#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <experimental/mdspan>
#include "integrator/sigma_estimation.hpp"
#include "ellipsoid_parameterisation.hpp"
#include "calculations.hpp"
#include "fisher_scoring_ml.hpp"
#include "target.hpp"
#include <iostream>
#include "ssx_integrate.hpp"

using DerivativeMatrices = std::array<Eigen::Matrix3d, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix3d = Eigen::Matrix3d;
using Vector2d = Eigen::Vector2d;

template <typename T>
using mdspan_type =
  std::experimental::mdspan<T, std::experimental::dextents<size_t, 2>>;

// Using xyzobs.px and xyzcal.px, filter based on a separation value and return the selection.
std::vector<bool> max_separation_filter(
    std::vector<double> xyzobs_px_data,
    std::vector<double> xyzcal_px_data,
    double max_separation
){
    mdspan_type<double> xyzobs_px =
      mdspan_type<double>(xyzobs_px_data.data(), xyzobs_px_data.size() / 3, 3);
    mdspan_type<double> xyzcal_px =
      mdspan_type<double>(xyzcal_px_data.data(), xyzcal_px_data.size() / 3, 3);
    std::vector<bool> selection(xyzcal_px_data.size(), true);
    for (int i=0;i<xyzcal_px.extent(0);i++){
        if (std::pow(
            std::pow(xyzcal_px(i,0) - xyzobs_px(i,0),2) +
            std::pow(xyzcal_px(i,1) - xyzobs_px(i,1),2), 0.5) > max_separation){
                selection[i*3] = false;
                selection[i*3+1] = false;
                selection[i*3+2] = false;
            }
    }
    return selection;
}

NB_MODULE(integrate, m) {
    m.def("max_separation_filter", &max_separation_filter, "Filter based on XY rmsds");
    m.def("ssx_integrate", &ssx_integrate, "ssx integrate");
}