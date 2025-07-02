#include <Eigen/Dense>
#include <cmath>
#include <experimental/mdspan>
#include "assign_indices.hpp"

using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;



assign_indices_results::assign_indices_results(int extent)
        : miller_indices_data(extent * 3),
          miller_indices(miller_indices_data.data(), extent, 3),
          number_indexed(0) {}

