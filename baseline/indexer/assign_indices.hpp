#ifndef ASSIGN_INDICES_H
#define ASSIGN_INDICES_H
#include <Eigen/Dense>
#include <cmath>
#include <experimental/mdspan>

using Eigen::Matrix3d;

template <typename T>
using mdspan_type =
  std::experimental::mdspan<T, std::experimental::dextents<size_t, 2>>;


struct assign_indices_results {
    std::vector<int> miller_indices_data;
    mdspan_type<int> miller_indices;
    int number_indexed;

    assign_indices_results(int extent);
};

/**
 * @brief Assigns miller indices to reciprocal lattice points.
 * @param A The crystal A-matrix.
 * @param rlp The vector of reciprocal lattice points.
 * @param xyzobs_mm The vector of observed xyz positions, in mm.
 * @param tolerance The tolerance within which the fractional miller index must be for acceptance.
 * @returns A struct containing the assigned miller indices and the number of reciprocal lattice points successfully indexed.
 */
assign_indices_results assign_indices_global(Matrix3d const &A,
                                             mdspan_type<double> const &rlp,
                                             mdspan_type<double> const &xyzobs_mm,
                                             const double tolerance = 0.3);

#endif