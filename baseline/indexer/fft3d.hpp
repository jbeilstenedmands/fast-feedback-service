#ifndef BASELINE_INDEXER_FFT3D_H
#define BASELINE_INDEXER_FFT3D_H
#include <assert.h>
#include <math.h>

#include <algorithm>
#include <chrono>
#include <experimental/mdspan>
#include <map>
#include <stack>
#include <tuple>

#define _USE_MATH_DEFINES
#include <cmath>

template <typename T>
using mdspan_type =
  std::experimental::mdspan<T, std::experimental::dextents<size_t, 2>>;

/**
 * @brief map reciprocal space vectors onto a grid of size n_points^3.
 * @param reciprocal_space_vectors Reciprocal space vectors to be mapped.
 * @param data_in The vector (grid) which the data will be mapped to.
 * @param selection The vector of the selection of points mapped to the grid.
 * @param d_min A resolution limit for mapping to the grid.
 * @param b_iso The isotropic B-factor used to weight the points as a function of resolution.
 * @param n_points The size of each dimension of the FFT grid.
 */
void map_centroids_to_reciprocal_space_grid(
  mdspan_type<double> const &reciprocal_space_vectors,
  std::vector<std::complex<double>> &data_in,
  std::vector<bool> &selection,
  double d_min,
  double b_iso = 0,
  uint32_t n_points = 256);

/**
 * @brief Perform a 3D FFT of the reciprocal space coordinates (spots).
 * @param reciprocal_space_vectors The input vector of reciprocal space coordinates.
 * @param real_out The (real) array that the FFT result will be written to.
 * @param d_min Cut the data at this resolution limit for the FFT
 * @param b_iso The isotropic B-factor used to weight the points as a function of resolution.
 * @param n_points The size of each dimension of the FFT grid.
 * @returns A boolean array indicating which coordinates were used for the FFT.
 */
std::vector<bool> fft3d(mdspan_type<double> const &reciprocal_space_vectors,
                        std::vector<double> &real_out,
                        double d_min,
                        double b_iso = 0,
                        uint32_t n_points = 256,
                        size_t nthreads = 1);

#endif // BASELINE_INDEXER_FFT3D_H
