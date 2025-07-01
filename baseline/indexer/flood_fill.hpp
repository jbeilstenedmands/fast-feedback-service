#include <math.h>

#include <Eigen/Dense>

using Eigen::Vector3d;


/**
 * @brief Perform a flood fill algorithm on a grid of data to determine connected areas of signal.
 * @param grid The input array (grid) of data
 * @param rmsd_cutoff Filter out grid points below this cutoff value
 * @param n_points The size of each dimension of the FFT grid.
 * @returns A tuple of grid points per peak and centres of mass of the peaks in fractional coordinates.
 */
std::tuple<std::vector<int>, std::vector<Vector3d>> flood_fill(
  std::vector<double> const& grid,
  double rmsd_cutoff = 15.0,
  int n_points = 256);

/**
 * @brief Perform a filter on the flood fill results.
 * @param grid_points_per_void The number of grid points in each peak
 * @param centres_of_mass_frac The centres of mass of each peak, in fractional coordinates
 * @param peak_volume_cutoff The minimum fractional threshold for peaks to be included.
 * @returns A tuple of grid points per peak and centres of mass of the peaks in fractional coordinates.
 */
std::tuple<std::vector<int>, std::vector<Vector3d>> flood_fill_filter(
  std::vector<int> grid_points_per_void,
  std::vector<Vector3d> centres_of_mass_frac,
  double peak_volume_cutoff = 0.15);