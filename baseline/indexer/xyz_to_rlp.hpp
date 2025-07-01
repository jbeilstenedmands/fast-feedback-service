#ifndef BASELINE_INDEXER_XYZ_TO_RLP_H
#define BASELINE_INDEXER_XYZ_TO_RLP_H
#include <math.h>

#include <dx2/beam.hpp>
#include <dx2/detector.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/scan.hpp>
#include <experimental/mdspan>
#include <tuple>


template <typename T>
using mdspan_type =
  std::experimental::mdspan<T, std::experimental::dextents<size_t, 2>>;

struct xyz_to_rlp_results {
    std::vector<double> rlp_data;
    std::vector<double> s1_data;
    std::vector<double> xyzobs_mm_data;
    mdspan_type<double> rlp;
    mdspan_type<double> s1;
    mdspan_type<double> xyzobs_mm;

    xyz_to_rlp_results(int extent);
};

/**
 * @brief Transform detector pixel coordinates into reciprocal space coordinates.
 * @param xyzobs_px A 1D array of detector pixel coordinates from a single panel.
 * @param panel A dx2 Panel object defining the corresponding detector panel.
 * @param beam A dx2 MonochromaticBeam object.
 * @param scan A dx2 Scan object.
 * @param gonio A dx2 Goniometer object.
 * @returns A struct containing reciprocal space coordinates, s1 vectors and pixel coordinates in mm.
 */
xyz_to_rlp_results xyz_to_rlp(const mdspan_type<double> &xyzobs_px,
                              const Panel &panel,
                              const MonochromaticBeam &beam,
                              const Scan &scan,
                              const Goniometer &gonio);

void px_to_mm(
  const mdspan_type<double> &px_input,
  mdspan_type<double> &mm_output,
  const Scan& scan,
  const Panel& panel
);

#endif // BASELINE_INDEXER_XYZ_TO_RLP_H