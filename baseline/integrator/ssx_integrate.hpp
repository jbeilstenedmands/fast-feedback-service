#include <experimental/mdspan>
#include "ellipsoid_parameterisation.hpp"
#include "integrator/sigma_estimation.hpp"
#include "calculations.hpp"
#include "fisher_scoring_ml.hpp"
#include "target.hpp"
#include <iostream>
#include <cmath>
#include <dx2/beam.hpp>
#include <dx2/beam_ops.hpp>
#include <math/math_utils.cuh>
#include <stdexcept>
#include <vector>

#include "ffs_logger.hpp"

using DerivativeMatrices = std::array<Eigen::Matrix3d, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix3d = Eigen::Matrix3d;
using Vector2d = Eigen::Vector2d;

template <typename T>
using mdspan_type =
  std::experimental::mdspan<T, std::experimental::dextents<size_t, 2>>;

void ssx_integrate(const std::vector<Vector3d> xyzcal_px,
    const std::vector<Vector3d> xyzobs_px,
    const std::vector<Vector3d> covariances,
    const std::vector<double> intensities,
    const std::vector<Eigen::Vector3i> miller_indices,
    const std::vector<Vector2d> mobs,
    const Vector3d &s0,
    const Panel &panel,
    const Matrix3d A){
    double tot_sigma_b = 0.0;
    int n = xyzcal_px.size();
    for (int i=0;i<n;i++){
        tot_sigma_b += (covariances[i][0] + covariances[i][1])/2.0;
    }
    double sigma_b_spot = std::pow(tot_sigma_b / n, 0.5);
    double sigma_b_rmsd = estimate_sigmab_2d(
            xyzcal_px, xyzobs_px,s0,
            panel);
    double overall_sigma_b = std::pow(std::pow(sigma_b_rmsd, 2) + std::pow(sigma_b_spot,2), 0.5);
    // for the sigma6 mosaicity model, sigma_b is used as the starting point for the diagonal terms
    // in the matrix

    // Note model must outlive MLTarget due to reference.
    Simple6MosaicityParameterisation model = Simple6MosaicityParameterisation::from_sigma_d(overall_sigma_b);
    double s0_length = s0.norm();
    const std::size_t n1 = miller_indices.size();
    std::vector<Vector3d> sp_list;
    for (std::size_t i = 0; i < n1; ++i) {
        auto [xmm, ymm] = panel.px_to_mm(xyzobs_px[i][0], xyzobs_px[i][1]);
        Vector3d sp_i = panel.get_lab_coord(xmm, ymm);
        sp_i.normalize();
        sp_i = sp_i * s0_length;
        sp_list.push_back(sp_i);
    }

    MaximumLikelihoodTarget target(
        model,
        A,
        s0,
        sp_list,
        covariances,
        intensities,
        miller_indices,
        mobs
    );
    FisherScoringMaximumLikelihood scorer = FisherScoringMaximumLikelihood(model, target);
    scorer.solve();
    Matrix3d sigma = model.sigma();
    print_eigen_values_and_vectors_static(sigma);
}