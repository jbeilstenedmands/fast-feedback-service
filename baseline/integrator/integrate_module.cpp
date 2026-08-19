#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <experimental/mdspan>
#include "integrator/sigma_estimation.hpp"
#include "ellipsoid_parameterisation.hpp"
//#include "model_state.hpp"
#include "calculations.hpp"
#include "target.hpp"
#include <iostream>

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

// Names chosen to match those used in dials.
class RefinerData {
//A class for holding the data needed for the profile refinement
public:
RefinerData(
    std::array<double, 3> s0,
    std::vector<double> sp_list,
    std::vector<int> h_list,
    std::vector<double> ctot_list,
    std::vector<double> mobs_list,
    std::vector<double> sobs_list) // assume single panel detector for now
    :   s0_(std::move(s0)),
        sp_list_(std::move(sp_list)),h_list_(std::move(h_list)),
        ctot_list_(std::move(ctot_list)),
        mobs_list_(std::move(mobs_list)),
        sobs_list_(std::move(sobs_list)) {
            damp_outlier_intensity_weights();
        }

private:
    std::array<double, 3> s0_;
    std::vector<double> sp_list_;
    std::vector<int> h_list_;
    std::vector<double> ctot_list_;
    std::vector<double> mobs_list_;
    std::vector<double> sobs_list_;

    void damp_outlier_intensity_weights() {
        if (ctot_list_.empty()) {
            return;
        }
        auto sorted = ctot_list_;
        std::ranges::sort(sorted);
        const std::size_t n = sorted.size();
        const double q1 = sorted[n / 4];
        const double q3 = sorted[(3 * n) / 4];
        const double iqr = q3 - q1;
        const double threshold = q3 + 1.5 * iqr;
        for (auto& value : ctot_list_) {
            if (value > threshold) {
                value = threshold;
            }
        }
    }
};

double estimate_rmsd_sigma(const std::vector<Vector3d> xyzcal,
    const std::vector<Vector3d> xyzobs,
    const Vector3d &s0,
    const Panel &panel){
    return estimate_sigmab_2d(xyzcal, xyzobs,s0,panel);
}

/*class ModelState {
public:

  Eigen::VectorXd active_parameters() const;

  void set_active_parameters(const Eigen::VectorXd&);

  Eigen::Matrix3d mosaicity_covariance_matrix() const;

  std::vector<Eigen::Matrix3d> dM_dp() const;

private:
  Simple6MosaicityParameterisation parameterisation_;
};*/



void ssx_integrate(const std::vector<Vector3d> xyzcal_px,
    const std::vector<Vector3d> xyzobs_px,
    const std::vector<Vector3d> covariances,
    const std::vector<double> intensities,
    const std::vector<Eigen::Vector3i> miller_indices,
     const Vector3d &s0,
    const Panel &panel){
    double tot_sigma_b = 0.0;
    int n = xyzcal_px.size();
    for (int i=0;i<n;i++){
        tot_sigma_b += (covariances[i][0] + covariances[i][1])/2.0;
    }
    double sigma_b_spot = std::pow(tot_sigma_b / n, 0.5);
    double sigma_b_rmsd = estimate_rmsd_sigma(
            xyzcal_px, xyzobs_px,s0,
            panel);
    double overall_sigma_b = std::pow(std::pow(sigma_b_rmsd, 2) + std::pow(sigma_b_spot,2), 0.5);
    std::cout << sigma_b_spot << std::endl;
    std::cout << sigma_b_rmsd << std::endl;
    std::cout << overall_sigma_b << std::endl;

    // for the sigma6 mosaicity model, sigma_b is used as the starting point for the diagonal terms
    // in the matrix

    // for refinerdata, we need miller index too.
    // in dials, refinerdata mainly reshapes the data and also damps the outlier weights. It is basically
    // a data container
    /*# Create the parameterisation
    state = ModelState(
        experiment,
        profile.parameterisation.parameterisation(),
        fix_orientation=True,
        fix_unit_cell=True,
        fix_wavelength_spread=wavelength_spread_model == "delta",
    )

    # Create the refiner and refine
    refiner = ProfileRefiner(state, refiner_data, max_iter, LL_tolerance)
    refiner.refine()

    # Set the profile parameters
    profile.parameterisation.update_model(refiner.state)
    # Set the mosaicity
    experiment.crystal.mosaicity = profile*/
    Matrix3d A;
    A << -0.004379, -0.008665,  0.008310,
        0.012045, -0.003290,  0.002932,
        0.000628,  0.008554,  0.009578;
    // Note model must outlive MLTarget due to reference.
    Simple6MosaicityParameterisation model = Simple6MosaicityParameterisation::from_sigma_d(overall_sigma_b);
    MaximumLikelihoodTarget target = MaximumLikelihoodTarget(
        model,
        A,
        s0,
        xyzcal_px,
        xyzobs_px,
        covariances,
        intensities,
        miller_indices,
        panel
    );
    /*auto profile =
    Simple6ProfileModel::from_sigma_d(overall_sigma_b);

    auto sigma =
        profile.sigma();

    auto derivs =
        profile.first_derivatives();

    auto mosaicity =
        profile.mosaicity();

    auto modelstate = ModelState(profile);*/
    std::cout << "here" << std::endl;
    

}



NB_MODULE(integrate, m) {
    m.def("max_separation_filter", &max_separation_filter, "Filter based on XY rmsds");
    m.def("ssx_integrate", &ssx_integrate, "ssx integrate");
}