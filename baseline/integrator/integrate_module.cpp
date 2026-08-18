#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <experimental/mdspan>
#include "integrator/sigma_estimation.hpp"

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

NB_MODULE(integrate, m) {
    m.def("max_separation_filter", &max_separation_filter, "Filter based on XY rmsds");
    m.def("estimate_rmsd_sigma", &estimate_rmsd_sigma, "Estimate sigmab from XY rmsds");
}