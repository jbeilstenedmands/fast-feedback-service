#pragma once
#include <dx2/beam.hpp>
#include <dx2/beam_ops.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/h5/h5read_processed.hpp>
#include <dx2/reflection.hpp>
#include <dx2/scan.hpp>

class MaximumLikelihoodTarget {
public:

    using ReflectionList = std::vector<ReflectionLikelihood>;

    MaximumLikelihoodTarget(
        const Simple6MosaicityParameterisation& model,
        const Eigen::Matrix3d& A,
        const Eigen::Vector3d& s0,
        const std::vector<Eigen::Vector3d>& xyzcal_px,
        const std::vector<Eigen::Vector3d>& xyzobs_px,
        const std::vector<Eigen::Vector3d>& covariances,
        const std::vector<double>& intensities,
        const std::vector<Eigen::Vector3i>& miller_indices,
        const std::vector<Eigen::Vector2d>& mobs,
        const Panel& panel);

    MaximumLikelihoodTarget(
        const Simple6MosaicityParameterisation& model,
        const Eigen::Matrix3d& A,
        const Eigen::Vector3d& s0,
        const std::vector<Eigen::Vector3d>& sp_list,
        const std::vector<Eigen::Vector3d>& covariances,
        const std::vector<double>& intensities,
        const std::vector<Eigen::Vector3i>& miller_indices,
        const std::vector<Eigen::Vector2d>& mobs);

    void update();

    double mse() const;

    double log_likelihood() const;

    ParameterVector first_derivatives() const;

    FisherMatrix fisher_information() const;

    const ReflectionList& reflections() const {
        return data_;
    }

private:

    const Simple6MosaicityParameterisation& model_;

    ReflectionList data_;
};

MaximumLikelihoodTarget::MaximumLikelihoodTarget(
    const Simple6MosaicityParameterisation& model,
    const Eigen::Matrix3d& A,
    const Eigen::Vector3d& s0,
    const std::vector<Eigen::Vector3d>& sp_list,
    const std::vector<Eigen::Vector3d>& covariances,
    const std::vector<double>& intensities,
    const std::vector<Eigen::Vector3i>& miller_indices,
    const std::vector<Eigen::Vector2d>& mobs)
    :
    model_(model)
    
{
    const std::size_t n = miller_indices.size();
    data_.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {
        Eigen::Matrix2d sobs;
        sobs << covariances[i][0], covariances[i][2],covariances[i][2], covariances[i][1];
        data_.emplace_back(
            model_,
            A,
            s0,
            sp_list[i],
            miller_indices[i],
            intensities[i],
            mobs[i],
            sobs);
    }
}

MaximumLikelihoodTarget::MaximumLikelihoodTarget(
    const Simple6MosaicityParameterisation& model,
    const Eigen::Matrix3d& A,
    const Eigen::Vector3d& s0,
    const std::vector<Eigen::Vector3d>& xyzcal_px,
    const std::vector<Eigen::Vector3d>& xyzobs_px,
    const std::vector<Eigen::Vector3d>& covariances,
    const std::vector<double>& intensities,
    const std::vector<Eigen::Vector3i>& miller_indices,
    const std::vector<Eigen::Vector2d>& mobs,
    const Panel& panel)
    :
    model_(model)
    
{
    const std::size_t n = miller_indices.size();
    double s0_length = s0.norm();
    /*assert xyzcal_mm.size() == n;
    DIALS_ASSERT(xyzobs_mm.size() == n);
    DIALS_ASSERT(covariances.size() == n);
    DIALS_ASSERT(intensities.size() == n);*/

    data_.reserve(n);

    for (std::size_t i = 0; i < n; ++i) {

        // FIXME is mobs, s1 in mm?
        auto [xmm, ymm] = panel.px_to_mm(xyzcal_px[i][0], xyzcal_px[i][1]);
        Vector3d s1cal = panel.get_lab_coord(xmm, ymm);
        s1cal.normalize();
        s1cal = s1cal * s0_length;
        auto [xomm, yomm] = panel.px_to_mm(xyzobs_px[i][0], xyzobs_px[i][1]);
        Vector3d s1obs = panel.get_lab_coord(xomm, yomm);
        s1obs.normalize();
        s1obs = s1obs * s0_length;

        //Eigen::Vector2d mobs = s1obs.head<2>();
        Eigen::Matrix2d sobs;
        sobs << covariances[i][0], covariances[i][2],covariances[i][2], covariances[i][1];

        //std::cout << "Sobs " << sobs << " mobs " << mobs << " s1cal " << s1cal << std::endl;

        data_.emplace_back(
            model_,
            A,
            s0,
            s1cal,
            miller_indices[i],
            intensities[i],
            mobs[i],
            sobs);
    }
}

void MaximumLikelihoodTarget::update()
{
    for (auto& r : data_) {
        r.update();
    }
}

double MaximumLikelihoodTarget::log_likelihood() const
{
    double lnL = 0.0;

    for (const auto& r : data_) {
        lnL += r.log_likelihood();
    }

    return lnL;
}

double MaximumLikelihoodTarget::mse() const
{
    double mse = 0.0;

    for (const auto& r : data_) {

        Eigen::Vector2d diff =
            r.mobs() -
            r.conditional().mean();

        mse += diff.squaredNorm();
    }

    return mse / static_cast<double>(data_.size());
}

ParameterVector MaximumLikelihoodTarget::first_derivatives() const
{
    ParameterVector dL =
        ParameterVector::Zero();

    for (const auto& r : data_) {
        dL += r.first_derivatives();
    }

    return dL;
}

FisherMatrix
MaximumLikelihoodTarget::fisher_information() const
{
    FisherMatrix I =
        FisherMatrix::Zero();

    for (const auto& r : data_) {
        I += r.fisher_information();
    }

    return I;
}
