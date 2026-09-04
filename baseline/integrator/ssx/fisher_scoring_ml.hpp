#pragma once

#include <Eigen/Core>
#include <vector>
#include "ellipsoid_parameterisation.hpp"
#include "target.hpp"

using Matrix3d = Eigen::Matrix3d;
using Matrix2d = Eigen::Matrix2d;
using Vector2d = Eigen::Vector2d;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat2 = Eigen::Matrix2d;
using Mat3 = Eigen::Matrix3d;
using ParameterVector = Eigen::Matrix<double, 6, 1>;
using FisherMatrix = Eigen::Matrix<double, 6, 6>;
using DerivativeMatrices = std::array<Eigen::Matrix3d, 6>;
using DerivativeVectors = std::array<Vec3, 6>;
using SigmaDerivativeMatrices = std::array<Eigen::Matrix2d, 6>;
using MuDerivativeVectors = std::array<Eigen::Vector2d, 6>;


struct RefinementStep {
    ParameterVector parameters;
    double log_likelihood;
    double mse;
};

class FisherScoringMaximumLikelihood {
public:

    FisherScoringMaximumLikelihood(
        Simple6MosaicityParameterisation& model,
        MaximumLikelihoodTarget& target,
        int max_iter = 1000,
        double tolerance = 1e-7,
        double ll_tolerance = 1e-6)
        :
        model_(model),
        target_(target),
        parameters_(model.parameters()),
        max_iter_(max_iter),
        tolerance_(tolerance),
        ll_tolerance_(ll_tolerance)
    {}

    void solve();

    const ParameterVector& parameters() const {
        return parameters_;
    }

    const std::vector<RefinementStep>& history() const {
        return history_;
    }

private:

    double log_likelihood(
        const ParameterVector& x);

    ParameterVector score(
        const ParameterVector& x);

    std::pair<ParameterVector,FisherMatrix>
    score_and_fisher_information(
        const ParameterVector& x);

    ParameterVector solve_update_equation(
        const ParameterVector& S,
        const FisherMatrix& I) const;

    double line_search(
        const ParameterVector& x,
        const ParameterVector& p,
        double tau = 0.5,
        double delta = 1.0);

    ParameterVector gradient_search(
        const ParameterVector& x);

    bool test_LL_convergence() const;

    void callback(
        const ParameterVector& x);

private:

    Simple6MosaicityParameterisation& model_;
    MaximumLikelihoodTarget& target_;

    ParameterVector parameters_;

    int max_iter_;
    double tolerance_;
    double ll_tolerance_;

    std::vector<RefinementStep> history_;
};

double FisherScoringMaximumLikelihood::log_likelihood(
    const ParameterVector& x)
{
    model_.set_parameters(x);

    target_.update();

    return target_.log_likelihood();
}

ParameterVector
FisherScoringMaximumLikelihood::score(
    const ParameterVector& x)
{
    model_.set_parameters(x);

    target_.update();

    return target_.first_derivatives();
}

std::pair<ParameterVector,FisherMatrix>
FisherScoringMaximumLikelihood::score_and_fisher_information(
    const ParameterVector& x)
{
    model_.set_parameters(x);

    target_.update();

    return {
        target_.first_derivatives(),
        target_.fisher_information()
    };
}

ParameterVector
FisherScoringMaximumLikelihood::solve_update_equation(
    const ParameterVector& S,
    const FisherMatrix& I) const
{
    return I.ldlt().solve(S);
}

double FisherScoringMaximumLikelihood::line_search(
    const ParameterVector& x,
    const ParameterVector& p,
    double tau,
    double delta)
{
    const double fa =
        -log_likelihood(x);

    const double min_delta =
        tolerance_ /
        std::max(1.0, p.norm());

    while (delta > min_delta) {

        try {

            const double fb =
                -log_likelihood(
                    x + delta * p);

            if (fb <= fa) {
                return delta;
            }

        } catch (...) {
        }

        delta *= tau;
    }

    return 0.0;
}

ParameterVector
FisherScoringMaximumLikelihood::gradient_search(
    const ParameterVector& x)
{
    const ParameterVector g =
        score(x);

    double alpha = 1e-3;

    return x + alpha * g;
}

bool
FisherScoringMaximumLikelihood::test_LL_convergence() const
{
    if (history_.size() < 2) {
        return false;
    }

    const double l1 =
        history_[history_.size()-1].log_likelihood;

    const double l2 =
        history_[history_.size()-2].log_likelihood;

    return std::abs(l1 - l2)
        < ll_tolerance_;
}

void FisherScoringMaximumLikelihood::callback(
    const ParameterVector& x)
{
    model_.set_parameters(x);

    target_.update();

    RefinementStep step;

    step.parameters = x;

    step.log_likelihood = target_.log_likelihood();

    step.mse = target_.mse();

    history_.push_back(
        std::move(step));
}

void FisherScoringMaximumLikelihood::solve()
{
    ParameterVector x =
        parameters_;

    for (int iter = 0;
         iter < max_iter_;
         ++iter)
    {
        auto [S,I] =
            score_and_fisher_information(x);

        ParameterVector p =
            solve_update_equation(S,I);

        double delta =
            line_search(x,p);

        ParameterVector x_new;

        if (delta > 0.0) {
            x_new =
                x + delta * p;
        }
        else {
            x_new =
                gradient_search(x);
        }

        callback(x_new);

        if ((x_new - x).norm()
            < tolerance_)
        {
            x = x_new;
            break;
        }

        if (test_LL_convergence()) {
            x = x_new;
            break;
        }

        x = x_new;
    }

    parameters_ = x;

    model_.set_parameters(parameters_);

    target_.update();
}