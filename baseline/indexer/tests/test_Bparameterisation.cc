#include "refinement/Bparameterisation.h"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <dx2/crystal.h>
#include <iostream>

using Eigen::Matrix3d;
using Eigen::Vector3d;
using json = nlohmann::json;

TEST(BaselineIndexer, Bparameterisationtest) {
    Vector3d a = {10,1.0,1.0};
    Vector3d b = {1.0,10.0,1.0};
    Vector3d c = {2.0,3.0,10.0};
    gemmi::SpaceGroup space_group(1);
    Crystal crystal{a,b,c,space_group};
    std::cout << crystal.get_B_matrix() << std::endl;
    SimpleBParameterisation B_param{crystal};
    std::vector<double> params = B_param.get_params();
    for (int i=0;i<6;i++){
        std::cout << params[i] << std::endl;
    }
    std::vector<double> expected_params{};
    std::vector<Matrix3d> dS_dp = B_param.get_dS_dp();
    for (int i=0;i<6;i++){
        std::cout << dS_dp[i] << std::endl;
    }
    std::vector<double> new_params{1.0,2.0,3.0,0.5,0.2,0.3};
    std::vector<double> new_params_vals = B_param.get_params();
    for (int i=0;i<6;i++){
        std::cout << new_params_vals[i] << std::endl;
    }
    std::vector<Matrix3d> new_dS_dp = B_param.get_dS_dp();
    for (int i=0;i<6;i++){
        std::cout << new_dS_dp[i] << std::endl;
    }
}