#ifndef DIALS_RESEARCH_GGRADIENTS
#define DIALS_RESEARCH_GGRADIENTS
#include <rstbx/symmetry/constraints/a_g_conversion.h>
#include <cctbx/sgtbx/tensor_rank_2.h>
#include <cmath>

scitbx::af::shared<scitbx::mat3<double>> calc_dB_dg(rstbx::symmetry::AG Bconverter){
    // note we don't need the first three elements from the equivalent in g_gradients.py
    /*def get_all_da(self):
    # Aij : 9 elements of the reciprocal A* matrix [Rsymop paper, eqn(3)]
    # df_Aij: 9 gradients of the functional with respect to Aij elements

    # G=(g0,g1,g2,g3,g4,g5) 6 elements of the symmetrized metrical matrix,
    #   with current increments already applied ==(a*.a*,b*.b*,c*.c*,a*.b*,a*.c*,b*.c*)*/
    scitbx::sym_mat3<double> g = Bconverter.G;
    double g0 = g[0];
    double g1 = g[1];
    double g2 = g[2];
    double g3 = g[3];
    double g4 = g[4];
    double g5 = g[5];
    scitbx::af::shared<scitbx::mat3<double>> gradients {};

    //# the angles f = phi, p = psi, t = theta, along with convenient trig
    //# expressions
    double f = Bconverter.phi;
    double p = Bconverter.psi;
    double t = Bconverter.theta;

    double cosf = cos(f);
    double sinf = sin(f);
    double cosp = cos(p);
    double sinp = sin(p);
    double cost = cos(t);
    double sint = sin(t);

    double trig1 =  (cosf*cost*sinp)-(sinf*sint);
    double trig2 =  (cost*sinf)+(cosf*sinp*sint);
    double trig3 = (-1.0*cost*sinf*sinp)-(cosf*sint);
    double trig4 =  (cosf*cost)-(sinf*sinp*sint);

    //# Partial derivatives of the 9 A components with respect to metrical matrix elements
    double rad3 = g0-(((g2*g3*g3)+(g1*g4*g4)-(2*g3*g4*g5))/((g1*g2)-(g5*g5)));
    double sqrt_rad3 = sqrt(rad3);

    gradients.push_back({
      0.5*trig4/sqrt_rad3,
      0,
      0,
      0.5*cosp*sint/sqrt_rad3,
      0,
      0,
      0.5*trig2/sqrt_rad3,
      0,
      0
    });

    double fac4 = (g2*g3)-(g4*g5);
    double rad1 = g1-(g5*g5/g2);
    double rad1_three_half = sqrt(rad1*rad1*rad1);
    double fac3 = (g5*g5)-(g1*g2);
    double rad2 = -((g2*g3*g3) +g4*((g1*g4)-(2*g3*g5))+(g0*fac3))/((g1*g2)-(g5*g5));
    double factor_dg1 = (fac4*fac4)/(fac3*fac3*sqrt(rad2));
    double fac5 = g3-(g4*g5/g2);

    gradients.push_back({
     -0.5*(fac5*trig3/rad1_three_half) + 0.5*factor_dg1*trig4,
      0.5*trig3/sqrt(rad1),
      0,
     -0.5*(fac5*cosp*cost/rad1_three_half) + 0.5*factor_dg1*cosp*sint,
      0.5*cosp*cost/sqrt(rad1),
      0,
     -0.5*(fac5*trig1/rad1_three_half) + 0.5*factor_dg1*trig2,
      0.5*trig1/sqrt(rad1),
      0
    });

    double rat5_22 = g5/(g2*g2);
    double fac1 = g5*(g3-(g4*g5/g2));
    double fac2 = ((g1*g4)-(g3*g5));
    double fac2sq = fac2*fac2;

    gradients.push_back({
        -0.5*rat5_22*fac1*trig3/rad1_three_half + g4*rat5_22*trig3/sqrt(rad1) +
            0.5*fac2sq*trig4/(fac3*fac3*sqrt(rad2)) + 0.5*g4*cosp*sinf/pow(g2,1.5),
        0.5*rat5_22*(g5*trig3/sqrt(rad1)+sqrt(g2)*cosp*sinf),
        -0.5*cosp*sinf/sqrt(g2),
        -0.5*rat5_22*fac1*cosp*cost/rad1_three_half + g4*rat5_22*cosp*cost/sqrt(rad1) +
            0.5*g4*sinp/pow(g2,1.5) + 0.5*(fac2sq/fac3)*cosp*sint/(fac3*sqrt(rad2)),
        0.5*rat5_22*(g5*cosp*cost/sqrt(rad1)+sqrt(g2)*sinp),
        -0.5*sinp/sqrt(g2),
        -0.5*rat5_22*fac1*trig1/rad1_three_half + g4*rat5_22*trig1/sqrt(rad1) +
            0.5*fac2sq*trig2/(fac3*fac3*sqrt(rad2)) - 0.5*g4*cosf*cosp/pow(g2,1.5),
        0.5*rat5_22*(g5*trig1/sqrt(rad1)-sqrt(g2)*cosf*cosp),
        0.5*cosf*cosp/sqrt(g2)});

    gradients.push_back({
        trig3/sqrt(rad1) + fac4*trig4/(fac3*sqrt(rad2)),
        0,
        0,
        cosp*cost/sqrt(rad1) + fac4*cosp*sint/(fac3*sqrt(rad2)),
        0,
        0,
        trig1/sqrt(rad1) + fac4*trig2/(fac3*sqrt(rad2)),
        0,
        0});

    gradients.push_back({
        -g5*trig3/(g2*sqrt(rad1)) + fac2*trig4/(fac3*sqrt(rad2)) - cosp*sinf/sqrt(g2),
        0,
        0,
        -g5*cosp*cost/(g2*sqrt(rad1)) - sinp/sqrt(g2) + fac2*cosp*sint/(fac3*sqrt(rad2)),
        0,
        0,
        -g5*trig1/(g2*sqrt(rad1)) + fac2*trig2/(fac3*sqrt(rad2)) + cosf*cosp/sqrt(g2),
        0,
        0});

    double better_ratio = (fac2/fac3)*(fac4/fac3);

    gradients.push_back({
        fac1*trig3/(g2*rad1_three_half) -g4*trig3/(g2*sqrt(rad1)) +
        better_ratio*trig4/sqrt(rad2),
        -g5*trig3/(g2*sqrt(rad1)) - cosp*sinf/sqrt(g2),
        0,
        fac1*cosp*cost/(g2*rad1_three_half) - g4*cosp*cost/(g2*sqrt(rad1)) +
        better_ratio*cosp*sint/sqrt(rad2),
        -g5*cosp*cost/(g2*sqrt(rad1)) -sinp/sqrt(g2),
        0,
        fac1*trig1/(g2*rad1_three_half) - g4*trig1/(g2*sqrt(rad1)) +
        better_ratio*trig2/sqrt(rad2),
        -g5*trig1/(g2*sqrt(rad1))+cosf*cosp/sqrt(g2),
        0});
    return gradients;
}

scitbx::af::shared<scitbx::mat3<double>> dB_dp(
    rstbx::symmetry::AG Bconverter,
cctbx::sgtbx::tensor_rank_2::constraints<double> constraints_){
    int n_p = constraints_.n_independent_params();
    scitbx::af::shared<scitbx::mat3<double>> dB_dg_ = calc_dB_dg(Bconverter);
    if (n_p = 6){
        return dB_dg_;
    }
    scitbx::af::shared<scitbx::mat3<double>> dB_dp_(n_p);
    for (int i=0;i<9;i++){
        scitbx::sym_mat3<double> all_gradients = {
            dB_dg_[0][i],dB_dg_[1][i],dB_dg_[2][i],dB_dg_[3][i],dB_dg_[4][i],dB_dg_[5][i]
        };
        scitbx::af::small<double, 6> g_indep = constraints_.independent_gradients(all_gradients);
        for (int j=0;j<n_p;j++){
            dB_dp_[j][i] = g_indep[j];
        }
    }
    return dB_dp_;
}

#endif // DIALS_RESEARCH_GGRADIENTS