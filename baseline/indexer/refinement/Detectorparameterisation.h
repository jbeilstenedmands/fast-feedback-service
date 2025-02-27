#ifndef DIALS_RESEARCH_DPARAM
#define DIALS_RESEARCH_DPARAM
#include <dxtbx/model/detector.h>
#include <scitbx/vec3.h>
#include <cmath>
#include <dxtbx/model/experiment.h>
#include <dials_research/fast_dp2/indexing/Beamparameterisation.h>

class SimpleDetectorParameterisation {
public:
  SimpleDetectorParameterisation(
    const dxtbx::model::Detector &Detector,
    bool fix_dist, bool fix_shift1, bool fix_shift2,
    bool fix_tau1, bool fix_tau2, bool fix_tau3);
  scitbx::af::shared<double> get_params();
  void set_params(scitbx::af::shared<double> p);
  scitbx::mat3<double> get_state();
  scitbx::af::shared<scitbx::mat3<double>> get_dS_dp();
  bool dist_fixed();
  bool shift1_fixed();
  bool shift2_fixed();
  bool tau1_fixed();
  bool tau2_fixed();
  bool tau3_fixed();

private:
  scitbx::af::shared<double> params{6, 0.0}; //
  void compose();
  scitbx::af::shared<scitbx::mat3<double>> dS_dp{
    6,
    scitbx::mat3<double>(0,0, 0, 0,0,0, 0,0,0)};
  scitbx::vec3<double> initial_offset{0.0,0.0,0.0};
  scitbx::vec3<double> initial_d1{0.0,0.0,0.0};
  scitbx::vec3<double> initial_d2{0.0,0.0,0.0};
  scitbx::vec3<double> initial_dn{0.0,0.0,0.0};
  scitbx::vec3<double> initial_origin{0.0,0.0,0.0};
  scitbx::vec3<double> current_origin{0.0,0.0,0.0};
  scitbx::vec3<double> current_d1{0.0,0.0,0.0};
  scitbx::vec3<double> current_d2{0.0,0.0,0.0};
  bool _fix_dist{true};
  bool _fix_shift1{false};
  bool _fix_shift2{true};
  bool _fix_tau1{true};
  bool _fix_tau2{true};
  bool _fix_tau3{true};
};

void SimpleDetectorParameterisation::compose(){
    double t1r = params[3] / 1000.0;
    double t2r = params[4] / 1000.0;
    double t3r = params[5] / 1000.0;
    scitbx::mat3<double> Tau1 = axis_and_angle_as_rot(initial_dn, t1r);
    scitbx::mat3<double> dTau1_dtau1 = dR_from_axis_and_angle(initial_dn, t1r);
    scitbx::mat3<double> Tau2 = axis_and_angle_as_rot(initial_d1, t2r);
    scitbx::mat3<double> dTau2_dtau2 = dR_from_axis_and_angle(initial_d1, t2r);
    scitbx::mat3<double> Tau3 = axis_and_angle_as_rot(initial_d2, t3r);
    scitbx::mat3<double> dTau3_dtau3 = dR_from_axis_and_angle(initial_d2, t3r);
    scitbx::mat3<double> Tau32 = Tau3 * Tau2;
    scitbx::mat3<double> Tau321 = Tau32 * Tau1;
    scitbx::vec3<double> P0 = params[0] * initial_dn;
    scitbx::vec3<double> Px = P0 + initial_d1;
    scitbx::vec3<double> Py = P0 + initial_d2;
    scitbx::vec3<double> dsv = P0 + (params[1] * initial_d1) + (params[2] * initial_d2);
    scitbx::vec3<double> dorg = (Tau321 * dsv) - (Tau32 * P0) + P0;
    scitbx::vec3<double> d1 = (Tau321 * (Px-P0)).normalize();
    scitbx::vec3<double> d2 = (Tau321 * (Py-P0)).normalize();
    scitbx::vec3<double> dn = d1.cross(d2).normalize();
    current_d1 = d1;
    current_d2 = d2;

    //scitbx::vec3<double> d2 = dn.cross(d1);
    scitbx::vec3<double> o = dorg + initial_offset[0] * d1 + initial_offset[1] * d2;
    //new_state = {"d1": d1, "d2": d2, "origin": o}
    current_origin = o;

    // derivative of dorg wrt dist
    scitbx::vec3<double> ddorg_ddist = (Tau321*initial_dn) - (Tau32*initial_dn) + initial_dn;
    scitbx::vec3<double> ddorg_dshift1 = Tau321 * initial_d1;
    scitbx::vec3<double> ddorg_dshift2 = Tau321 * initial_d2;
    // derivative wrt tau1, tau2, tau3
    scitbx::mat3<double> dTau321_dtau1 = Tau32 * dTau1_dtau1;
    scitbx::vec3<double> ddorg_dtau1 = dTau321_dtau1 * dsv;
    scitbx::mat3<double> dTau32_dtau2 = Tau3 * dTau2_dtau2;
    scitbx::mat3<double> dTau321_dtau2 = dTau32_dtau2 * Tau1;
    scitbx::vec3<double> ddorg_dtau2 = dTau321_dtau2 * dsv - dTau32_dtau2 * P0;
    scitbx::mat3<double> dTau32_dtau3 = dTau3_dtau3 * Tau2;
    scitbx::mat3<double> dTau321_dtau3 = dTau32_dtau3 * Tau1;
    scitbx::vec3<double> ddorg_dtau3 = dTau321_dtau3 * dsv - dTau32_dtau3 * P0;

    // Now derivatives of the direction d1, where d1 = (Tau321 * (Px - P0)).normalize()
    //scitbx::vec3<double> dd1_ddist{0.0, 0.0, 0.0};
    //scitbx::vec3<double> dd1_dshift1{0.0, 0.0, 0.0};
    //scitbx::vec3<double> dd1_dshift2{0.0, 0.0, 0.0};
    scitbx::vec3<double> dd1_dtau1 = dTau321_dtau1 * (Px - P0);
    scitbx::vec3<double> dd1_dtau2 = dTau321_dtau2 * (Px - P0);
    scitbx::vec3<double> dd1_dtau3 = dTau321_dtau3 * (Px - P0);

    // Derivatives of the direction d2, where d2 = (Tau321 * (Py - P0)).normalize()
    //scitbx::vec3<double> dd2_ddist{0.0, 0.0, 0.0};
    //scitbx::vec3<double> dd2_dshift1{0.0, 0.0, 0.0};
    //scitbx::vec3<double> dd2_dshift2{0.0, 0.0, 0.0};
    scitbx::vec3<double> dd2_dtau1 = dTau321_dtau1 * (Py - P0);
    scitbx::vec3<double> dd2_dtau2 = dTau321_dtau2 * (Py - P0);
    scitbx::vec3<double> dd2_dtau3 = dTau321_dtau3 * (Py - P0);

    // Derivatives of the direction dn, where dn = d1.cross(d2).normalize()
    // These derivatives are not used
    scitbx::vec3<double> do_ddist = ddorg_ddist;// + initial_offset[0] * dd1_ddist + initial_offset[1] * dd2_ddist;
    scitbx::vec3<double> do_dshift1 = ddorg_dshift1;// + initial_offset[0] * dd1_dshift1 + initial_offset[1] * dd2_dshift1;
    scitbx::vec3<double> do_dshift2 = ddorg_dshift2;// + initial_offset[0] * dd1_dshift2 + initial_offset[1] * dd2_dshift2;
    scitbx::vec3<double> do_dtau1 = ddorg_dtau1 + initial_offset[0] * dd1_dtau1 + initial_offset[1] * dd2_dtau1;
    scitbx::vec3<double> do_dtau2 = ddorg_dtau2 + initial_offset[0] * dd1_dtau2 + initial_offset[1] * dd2_dtau2;
    scitbx::vec3<double> do_dtau3 = ddorg_dtau3 + initial_offset[0] * dd1_dtau3 + initial_offset[1] * dd2_dtau3;
    do_dtau1 /= 1000.0;
    do_dtau2 /= 1000.0;
    do_dtau3 /= 1000.0;
    dd1_dtau1 /= 1000.0;
    dd1_dtau2 /= 1000.0;
    dd1_dtau3 /= 1000.0;
    dd2_dtau1 /= 1000.0;
    dd2_dtau2 /= 1000.0;
    dd2_dtau3 /= 1000.0;

    dS_dp[0] = scitbx::mat3<double>{
        0.0,0.0,0.0,//dd1_ddist[0], dd1_ddist[1], dd1_ddist[2],
        0.0,0.0,0.0,//dd2_ddist[0], dd2_ddist[1], dd2_ddist[2],
        do_ddist[0], do_ddist[1], do_ddist[2],
    }.transpose();
    dS_dp[1] = scitbx::mat3<double>{
        0.0,0.0,0.0,//dd1_dshift1[0], dd1_dshift1[1], dd1_dshift1[2],
        0.0,0.0,0.0,//dd2_dshift1[0], dd2_dshift1[1], dd2_dshift1[2],
        do_dshift1[0], do_dshift1[1], do_dshift1[2],
    }.transpose();
    dS_dp[2] = scitbx::mat3<double>{
        0.0,0.0,0.0,//dd1_dshift2[0], dd1_dshift2[1], dd1_dshift2[2],
        0.0,0.0,0.0,//dd2_dshift2[0], dd2_dshift2[1], dd2_dshift2[2],
        do_dshift2[0], do_dshift2[1], do_dshift2[2],
    }.transpose();

    dS_dp[3] = scitbx::mat3<double>{
        dd1_dtau1[0], dd1_dtau1[1], dd1_dtau1[2],
        dd2_dtau1[0], dd2_dtau1[1], dd2_dtau1[2],
        do_dtau1[0], do_dtau1[1], do_dtau1[2],
    }.transpose();
    dS_dp[4] = scitbx::mat3<double>{
        dd1_dtau2[0], dd1_dtau2[1], dd1_dtau2[2],
        dd2_dtau2[0], dd2_dtau2[1], dd2_dtau2[2],
        do_dtau2[0], do_dtau2[1], do_dtau2[2],
    }.transpose();
    dS_dp[5] = scitbx::mat3<double>{
        dd1_dtau3[0], dd1_dtau3[1], dd1_dtau3[2],
        dd2_dtau3[0], dd2_dtau3[1], dd2_dtau3[2],
        do_dtau3[0], do_dtau3[1], do_dtau3[2],
    }.transpose();
}

SimpleDetectorParameterisation::SimpleDetectorParameterisation(
    const dxtbx::model::Detector &Detector,
    bool fix_dist=true, bool fix_shift1=true, bool fix_shift2=true,
    bool fix_tau1=true, bool fix_tau2=true, bool fix_tau3=true): 
        _fix_dist{fix_dist}, _fix_shift1{fix_shift1}, _fix_shift2{fix_shift2},
        _fix_tau1{fix_tau1}, _fix_tau2{fix_tau2}, _fix_tau3{fix_tau3}{
    const dxtbx::model::Panel& p = Detector[0];
    scitbx::vec3<double> so = p.get_origin();
    scitbx::vec3<double> d1 = p.get_fast_axis();
    scitbx::vec3<double> d2 = p.get_slow_axis();
    scitbx::vec3<double> dn = p.get_normal();
    initial_d1 = d1;
    initial_d2 = d2;
    initial_dn = dn;
    double panel_lim_x = p.get_image_size_mm()[0];
    double panel_lim_y = p.get_image_size_mm()[1];
    initial_offset = {-0.5 * panel_lim_x, -0.5 * panel_lim_y, 0.0};
    scitbx::vec3<double> dorg = so - (initial_offset[0]*d1) - (initial_offset[1]*d2);
    params[0] = p.get_directed_distance();
    scitbx::vec3<double> shift = dorg - dn*params[0];
    params[1] = shift * d1;
    params[2] = shift * d2;
    compose();
}

scitbx::mat3<double> SimpleDetectorParameterisation::get_state(){
    scitbx::mat3<double> m = {
        current_d1[0], current_d2[0], current_origin[0],
        current_d1[1], current_d2[1], current_origin[1],
        current_d1[2], current_d2[2], current_origin[2]
    };
    return m;
}

scitbx::af::shared<double> SimpleDetectorParameterisation::get_params() {
  return params;
}
void SimpleDetectorParameterisation::set_params(scitbx::af::shared<double> p) {
  params = p;
  compose();
}
scitbx::af::shared<scitbx::mat3<double>> SimpleDetectorParameterisation::get_dS_dp() {
  return dS_dp;
}

bool SimpleDetectorParameterisation::dist_fixed(){
    return _fix_dist;
}
bool SimpleDetectorParameterisation::shift1_fixed(){
    return _fix_shift1;
}
bool SimpleDetectorParameterisation::shift2_fixed(){
    return _fix_shift2;
}
bool SimpleDetectorParameterisation::tau1_fixed(){
    return _fix_tau1;
}
bool SimpleDetectorParameterisation::tau2_fixed(){
    return _fix_tau2;
}
bool SimpleDetectorParameterisation::tau3_fixed(){
    return _fix_tau3;
}

#endif  // DIALS_RESEARCH_DPARAM