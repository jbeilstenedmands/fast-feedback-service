#ifndef DIALS_RESEARCH_BEAMPARAM
#define DIALS_RESEARCH_BEAMPARAM
#include <scitbx/vec3.h>
#include <cmath>
#include <dxtbx/model/experiment.h>

class SimpleBeamParameterisation {
public:
  SimpleBeamParameterisation(
    const dxtbx::model::Beam &Beam, const dxtbx::model::Goniometer &Goniometer,
    bool fix_in_spindle_plane, bool fix_out_spindle_plane, bool fix_wavelength);
  scitbx::af::shared<double> get_params();
  void set_params(scitbx::af::shared<double> p);
  scitbx::vec3<double> get_state();
  scitbx::af::shared<scitbx::vec3<double>> get_dS_dp();
  bool in_spindle_plane_fixed();
  bool out_spindle_plane_fixed();
  bool wavelength_fixed();

private:
  scitbx::af::shared<double> params{3, 0.0}; //mu1, mu2, nu
  void compose();
  scitbx::vec3<double> istate_s0{};
  scitbx::vec3<double> istate_pol_norm{};
  scitbx::vec3<double> s0{};
  scitbx::vec3<double> pn{};
  scitbx::vec3<double> s0_plane_dir1{};
  scitbx::vec3<double> s0_plane_dir2{};
  scitbx::af::shared<scitbx::vec3<double>> dS_dp{
    3,
    scitbx::vec3<double>(0.0, 0, 0)};
  bool _fix_in_spindle_plane{true};
  bool _fix_out_spindle_plane{false};
  bool _fix_wavelength{true};
};

scitbx::mat3<double> dR_from_axis_and_angle(const scitbx::vec3<double> &axis,
                                             double angle) {
    scitbx::vec3<double> axis_ = axis.normalize();
    double ca = cos(angle);
    double sa = sin(angle);
    return scitbx::mat3<double>(sa * axis_[0] * axis_[0] - sa,
                        sa * axis_[0] * axis_[1] - ca * axis_[2],
                        sa * axis_[0] * axis_[2] + ca * axis_[1],
                        sa * axis_[1] * axis_[0] + ca * axis_[2],
                        sa * axis_[1] * axis_[1] - sa,
                        sa * axis_[1] * axis_[2] - ca * axis_[0],
                        sa * axis_[2] * axis_[0] - ca * axis_[1],
                        sa * axis_[2] * axis_[1] + ca * axis_[0],
                        sa * axis_[2] * axis_[2] - sa);
                                             }

// axis and angle as rot mat
scitbx::mat3<double> axis_and_angle_as_rot(scitbx::vec3<double> axis, double angle){
    double q0=0.0;
    double q1=0.0;
    double q2=0.0;
    double q3=0.0;
    if (!(std::fmod(angle, 2.0*M_PI))){
        q0=1.0;
    }
    else {
        double h = 0.5 * angle;
        q0 = cos(h);
        double s = sin(h);
        scitbx::vec3<double> n = axis.normalize();
        q1 = n[0]*s;
        q2 = n[1]*s;
        q3 = n[2]*s;
    }
    scitbx::mat3<double> m = {
        2*(q0*q0+q1*q1)-1, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2),
        2*(q1*q2+q0*q3),   2*(q0*q0+q2*q2)-1, 2*(q2*q3-q0*q1),
        2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1),   2*(q0*q0+q3*q3)-1};
    return m;
}

void SimpleBeamParameterisation::compose(){
    double mu1rad = params[0] / 1000.0;
    double mu2rad = params[1] / 1000.0;
    scitbx::mat3<double> Mu1 = axis_and_angle_as_rot(s0_plane_dir1, mu1rad);
    scitbx::mat3<double> Mu2 = axis_and_angle_as_rot(s0_plane_dir2, mu2rad);
    scitbx::mat3<double> dMu1_dmu1 = dR_from_axis_and_angle(s0_plane_dir1, mu1rad);
    scitbx::mat3<double> dMu2_dmu2 = dR_from_axis_and_angle(s0_plane_dir2, mu2rad);
    // compose new state
    scitbx::mat3<double> Mu21 = Mu2 * Mu1;
    scitbx::vec3<double> s0_new_dir = (Mu21 * istate_s0).normalize();
    pn = (Mu21 * istate_pol_norm).normalize();
    s0 = params[2] * s0_new_dir;

    //    # calculate derivatives of the beam direction wrt angles:
    //    #  1) derivative wrt mu1
    scitbx::mat3<double> dMu21_dmu1 = Mu2 * dMu1_dmu1;
    scitbx::vec3<double> ds0_new_dir_dmu1 = dMu21_dmu1 * istate_s0;

    //    #  2) derivative wrt mu2
    scitbx::mat3<double> dMu21_dmu2 = dMu2_dmu2 * Mu1;
    scitbx::vec3<double> ds0_new_dir_dmu2 = dMu21_dmu2 * istate_s0;

    //# calculate derivatives of the attached beam vector, converting
    //# parameters back to mrad
    dS_dp[0] = ds0_new_dir_dmu1 * params[2] / 1000.0;
    dS_dp[1] = ds0_new_dir_dmu2 * params[2] / 1000.0;
    dS_dp[2] = s0_new_dir;
}

SimpleBeamParameterisation::SimpleBeamParameterisation(
    const dxtbx::model::Beam &beam, const dxtbx::model::Goniometer &goniometer,
    bool fix_in_spindle_plane=true, bool fix_out_spindle_plane=false, bool fix_wavelength=true): _fix_in_spindle_plane{fix_in_spindle_plane}, _fix_out_spindle_plane{fix_out_spindle_plane}, _fix_wavelength{fix_wavelength} {
        s0 = beam.get_s0();
        istate_s0 = beam.get_unit_s0();
        istate_pol_norm = beam.get_polarization_normal();
        scitbx::vec3<double> spindle = goniometer.get_rotation_axis_datum();
        s0_plane_dir2 = s0.cross(spindle).normalize(); // axis associated with mu2
        s0_plane_dir1 = s0_plane_dir2.cross(s0).normalize(); //axis associated with mu1
        params[2] = s0.length();
        compose();
    }

scitbx::vec3<double> SimpleBeamParameterisation::get_state() {
  return s0;
}
scitbx::af::shared<double> SimpleBeamParameterisation::get_params() {
  return params;
}
void SimpleBeamParameterisation::set_params(scitbx::af::shared<double> p) {
  params = p;
  compose();
}
scitbx::af::shared<scitbx::vec3<double>> SimpleBeamParameterisation::get_dS_dp() {
  return dS_dp;
}
bool SimpleBeamParameterisation::in_spindle_plane_fixed(){
  return _fix_in_spindle_plane;
}
bool SimpleBeamParameterisation::out_spindle_plane_fixed(){
  return _fix_out_spindle_plane;
}
bool SimpleBeamParameterisation::wavelength_fixed(){
  return _fix_wavelength;
}

#endif  // DIALS_RESEARCH_BEAMPARAM