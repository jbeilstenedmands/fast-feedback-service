#ifndef DIALS_RESEARCH_PREDICTOR
#define DIALS_RESEARCH_PREDICTOR
#include <scitbx/vec3.h>
#include <scitbx/mat3.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <scitbx/array_family/flex_types.h>
#include <dxtbx/model/detector.h>
#include <dxtbx/model/scan.h>
#include <dials/array_family/reflection_table.h>
#include <cmath>
#include <cassert>
#include <iostream>
const double two_pi = 2 * M_PI;

inline double mod2pi(double angle) {
  // E.g. treat 359.9999999 as 360
  if (std::abs(angle - two_pi) <= 1e-7) {
    angle = two_pi;
  }
  return angle - two_pi * floor(angle / two_pi);
}

// actually a repredictor, assumes all successful.
void simple_reflection_predictor(
  const scitbx::vec3<double> s0,//beam s0
  const scitbx::mat3<double> F,//fixed rot
  const scitbx::mat3<double> S,//setting rot
  const scitbx::vec3<double> R, //get_rotation_axis_datum
  const scitbx::mat3<double> UB,
  const dxtbx::model::Detector &Detector,
  dials::af::reflection_table &obs
  //const int image_range_start,
  //const double osc_start,
  //const double osc_width
){
  // these setup bits are the same for all refls.
  scitbx::vec3<double> s0_ = S.inverse() * s0;
  scitbx::mat3<double> FUB = F * UB;
  scitbx::vec3<double> m2 = R.normalize();//fixed during refine
  scitbx::vec3<double> s0_m2_plane = (s0.cross(S * R)).normalize();

  scitbx::vec3<double> m1 = m2.cross(s0_).normalize(); //vary with s0
  scitbx::vec3<double> m3 = m1.cross(m2).normalize(); //vary with s0
  double s0_d_m2 = s0_ * m2;
  double s0_d_m3 = s0_ * m3;

  scitbx::af::shared<scitbx::vec3<double>> xyzcalmm = obs["xyzcal.mm"];
  scitbx::af::shared<scitbx::vec3<double>> s1_all = obs["s1"];
  scitbx::af::shared<bool> entering = obs["entering"];
  scitbx::af::shared<scitbx::vec3<double>> xyzobs = obs["xyzobs.mm.value"];
  scitbx::af::shared<cctbx::miller::index<>> hkl = obs["miller_index"];
  scitbx::af::shared<std::size_t> flags = obs["flags"];
  size_t predicted_value = (1 << 0); //predicted flag
  // now call predict_rays with h and UB for a given refl
  for (int i=0;i<obs.size();i++){
    const cctbx::miller::index<> h = hkl[i];
    bool entering_i = entering[i];

    scitbx::vec3<double> pstar0 = FUB * h;
    double pstar0_len_sq = pstar0.length_sq();
    if (pstar0_len_sq > 4 * s0_.length_sq()){
      flags[i] &= ~predicted_value;
      continue;
    }
    double pstar0_d_m1 = pstar0 * m1;
    double pstar0_d_m2 = pstar0 * m2;
    double pstar0_d_m3 = pstar0 * m3;
    double pstar_d_m3 = (-(0.5 * pstar0_len_sq) - (pstar0_d_m2 * s0_d_m2)) / s0_d_m3;
    double rho_sq = (pstar0_len_sq - (pstar0_d_m2*pstar0_d_m2));
    double psq = pstar_d_m3*pstar_d_m3;
    if (rho_sq < psq){
      flags[i] &= ~predicted_value;
      continue;
    }
    //DIALS_ASSERT(rho_sq >= sqr(pstar_d_m3));
    double pstar_d_m1 = sqrt(rho_sq - (psq));
    double p1 = pstar_d_m1 * pstar0_d_m1;
    double p2 = pstar_d_m3 * pstar0_d_m3;
    double p3 = pstar_d_m1 * pstar0_d_m3;
    double p4 = pstar_d_m3 * pstar0_d_m1;
    
    double cosphi1 = p1 + p2;
    double sinphi1 = p3 - p4;
    double a1 = atan2(sinphi1, cosphi1);
    // ASSERT must be in range? is_angle_in_range

    // check each angle
    scitbx::vec3<double> pstar = S * pstar0.unit_rotate_around_origin(m2, a1);
    scitbx::vec3<double> s1 = s0_ + pstar;
    bool this_entering = s1 * s0_m2_plane < 0.;
    double angle;
    if (this_entering == entering_i){
      // use this s1 and a1 (mod 2pi)
      angle = mod2pi(a1);
    }
    else {
      double cosphi2 = -p1 + p2;
      double sinphi2 = -p3 - p4;
      double a2 = atan2(sinphi2, cosphi2);
      pstar = S * pstar0.unit_rotate_around_origin(m2, a2);
      s1 = s0_ + pstar;
      this_entering = s1 * s0_m2_plane < 0.;
      assert(this_entering == entering_i);
      angle = mod2pi(a2);
    }

    // only need frame if calculating xyzcalpx, but not needed for evaluation
    //double frame = image_range_start + ((angle - osc_start) / osc_width) - 1;
    //scitbx::vec3<double> v = D * s1;
    scitbx::vec2<double> mm = Detector[0].get_ray_intersection(s1); //v = D * s1; v[0]/v[2], v[1]/v[2]
    //scitbx::vec2<double> px = Detector[0].millimeter_to_pixel(mm); // requires call to parallax corr
    
    // match full turns
    double phiobs = xyzobs[i][2];
    // first fmod positive
    double val = std::fmod(phiobs, two_pi);
    while (val < 0) val += two_pi;
    double resid = angle - val;
    // second fmod positive
    double val2 = std::fmod(resid+M_PI, two_pi);
    while (val2 < 0) val2 += two_pi;
    val2 -= M_PI;
    
    xyzcalmm[i] = {mm[0], mm[1], phiobs + val2};
    //xyzcalpx[i] = {px[0], px[1], frame};
    s1_all[i] = s1;
    flags[i] |= predicted_value;
  }
}

#endif  // DIALS_RESEARCH_PREDICTOR