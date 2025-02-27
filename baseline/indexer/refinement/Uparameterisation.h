#ifndef DIALS_RESEARCH_UPARAM
#define DIALS_RESEARCH_UPARAM
#include <scitbx/vec3.h>
#include <cmath>
#include <dials/algorithms/refinement/parameterisation/parameterisation_helpers.h>
#include <dxtbx/model/experiment.h>

class SimpleUParameterisation {
public:
  SimpleUParameterisation(const dxtbx::model::Crystal &crystal);
  scitbx::af::shared<double> get_params();
  void set_params(scitbx::af::shared<double> p);
  scitbx::mat3<double> get_state();
  scitbx::af::shared<scitbx::mat3<double>> get_dS_dp();

private:
  scitbx::af::shared<double> params{3, 0.0};
  scitbx::af::shared<scitbx::vec3<double>> axes{3, scitbx::vec3<double>(1.0, 0.0, 0.0)};
  void compose();
  scitbx::mat3<double> istate{};
  scitbx::mat3<double> U_{};
  scitbx::af::shared<scitbx::mat3<double>> dS_dp{
    3,
    scitbx::mat3<double>(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)};
};

void SimpleUParameterisation::compose() {
  dials::refinement::CrystalOrientationCompose coc(
    istate, params[0], axes[0], params[1], axes[1], params[2], axes[2]);
  U_ = coc.U();
  dS_dp[0] = coc.dU_dphi1();
  dS_dp[1] = coc.dU_dphi2();
  dS_dp[2] = coc.dU_dphi3();
}

SimpleUParameterisation::SimpleUParameterisation(const dxtbx::model::Crystal &crystal) {
  istate = crystal.get_U();
  axes[1] = scitbx::vec3<double>(0.0, 1.0, 0.0);
  axes[2] = scitbx::vec3<double>(0.0, 0.0, 1.0);
  compose();
}

scitbx::af::shared<double> SimpleUParameterisation::get_params() {
  return params;
}
scitbx::mat3<double> SimpleUParameterisation::get_state() {
  return U_;
}
void SimpleUParameterisation::set_params(scitbx::af::shared<double> p) {
  assert(p.size() == 3);
  params = p;
  compose();
}
scitbx::af::shared<scitbx::mat3<double>> SimpleUParameterisation::get_dS_dp() {
  return dS_dp;
}

#endif  // DIALS_RESEARCH_UPARAM