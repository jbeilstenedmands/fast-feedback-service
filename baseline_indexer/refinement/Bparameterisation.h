#ifndef DIALS_RESEARCH_BPARAM
#define DIALS_RESEARCH_BPARAM
#include <scitbx/vec2.h>
#include <scitbx/vec3.h>
#include <cctbx/sgtbx/space_group.h>
#include <cctbx/sgtbx/tensor_rank_2.h>
#include <cctbx/crystal_orientation.h>
#include <rstbx/symmetry/constraints/a_g_conversion.h>
#include <dxtbx/model/experiment.h>
#include <dials_research/fast_dp2/indexing/g_gradients.h>

class SymmetrizeReduceEnlarge {
public:
  SymmetrizeReduceEnlarge(cctbx::sgtbx::space_group space_group);
  void set_orientation(scitbx::mat3<double> B);
  scitbx::af::small<double, 6> forward_independent_parameters();
  cctbx::crystal_orientation backward_orientation(
    scitbx::af::small<double, 6> independent);
  scitbx::af::shared<scitbx::mat3<double>> forward_gradients();

private:
  cctbx::sgtbx::space_group space_group_;
  cctbx::sgtbx::tensor_rank_2::constraints<double> constraints_;
  cctbx::crystal_orientation orientation_{};
  rstbx::symmetry::AG Bconverter{};
};

class SimpleBParameterisation {
public:
  SimpleBParameterisation(const dxtbx::model::Crystal &crystal);
  scitbx::af::shared<double> get_params();
  void set_params(scitbx::af::shared<double>);
  scitbx::mat3<double> get_state();
  scitbx::af::shared<scitbx::mat3<double>> get_dS_dp();

private:
  scitbx::af::shared<double> params;
  void compose();
  scitbx::mat3<double> B_{};
  scitbx::af::shared<scitbx::mat3<double>> dS_dp{};
  SymmetrizeReduceEnlarge SRE;
};

SymmetrizeReduceEnlarge::SymmetrizeReduceEnlarge(cctbx::sgtbx::space_group space_group)
    : space_group_(space_group), constraints_(space_group, true) {}

void SymmetrizeReduceEnlarge::set_orientation(scitbx::mat3<double> B) {
  orientation_ = cctbx::crystal_orientation(B, true);
}

scitbx::af::small<double, 6> SymmetrizeReduceEnlarge::forward_independent_parameters() {
  Bconverter.forward(orientation_);
  return constraints_.independent_params(Bconverter.G);
}

cctbx::crystal_orientation SymmetrizeReduceEnlarge::backward_orientation(
  scitbx::af::small<double, 6> independent) {
  scitbx::sym_mat3<double> ustar = constraints_.all_params(independent);
  Bconverter.validate_and_setG(ustar);
  orientation_ = Bconverter.back_as_orientation();
  return orientation_;
}

scitbx::af::shared<scitbx::mat3<double>> SymmetrizeReduceEnlarge::forward_gradients() {
  return dB_dp(Bconverter, constraints_);
}

void SimpleBParameterisation::compose() {
  scitbx::af::small<double, 6> vals(params.size());
  for (int i = 0; i < params.size(); ++i) {
    vals[i] = 1E-5 * params[i];
  }
  SRE.set_orientation(B_);
  SRE.forward_independent_parameters();
  B_ = SRE.backward_orientation(vals).reciprocal_matrix();
  dS_dp = SRE.forward_gradients();
  for (int i = 0; i < dS_dp.size(); ++i) {
    for (int j = 0; j < 9; ++j) {
      dS_dp[i][j] *= 1E-5;
    }
  }
}

SimpleBParameterisation::SimpleBParameterisation(
  const dxtbx::model::Crystal &crystal)
    : B_(crystal.get_B()), SRE(crystal.get_space_group()) {
  // first get params
  SRE.set_orientation(B_);
  scitbx::af::small<double, 6> X = SRE.forward_independent_parameters();
  params = scitbx::af::shared<double>(X.size());
  for (int i = 0; i < X.size(); ++i) {
    params[i] = 1E5 * X[i];
  }
  compose();
}

scitbx::af::shared<double> SimpleBParameterisation::get_params() {
  return params;
}
scitbx::mat3<double> SimpleBParameterisation::get_state() {
  return B_;
}
void SimpleBParameterisation::set_params(scitbx::af::shared<double> p) {
  params = p;
  compose();
}
scitbx::af::shared<scitbx::mat3<double>> SimpleBParameterisation::get_dS_dp() {
  return dS_dp;
}

#endif  // DIALS_RESEARCH_BPARAM