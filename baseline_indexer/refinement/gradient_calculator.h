
#include <scitbx/vec3.h>
#include <dials/array_family/scitbx_shared_and_versa.h>
#include <scitbx/array_family/flex_types.h>
#include <dxtbx/model/experiment.h>
#include <dials_research/fast_dp2/indexing/Uparameterisation.h>
#include <dials_research/fast_dp2/indexing/Bparameterisation.h>
#include <dials_research/fast_dp2/indexing/Beamparameterisation.h>
#include <dials_research/fast_dp2/indexing/Detectorparameterisation.h>
#include <cctbx/miller.h>
#include <dials/array_family/reflection_table.h>

class GradientsCalculator {
public:
    GradientsCalculator(
        dxtbx::model::Experiment &experiment,
        SimpleUParameterisation &uparam,
        SimpleBParameterisation &bparam,
        SimpleBeamParameterisation &beamparam,
        SimpleDetectorParameterisation &Dparam);
    boost::python::list get_gradients(dials::af::reflection_table &obs);

private:
  dxtbx::model::Experiment experiment;
  SimpleUParameterisation uparam;
  SimpleBParameterisation bparam;
  SimpleBeamParameterisation beamparam;
  SimpleDetectorParameterisation Dparam;
};

GradientsCalculator::GradientsCalculator(dxtbx::model::Experiment &experiment,
SimpleUParameterisation &uparam, SimpleBParameterisation &bparam,
SimpleBeamParameterisation &beamparam, SimpleDetectorParameterisation &Dparam) :
experiment(experiment), uparam(uparam), bparam(bparam), beamparam(beamparam), Dparam(Dparam){};

boost::python::list GradientsCalculator::get_gradients(dials::af::reflection_table &obs){
    int n_ref = obs.size();
    // assume one panel detector for now
    // Some templating to handle multi-panel/single panel detectors? will need a function to pass in a panel array
    // and return D array.
    scitbx::af::shared<scitbx::mat3<double>> D(n_ref, Dparam.get_state().inverse());

    scitbx::vec3<double> s0 = beamparam.get_state();
    scitbx::mat3<double> B = bparam.get_state();
    scitbx::mat3<double> U = uparam.get_state();
    scitbx::mat3<double> S = (*experiment.get_goniometer()).get_setting_rotation();
    scitbx::vec3<double> axis = (*experiment.get_goniometer()).get_rotation_axis_datum();
    scitbx::mat3<double> F = (*experiment.get_goniometer()).get_fixed_rotation();
    scitbx::mat3<double> UB = U * B;
    scitbx::af::shared<scitbx::vec3<double>> s1 = obs["s1"];
    scitbx::af::shared<scitbx::vec3<double>> xyz = obs["xyzcal.mm"];
    scitbx::af::const_ref<cctbx::miller::index<>> hkl = obs["miller_index"];
    scitbx::af::shared<scitbx::vec3<double>> r(n_ref);
    scitbx::af::shared<scitbx::vec3<double>> pv(n_ref);
    scitbx::af::shared<scitbx::vec3<double>> e_X_r(n_ref);
    scitbx::af::shared<double> e_r_s0(n_ref);
    scitbx::af::shared<double> w_inv(n_ref);
    scitbx::af::shared<double> uw_inv(n_ref);
    scitbx::af::shared<double> vw_inv(n_ref);
    for (int i=0;i<n_ref;i++){
        scitbx::vec3<double> pv_this = D[i] * s1[i];
        pv[i] = pv_this;
        double pvinv = 1.0 / pv_this[2];
        w_inv[i] = pvinv;
        uw_inv[i] = pvinv * pv_this[0];
        vw_inv[i] = pvinv * pv_this[1];
        scitbx::vec3<double> q = F * (UB * hkl[i]);
        scitbx::vec3<double> r_i = S * q.rotate_around_origin(axis, xyz[i][2]);
        r[i] = r_i;
        e_X_r[i] = (S * axis).cross(r_i);
        e_r_s0[i] = e_X_r[i] * s0;
    }

    boost::python::list gradients;
    scitbx::af::shared<scitbx::mat3<double>> ds_dp = uparam.get_dS_dp();
    scitbx::af::shared<scitbx::mat3<double>> db_dp = bparam.get_dS_dp();
    scitbx::af::shared<scitbx::vec3<double>> dbeam_dp = beamparam.get_dS_dp();
    scitbx::af::shared<scitbx::mat3<double>> dD_dp = Dparam.get_dS_dp();

    //beam derivatives
    std::vector<bool> free;
    free.push_back(!beamparam.in_spindle_plane_fixed());
    free.push_back(!beamparam.out_spindle_plane_fixed());
    free.push_back(!beamparam.wavelength_fixed());
    for (int j=0;j<dbeam_dp.size();j++){
        // each gradient is dx/dp, dy/dp, dz/dp
        scitbx::af::shared<double> gradient(n_ref*3);
        scitbx::vec3<double> dbeam_dp_j = dbeam_dp[j];
        if (free[j]){
            for (int k=0;k<n_ref;k++){
                double dphi = -1.0 * dbeam_dp_j * r[k] / e_r_s0[k];
                scitbx::vec3<double> dpv = D[k] * ((e_X_r[k] * dphi) + dbeam_dp_j);
                double w_inv_this = w_inv[k];
                gradient[k] = w_inv_this * (dpv[0] - dpv[2] * uw_inv[k]);
                gradient[k+n_ref] = w_inv_this * (dpv[1] - dpv[2] * vw_inv[k]);
                gradient[k+(2*n_ref)] = dphi;
            }
        }
        gradients.append(gradient);
    }

    for (int j=0;j<ds_dp.size();j++){
        // each gradient is dx/dp, dy/dp, dz/dp
        scitbx::af::shared<double> gradient(n_ref*3);
        scitbx::mat3<double> ds_dp_j = ds_dp[j];
        for (int k=0;k<n_ref;k++){
            scitbx::vec3<double> tmp = F * (ds_dp_j * B * hkl[k]);
            scitbx::vec3<double> dr = S * tmp.rotate_around_origin(axis, xyz[k][2]);
            double dphi = -1.0 * (dr * s1[k]) / e_r_s0[k];
            scitbx::vec3<double> dpv = D[k] * (dr + e_X_r[k]*dphi);
            double w_inv_this = w_inv[k];
            gradient[k] = w_inv_this * (dpv[0] - dpv[2] * uw_inv[k]);
            gradient[k+n_ref] = w_inv_this * (dpv[1] - dpv[2] * vw_inv[k]);
            gradient[k+(2*n_ref)] = dphi;
        }
        gradients.append(gradient);
    }
    for (int j=0;j<db_dp.size();j++){
        // each gradient is dx/dp, dy/dp, dz/dp
        scitbx::af::shared<double> gradient(n_ref*3);
        scitbx::mat3<double> db_dp_j = db_dp[j];
        for (int k=0;k<n_ref;k++){
            scitbx::vec3<double> tmp = F * (U * db_dp_j * hkl[k]);
            scitbx::vec3<double> dr = S * tmp.rotate_around_origin(axis, xyz[k][2]);
            double dphi = -1.0 * (dr * s1[k]) / e_r_s0[k];
            scitbx::vec3<double> dpv = D[k] * (dr + e_X_r[k]*dphi);
            double w_inv_this = w_inv[k];
            gradient[k] = w_inv_this * (dpv[0] - dpv[2] * uw_inv[k]);
            gradient[k+n_ref] = w_inv_this * (dpv[1] - dpv[2] * vw_inv[k]);
            gradient[k+(2*n_ref)] = dphi;
        }
        gradients.append(gradient);
    }

    for (int j=0;j<dD_dp.size();j++){
        // each gradient is dx/dp, dy/dp, dz/dp
        scitbx::af::shared<double> gradient(n_ref*3);
        scitbx::mat3<double> dD_dp_j = dD_dp[j];
        for (int k=0;k<n_ref;k++){
            scitbx::vec3<double> dpv = (D[k] * dD_dp_j * -1.0) * pv[k];
            double w_inv_this = w_inv[k];
            gradient[k] = w_inv_this * (dpv[0] - dpv[2] * uw_inv[k]);
            gradient[k+n_ref] = w_inv_this * (dpv[1] - dpv[2] * vw_inv[k]);
            // note there is no dphi component here.
        }
        gradients.append(gradient);
    }
    

    return gradients;
}