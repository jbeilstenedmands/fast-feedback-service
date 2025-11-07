/**
 * @file predict.cc
 * @brief This program implements the algorithm for spot prediction.
 *
 * ReekeIndexGenerator outlined in in the LURE workshop notes.
 *
 */

#include <Eigen/Dense>
#include <argparse/argparse.hpp>
#include <chrono>
#include <cmath>
#include <common.hpp>
#include <concepts>
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>
#include <dx2/scan.hpp>
#include "index_generators.cc"
#include "ray_predictors.cc"
#include "utils.cc"
#include <exception>
#include <fstream>
#include <gemmi/symmetry.hpp>
#include <iostream>  // Debugging
#include <nlohmann/json.hpp>
#include <thread>
#include <type_traits>
#include <vector>

using json = nlohmann::json;

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::Vector3d;

// Enums to specify information about the experiment and beam.
enum class ExperimentType { Stills, Rotational };
enum class RotationalType { Static, ScanVarying };
enum class BeamType { Monochromatic, Polychromatic };

struct ScanVaryingType {
    bool beam = false;
    bool crystal = false;
    bool r_setting = false;
};

struct PredictionType {
    ExperimentType experiment_type = ExperimentType::Rotational;
    RotationalType rotational_type = RotationalType::ScanVarying;
};

const uint64_t predicted_flag = (1 << 0);


#pragma region Argument Parser Configuration
/**
 * @brief Take a default-initialized ArgumentParser object and configure it
 *      with the arguments to be parsed; assign various properties to each
 *      argument, eg. help message, default value, etc.
 *
 * @param parser The ArgumentParser object (pre-input) to be configured.
 */
void configure_parser(argparse::ArgumentParser& parser) {
    parser.add_argument("-e", "--expt")
      .help("path to DIALS expt file")
      .nargs(argparse::nargs_pattern::at_least_one);  //.required();
    parser.add_argument("--dmin")
      .help("minimum d-spacing of predicted reflections")
      .scan<'f', double>()
      .default_value(-1.0);
    //.required();
    parser.add_argument("-s", "--force_static")
      .help("for a scan varying model, forces static prediction")
      .default_value(false)
      .implicit_value(true);
    parser.add_argument("-b", "--buffer_size")
      .help(
        "calculates predictions within a buffer zone of n images either side"
        "of the scan")
      .scan<'i', int>()
      .default_value<int>(0);
    parser.add_argument("-n", "--nthreads")
      .help(
        "the number of threads to use for the fft calculation, "
        "defaults to the value of std::thread::hardware_concurrency, "
        "better performance can typically be obtained with a higher number"
        "of threads than this. UNUSED.")
      .scan<'u', size_t>()
      .default_value<size_t>(std::thread::hardware_concurrency());
}

/**
 * @brief Take an ArgumentParser object after the user has entered input and check
 *      it for consistency; output errors and exit the program if a check fails.
 *
 * @param parser The ArgumentParser object (post-input) to be verified.
 */
void verify_arguments(const argparse::ArgumentParser& parser) {
    if (!parser.is_used("expt")) {
        logger.error("Must specify experiment list file with -e or --expt\n");
        std::exit(1);
    }
    if (parser.is_used("buffer_size") && parser.get<int>("buffer_size") < 0) {
        logger.error("--buffer_size cannot be negative\n");
    }
    if (parser.is_used("nthreads") && parser.get<size_t>("nthreads") < 1) {
        logger.error("--nthreads cannot be less than 1\n");
        std::exit(1);
    }
}
#pragma endregion

// Implemented use cases
//   - Monochromatic rotation (scan static or scan varying)
//   - Monochromatic stills
//   - Polychromatic stills

struct predicted_data_rotation {
  // Shape {size, 3}
  std::vector<int> hkl;
  std::vector<double> s1;
  std::vector<double> xyz_px;
  std::vector<double> xyz_mm;
  //std::vector<double> s0_cal; // Only for poly stills
  // Shape {size, 1}
  std::vector<uint64_t> panels;
  std::vector<bool> enter;
  std::vector<uint64_t> flags;
  //std::vector<double> delpsi; // Only for mono stills
  //std::vector<double> wavelength_cal; // Only for poly stills
  std::vector<int> ids;
  std::vector<uint64_t> experiment_ids;
  std::vector<std::string> identifiers;

  void add(
        const std::array<int, 3>& hkl_entry,
        const std::array<double, 3>& s1_entry,
        const std::array<double, 3>& xyz_px_entry,
        const std::array<double, 3>& xyz_mm_entry,
        uint64_t panel,
        bool enter_flag,
        uint64_t flag
    ) {
        hkl.insert(hkl.end(), hkl_entry.begin(), hkl_entry.end());
        s1.insert(s1.end(), s1_entry.begin(), s1_entry.end());
        xyz_px.insert(xyz_px.end(), xyz_px_entry.begin(), xyz_px_entry.end());
        xyz_mm.insert(xyz_mm.end(), xyz_mm_entry.begin(), xyz_mm_entry.end());
        panels.push_back(panel);
        enter.push_back(enter_flag);
        flags.push_back(flag);
    }
};

struct scan_varying_data {
  std::vector<Vector3d> s0_at_scan_points;
  std::vector<Matrix3d> A_at_scan_points;
  std::vector<Matrix3d> r_setting_at_scan_points;
};

predicted_data_rotation predict_rotation(
  const Goniometer& goniometer, const Scan& scan, const Crystal& crystal, const MonochromaticBeam& beam, const Detector& detector, const scan_varying_data& sv_data, double param_dmin){
  bool scan_varying = false;
  if (!sv_data.s0_at_scan_points.empty()){
    scan_varying = true;
  }
  else if (!sv_data.A_at_scan_points.empty()){
    scan_varying = true;
  }
  else if (!sv_data.r_setting_at_scan_points.empty()){
    scan_varying = true;
  }
  gemmi::GroupOps crystal_symmetry_operations = crystal.get_space_group().operations();
  const Matrix3d A = crystal.get_A_matrix();
  const Vector3d m2 = goniometer.get_rotation_axis();
  // A Rotator object that generates rotations around axis m2
  const Rotator rotator(m2);
  const Matrix3d r_fixed = goniometer.get_sample_rotation();
  const Matrix3d r_setting = goniometer.get_setting_rotation();
  const double d_osc = scan.get_oscillation()[1];
  const double osc0 = scan.get_oscillation()[0];

  int z0 = scan.get_image_range()[0] - 1;
  int z1 = scan.get_image_range()[1];

  Vector3d s0 = beam.get_s0();

  predicted_data_rotation output_data; // to store the predictions.
  bool use_mono = true;

  for (int frame = z0; frame < z1; frame++) {
      int image_index = frame - z0;

      // Define the potentially scan-varying vector (s0) and matrices (A and r_setting)
      Vector3d s0_1 =
        sv_data.s0_at_scan_points.empty() ? s0 : sv_data.s0_at_scan_points[image_index];
      Vector3d s0_2 =
        sv_data.s0_at_scan_points.empty() ? s0 : sv_data.s0_at_scan_points[image_index+1];
      Matrix3d A1 = sv_data.A_at_scan_points.empty() ? A : sv_data.A_at_scan_points[image_index];
      Matrix3d A2 = sv_data.A_at_scan_points.empty() ? A : sv_data.A_at_scan_points[image_index+1];
      Matrix3d r_setting_1 = sv_data.r_setting_at_scan_points.empty() ? r_setting : sv_data.r_setting_at_scan_points[image_index];
      Matrix3d r_setting_2 = sv_data.r_setting_at_scan_points.empty() ? r_setting : sv_data.r_setting_at_scan_points[image_index+1];
      Matrix3d r_setting_1_inv = r_setting_1.inverse();
      // Redefine A1 and A2 to encompass all 3 rotations
      const double phi_beg = osc0 + image_index * d_osc;
      const double phi_end = phi_beg + d_osc;
      Matrix3d r_beg = rotator.rotation_matrix(phi_beg);
      Matrix3d r_end = rotator.rotation_matrix(phi_end);
      A1 = r_setting_1 * r_beg * r_fixed * A1;
      A2 = r_setting_2 * r_end * r_fixed * A2;

      ReekeIndexGenerator index_generator(
        A1,
        A2,
        crystal_symmetry_operations,
        s0_1,
        s0_2,
        param_dmin,
        use_mono
      );

      std::function<std::array<std::optional<Ray>, 2>(const std::array<int, 3>&)> predict_ray;
      if (scan_varying) {
          predict_ray = [=](const std::array<int, 3>& index) {
              std::array<std::optional<Ray>, 2> rays;
              rays[0] = predict_ray_monochromatic_sv(index, A1, A2, s0_1, s0_2, param_dmin, phi_beg, d_osc);
              return rays;
          };
      } else {
          predict_ray = [=](const std::array<int, 3>& index) {
              return predict_ray_monochromatic_static(index, A1, r_setting_1, r_setting_1_inv, s0, m2, rotator, param_dmin, phi_beg, d_osc);
          };
      }

      for (;;) {
        std::optional<std::array<int, 3>> index = index_generator.next();
        if (!index) break;

        // Check if a reflection occurs at the given Miller index
        // within the required resolution.
        std::array<std::optional<Ray>, 2> rays = predict_ray(index.value());
        for (std::optional<Ray> ray : rays) {
          if (!ray) continue;
          // Append the ray
          auto impact = detector.get_ray_intersection(ray->s1);
          if (!impact.has_value()) continue;
          // Get the frame that a reflection with this angle will be observed at
          double frame = z0 + (ray->angle - osc0) / d_osc;
          intersection result = impact.value();
          auto panel = result.panel_id;
          std::array<double, 3> coords_mm = {result.xymm[0], result.xymm[1], ray->angle * M_PI / 180};
          std::array<double, 2> xycoords_px = detector.panels()[panel].mm_to_px(
            coords_mm[0], coords_mm[1]);
          std::array<double, 3> coords_px = {xycoords_px[0], xycoords_px[1], frame};
          std::array<double, 3> s1 = { ray->s1[0], ray->s1[1], ray->s1[2] };
          output_data.add(index.value(), s1, coords_px, coords_mm, panel, ray->entering, predicted_flag);
        }
      }

  }
  
  return output_data;
}

int main(int argc, char** argv) {
    auto t1 = std::chrono::system_clock::now();
    auto parser = argparse::ArgumentParser();
    configure_parser(parser);

    // Parse the command-line input against the defined parser
    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        logger.error(err.what());
        std::exit(1);
    }

    verify_arguments(parser);

    // Obtain argument values from the parsed command-line input
    const auto param_expt_paths = parser.get<std::vector<std::string>>("expt");
    auto param_dmin = parser.get<double>("dmin");
    auto param_force_static = parser.get<bool>("force_static");
    const auto param_buffer_size = parser.get<int>("buffer_size");
    const auto param_nthreads = parser.get<size_t>("nthreads");
    const std::string output_file_path = "predicted.refl";
    

#pragma region Create Reflection Data Containers
    // Create std::vectors to store results in, and later add them as columns to a ReflectionTable.
    
    
    predicted_data_rotation output_data;
#pragma endregion

    for (const std::string& expt_path : param_expt_paths) {
        // Get data from .expt file
        json data = json::parse(std::ifstream(expt_path));
        if (data.size() == 0) {
            logger.error("Experiment file " + expt_path + " is empty.\n");
            std::exit(1);
        }

        json experiment_list = data.at("experiment");

        for (int i_expt = 0; i_expt < experiment_list.size(); i_expt++) {
            // Compute the number of reflections predicted already (used when storing ids)
            const std::size_t num_reflections_initial = output_data.panels.size();

#pragma region Determine Experiment Parameters
            // FIXME: Extracting information from the json object manually here, as the ExperimentList
            // class does not exist yet.
            // In the future, please update the below logic.
            const json expt_details = experiment_list[i_expt];
            const std::string identifier = expt_details.at("identifier");

            // Obtain the indices indicating where experiment data can be found
            const std::size_t i_detector = expt_details.at("detector");
            const std::size_t i_goniometer = expt_details.at("goniometer");
            const std::size_t i_scan = expt_details.at("scan");
            const std::size_t i_crystal = expt_details.at("crystal");
            const std::size_t i_beam_data = expt_details.at("beam");
            const std::size_t i_imageset = expt_details.at("imageset");

            // Obtain json data components at the corresponding indices
            const json detector_data = data.at("detector")[i_detector];
            const json goniometer_data = data.at("goniometer")[i_goniometer];
            const json scan_data = data.at("scan")[i_scan];
            const json crystal_data = data.at("crystal")[i_crystal];
            const json beam_data = data.at("beam")[i_beam_data];

            // Construct dx2 objects from the json data
            Detector detector(detector_data);
            Goniometer goniometer(goniometer_data);
            Scan scan(scan_data);
            Crystal crystal(crystal_data);
            // Note: Make crystal a shared_ptr to potentially add MosaicCrystalSauter2014 functionality in the future
            //std::shared_ptr<Crystal> crystal;
            // if not MosaicCrystalSauter2014:
            //crystal = std::make_shared<Crystal>(crystal_data);
            // if MosaicCrystalSauter2014:
            // crystal = std::make_shared<MosaicCrystalSauter2014>(crystal(crystal_data));
            /*gemmi::GroupOps crystal_symmetry_operations =
              crystal->get_space_group().operations();

            // Extract the A matrix
            const Matrix3d A = crystal->get_A_matrix();*/
#pragma endregion

#pragma region Determine Scan Parameters
            // Edit scan range and oscillation start. This adds 2 * param_buffer_size
            // images to the predictions.
            // FIXME: In most cases, this makes the program default to static prediction,
            // however the number of images after buffer adjustments may be such that the
            // condition for scan varying prediction is met. Explicitly set
            // param_force_static to true here?
            const int num_images =
              scan.get_image_range()[1] - scan.get_image_range()[0] + 1;
            const double osc0 = scan.get_oscillation()[0];
            if (param_buffer_size > 0) {
                scan = Scan({scan.get_image_range()[0] - param_buffer_size,
                             scan.get_image_range()[1] + param_buffer_size},
                            {scan.get_oscillation()[0]
                               - param_buffer_size * scan.get_oscillation()[1],
                             scan.get_oscillation()[1]});
                param_force_static = true;
            }
#pragma endregion

#pragma region Determine Beam Parameters
            BeamType beam_type;
            // Use the below for monochromatic and polychromatic prediction (where they represent the lower end of the wavelength)
            double wavelength;
            Vector3d s0;
            // Use the below for polychromatic prediction only
            double wavelength_poly_max = 0;
            Experiment<MonochromaticBeam> mono_expt;
            Experiment<PolychromaticBeam> poly_expt;
            if (beam_data.at("__id__") == "monochromatic") {
                mono_expt = Experiment<MonochromaticBeam>(data);
                //MonochromaticBeam beam(beam_data);
                beam_type = BeamType::Monochromatic;
                wavelength = mono_expt.beam().get_wavelength();
                wavelength_poly_max = wavelength;
                s0 = mono_expt.beam().get_s0();
            } else if (beam_data.at("__id__") == "polychromatic") {
                poly_expt = Experiment<PolychromaticBeam>(data);
                //PolychromaticBeam beam(beam_data);
                beam_type = BeamType::Polychromatic;
                wavelength = poly_expt.beam().get_wavelength_range()[0];
                wavelength_poly_max = poly_expt.beam().get_wavelength_range()[1];
                s0 = -1.0 * poly_expt.beam().get_sample_to_source_direction() / wavelength;
            } else {
                logger.error(
                  "The beam's __id__ should be either monochromatic or polychromatic.");
                std::exit(1);
            }
#pragma endregion

#pragma region Determine dmin
            // Check if the minimum resolution paramenter (dmin) was passed in by the user,
            // if yes, check if it is a valid value; if not, assign a default.
            double dmin_min = 0.5 * wavelength;
            // FIXME: Need a better dmin_default from .expt file (like in DIALS)
            double dmin_default = dmin_min;
            if (!parser.is_used("dmin")) {
                param_dmin = dmin_default;
            } else if (param_dmin < dmin_min) {
                logger.error(
                  "Prediction at a dmin of {} is not possible with wavelength {}. "
                  "dmin "
                  "must be at least 0.5 times the wavelength.\nSetting dmin to the "
                  "default value of {}.\n",
                  param_dmin,
                  wavelength,
                  dmin_default);
                param_dmin = dmin_default;
            }
#pragma endregion

#pragma region Determine Prediction Parameters
            // Determine experiment type and extract data depending on type of scan
            PredictionType prediction_type{ExperimentType::Rotational,
                                           RotationalType::Static};
            ScanVaryingType sv_type;

            scan_varying_data sv_data;

            json s0_at_scan_points;
            json A_at_scan_points;
            json r_setting_at_scan_points;
            if (scan.get_oscillation()[1] == 0.0){
              prediction_type.experiment_type = ExperimentType::Stills;
                if (param_force_static) {
                  logger.info(
                      "The experiment is not a rotation. Ignoring the "
                      "--force_static "
                      "flag and falling back on stills prediction.");
                }
            } else if (param_force_static) {
                prediction_type.experiment_type = ExperimentType::Rotational;
                prediction_type.rotational_type = RotationalType::Static;
            } else {
                if (beam_data.contains("s0_at_scan_points")) {
                    s0_at_scan_points = beam_data.at("s0_at_scan_points");
                    if (s0_at_scan_points.size() == num_images + 1){ // i.e. is expected length.
                        std::vector<Vector3d> scan_varying_s0;
                        for (const auto& entry : s0_at_scan_points) {
                            Vector3d vec(entry[0].get<double>(),
                                         entry[1].get<double>(),
                                         entry[2].get<double>());
                            scan_varying_s0.push_back(vec);
                        }
                        sv_data.s0_at_scan_points = scan_varying_s0;
                    }
                }
                if (crystal_data.contains("A_at_scan_points")) {
                    A_at_scan_points = crystal_data.at("A_at_scan_points");
                    if (A_at_scan_points.size() == num_images + 1){
                        std::vector<Matrix3d> scan_varying_A;
                        for (const auto& entry : A_at_scan_points) {
                            Matrix3d A_mat;
                            A_mat << entry[0].get<double>(),
                                         entry[1].get<double>(),
                                         entry[2].get<double>(),
                                         entry[3].get<double>(),
                                         entry[4].get<double>(),
                                         entry[5].get<double>(),
                                         entry[6].get<double>(),
                                         entry[7].get<double>(),
                                         entry[8].get<double>();
                            scan_varying_A.push_back(A_mat);
                        }
                        sv_data.A_at_scan_points = scan_varying_A;
                    }
                }
                if (goniometer_data.contains("setting_rotation_at_scan_points")) {
                    r_setting_at_scan_points =
                      goniometer_data.at("setting_rotation_at_scan_points");
                    if (r_setting_at_scan_points.size() == num_images + 1){
                        std::vector<Matrix3d> scan_varying_r;
                        for (const auto& entry : r_setting_at_scan_points) {
                            Matrix3d r_mat;
                            r_mat << entry[0].get<double>(),
                                         entry[1].get<double>(),
                                         entry[2].get<double>(),
                                         entry[3].get<double>(),
                                         entry[4].get<double>(),
                                         entry[5].get<double>(),
                                         entry[6].get<double>(),
                                         entry[7].get<double>(),
                                         entry[8].get<double>();
                            scan_varying_r.push_back(r_mat);
                        }
                        sv_data.r_setting_at_scan_points = scan_varying_r;
                    }
                }

                prediction_type.experiment_type = ExperimentType::Rotational;
                prediction_type.rotational_type =
                  (sv_type.beam || sv_type.crystal || sv_type.r_setting)
                    ? RotationalType::ScanVarying
                    : RotationalType::Static;
            }

            auto prediction_type_string = [&]() {
                if (prediction_type.experiment_type == ExperimentType::Stills)
                    return "stills";
                else {
                    // Rotational branch
                    if (prediction_type.rotational_type == RotationalType::ScanVarying)
                        return "scan-varying";
                    else
                        return "static";
                }
            };
            logger.info("{} {} prediction on {}",
                        (beam_type == BeamType::Monochromatic) ? "Monochromatic"
                                                               : "Polychromatic",
                        prediction_type_string(),
                        expt_path);
#pragma endregion

#pragma region Prediction
            /*if (prediction_type.experiment_type == ExperimentType::Stills) {
                // A large enough angular tolerance allows plenty of Miller indices to be
                // available for checking against a finer tolerance.
                // Typical delta_psi tolerance is 0.0015, so a default of 0.005 is reasonable.
                // FIXME: Is there a clean way to determine this dynamically if the
                // mosaicity values are provided in the experiment file? (See below.)

                double angular_tolerance =
                  (beam_type == BeamType::Monochromatic) ? 0.005 : 0;
                Vector3d s0_lower = s0 * wavelength / wavelength_poly_max;
                // FIXME: This is very ugly because I had to make last-minute adjustments to accommodate
                // polychromatic prediction. Perhaps it is a better ideas to branch into mono and poly first,
                // then construct this generator! s0 here (for polychormatic) represents s0_upper (i.e. the
                // s0 corresponding to the lower wavelength)
                StillsIndexGenerator index_generator(
                  A, crystal_symmetry_operations, s0, s0_lower, angular_tolerance);

                for (;;) {
                    auto index = index_generator.next();
                    if (!index) break;

                    // Check if a reflection occurs at the given Miller index
                    // within the required resolution.
                    std::optional<Ray> ray;
                    if (beam_type == BeamType::Monochromatic) {
                        double delta_psi_tolerance = 0.0015;
                        if (crystal_data.find("mosaicity") != crystal_data.end()
                            && crystal_data.at("mosaicity") > 0) {
                            double ML_domain_size_ang =
                              crystal_data.at("ML_domain_size_ang");
                            double ML_half_mosaicity_deg =
                              crystal_data.at("ML_half_mosaicity_deg");
                            Vector3d h_vec = {(double)index.value()[0],
                                              (double)index.value()[1],
                                              (double)index.value()[2]};
                            double d = 1.0 / (A * h_vec).norm();
                            delta_psi_tolerance =
                              (d / ML_domain_size_ang)
                              + (ML_half_mosaicity_deg * M_PI / 360);
                        }
                        ray = predict_ray_monochromatic_stills(
                          index.value(), A, s0, param_dmin, delta_psi_tolerance);
                    } else
                        // AKA Laue prediction
                        ray = predict_ray_polychromatic_stills(index.value(),
                                                               A,
                                                               s0.normalized(),
                                                               wavelength,
                                                               wavelength_poly_max,
                                                               param_dmin);

                    if (!ray) continue;
                    // Append the ray
                    auto impact = detector.get_ray_intersection(ray->s1);
                    if (!impact.has_value()) continue;
                    intersection result = impact.value();
                    auto panel = result.panel_id;
                    auto coords_mm = result.xymm;
                    auto coords_px = detector.panels()[panel].mm_to_px(
                      coords_mm[0], coords_mm[1]);
                    
                    output_data.add(index, ray->s1, coords_px, coords_mm, )

                    hkl.insert(
                      hkl.end(), std::begin(index.value()), std::end(index.value()));
                    enter.push_back(ray->entering);
                    s1.insert(s1.end(), std::begin(ray->s1), std::end(ray->s1));
                    xyz_mm.insert(
                      xyz_mm.end(), std::begin(coords_mm), std::end(coords_mm));
                    xyz_mm.push_back(0);
                    xyz_px.insert(
                      xyz_px.end(), std::begin(coords_px), std::end(coords_px));
                    xyz_px.push_back(0);
                    panels.push_back(panel);
                    flags.push_back(predicted_flag);
                    if (beam_type == BeamType::Monochromatic){
                        delpsi.push_back(ray->angle);
	            }
                    else {
                        wavelength_cal.push_back(1.0 / ray->s1.norm());
                        Vector3d s0_pred = s0.normalized() / ray->s1.norm();
                        s0_cal.insert(
                          s0_cal.end(), std::begin(s0_pred), std::end(s0_pred));
                    }
                }
            }*/
            //else {
            output_data = predict_rotation(goniometer, scan, crystal, mono_expt.beam(), detector, sv_data, param_dmin);
            /*    // Rotational-experiment-specific code goes here
                output_data = predict_rotation();

                const Vector3d m2 = goniometer.get_rotation_axis();
                // A Rotator object that generates rotations around axis m2
                const Rotator rotator(m2);
                const Matrix3d r_fixed = goniometer.get_sample_rotation();
                const Matrix3d r_setting = goniometer.get_setting_rotation();
                const double d_osc = scan.get_oscillation()[1];

                int z0 = scan.get_image_range()[0] - 1;
                int z1 = scan.get_image_range()[1];

                for (int frame = z0; frame < z1; frame++) {
                    int image_index = frame - z0;

                    // Define the potentially scan-varying vector (s0) and matrices (A and r_setting)
                    Vector3d s0_1 =
                      sv_type.beam ? vector_3d_from_json(s0_at_scan_points[image_index])
                                   : s0;
                    Vector3d s0_2 =
                      sv_type.beam
                        ? vector_3d_from_json(s0_at_scan_points[image_index + 1])
                        : s0;
                    Matrix3d A1 = sv_type.crystal
                                    ? matrix_3d_from_json(A_at_scan_points[image_index])
                                    : A;
                    Matrix3d A2 =
                      sv_type.crystal
                        ? matrix_3d_from_json(A_at_scan_points[image_index + 1])
                        : A;
                    Matrix3d r_setting_1 =
                      sv_type.r_setting
                        ? matrix_3d_from_json(r_setting_at_scan_points[image_index])
                        : r_setting;
                    Matrix3d r_setting_2 =
                      sv_type.r_setting
                        ? matrix_3d_from_json(r_setting_at_scan_points[image_index + 1])
                        : r_setting;

                    // Redefine A1 and A2 to encompass all 3 rotations
                    const double phi_beg = osc0 + image_index * d_osc;
                    const double phi_end = phi_beg + d_osc;
                    Matrix3d r_beg = rotator.rotation_matrix(phi_beg);
                    Matrix3d r_end = rotator.rotation_matrix(phi_end);
                    A1 = r_setting_1 * r_beg * r_fixed * A1;
                    A2 = r_setting_2 * r_end * r_fixed * A2;

                    ReekeIndexGenerator index_generator(
                      A1,
                      A2,
                      crystal_symmetry_operations,
                      s0_1,
                      s0_2,
                      param_dmin,
                      beam_type == BeamType::Monochromatic);

                    for (;;) {
                        auto index = index_generator.next();
                        if (!index) break;

                        // Check if a reflection occurs at the given Miller index
                        // within the required resolution.
                        std::array<std::optional<Ray>, 2> rays;
                        
                        if (prediction_type.rotational_type
                            == RotationalType::ScanVarying) {
                            rays[0] = predict_ray_monochromatic_sv(index.value(),
                                                                    A1,
                                                                    A2,
                                                                    s0_1,
                                                                    s0_2,
                                                                    param_dmin,
                                                                    phi_beg,
                                                                    d_osc);
                        }
                        else {
                            // Monochromatic, Static, Rotation > 5 degrees.
                            // This is a new use case not supported by DIALS.
                            // For d_osc > 10 degrees, this exact calculator is much better
                            // than the DIALS-style approximate predictor
                            rays = predict_ray_monochromatic_static(
                              index.value(),
                              A1,
                              r_setting_1,
                              r_setting_1.inverse(),
                              s0,
                              m2,
                              rotator,
                              param_dmin,
                              phi_beg,
                              d_osc);
                        }

                        for (std::optional<Ray> ray : rays) {
                            if (!ray) continue;
                            // Append the ray
                            auto impact = detector.get_ray_intersection(ray->s1);
                            if (!impact.has_value()) continue;
                            intersection result = impact.value();
                            auto panel = result.panel_id;
                            auto coords_mm = result.xymm;
                            auto coords_px = detector.panels()[panel].mm_to_px(
                              coords_mm[0], coords_mm[1]);

                            // Get the frame that a reflection with this angle will be observed at
                            double frame = z0 + (ray->angle - osc0) / d_osc;

                            hkl.insert(hkl.end(),
                                       std::begin(index.value()),
                                       std::end(index.value()));
                            enter.push_back(ray->entering);
                            s1.insert(s1.end(), std::begin(ray->s1), std::end(ray->s1));
                            xyz_mm.insert(
                              xyz_mm.end(), std::begin(coords_mm), std::end(coords_mm));
                            xyz_mm.push_back(ray->angle * M_PI / 180);
                            xyz_px.insert(
                              xyz_px.end(), std::begin(coords_px), std::end(coords_px));
                            xyz_px.push_back(frame);
                            panels.push_back(panel);
                            flags.push_back(predicted_flag);
                        }
                    }
                }
            }

            std::size_t num_new_reflections = panels.size() - num_reflections_initial;
            std::vector<int32_t> new_ids(num_new_reflections, i_expt);
            ids.insert(ids.end(), new_ids.begin(), new_ids.end());
            experiment_ids.push_back(i_expt);
            identifiers.push_back(identifier);
            */
            std::size_t num_new_reflections = output_data.panels.size() - num_reflections_initial;
            std::vector<int32_t> new_ids(num_new_reflections, i_expt);
            output_data.ids.insert(output_data.ids.end(), new_ids.begin(), new_ids.end());
            output_data.experiment_ids.push_back(i_expt);
            output_data.identifiers.push_back(identifier);
        }
    }
#pragma endregion

#pragma region Populate Reflection Table
    // Check if the vector sizes are consistent after prediction (before creating a ReflectionTable).
    // Note that delpsi, wavelength_cal, and s0_cal are allowed to either be 0 or equal to the row
    // size, depening on the type of prediction done.
    if ((output_data.hkl.size() != 3 * output_data.panels.size()) || (output_data.hkl.size() != 3 * output_data.enter.size())
        || (output_data.hkl.size() != output_data.s1.size()) || (output_data.hkl.size() != output_data.xyz_px.size())
        || (output_data.hkl.size() != output_data.xyz_mm.size()) || (output_data.hkl.size() != 3 * output_data.flags.size())){
        //|| !(output_data.hkl.size() == 3 * delpsi.size() || delpsi.size() == 0)
        //|| !(output_data.hkl.size() == 3 * wavelength_cal.size() || wavelength_cal.size() == 0)
        //|| !(output_data.hkl.size() == s0_cal.size() || s0_cal.size() == 0)) {
        logger.error(
          "The sizes of the columns after prediction are not "
          "consistent with "
          "each other"
          /*\n hkl.size() = {}\n panel.size() = {}\n "
          "enter.size() "
          "= {}\n s1.size() = {}\n xyz_px.size() = {}\n xyz_mm.size() "
          "= {}\n flags.size() = {}\n ids.size() = {}\n delpsi.size() = "
          "{}\n wavelength_cal.size() = {}\n s0_cal.size() = {}\n",
          hkl.size(),
          panels.size(),
          enter.size(),
          s1.size(),
          xyz_px.size(),
          xyz_mm.size(),
          flags.size(),
          ids.size(),
          delpsi.size(),
          wavelength_cal.size(),
          s0_cal.size()*/);
        std::exit(1);
    }

    // Store the size, once it has been verified as being consistent across columns.
    std::size_t sz = output_data.panels.size();
    ReflectionTable predicted(output_data.experiment_ids, output_data.identifiers);
    predicted.add_column("miller_index", sz, 3, output_data.hkl);
    predicted.add_column("panel", sz, 1, output_data.panels);
    predicted.add_column("entering", sz, 1, output_data.enter);
    predicted.add_column("s1", sz, 3, output_data.s1);
    predicted.add_column("xyzcal.px", sz, 3, output_data.xyz_px);
    predicted.add_column("xyzcal.mm", sz, 3, output_data.xyz_mm);
    predicted.add_column("flags", sz, 1, output_data.flags);
    predicted.add_column("id", sz, 1, output_data.ids);
    /*if (delpsi.size()) predicted.add_column("delpsical.rad", sz, 1, delpsi);
    if (wavelength_cal.size())
        predicted.add_column("wavelength_cal", sz, 1, wavelength_cal);
    if (s0_cal.size()) predicted.add_column("s0_cal", sz, 3, s0_cal);*/

#pragma endregion


#pragma region Write to File
    // Save reflections to file
    predicted.write(output_file_path);

    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    logger.info("Saved {} reflections to {}.", sz, output_file_path);
    logger.info("Total time for prediction: {:.4f}s", elapsed_time.count());
#pragma endregion
    return 0;
}
