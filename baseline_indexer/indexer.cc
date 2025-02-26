#include <assert.h>
#include <dx2/beam.h>
#include <dx2/crystal.h>
#include <dx2/detector.h>
#include <dx2/experiment.h>
#include <dx2/goniometer.h>
#include <dx2/h5/h5read_processed.hpp>
#include <dx2/scan.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/os.h>
#include <mutex>

#include <Eigen/Dense>
#include <argparse/argparse.hpp>
#include <chrono>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "common.hpp"
#include "fft3d.cc"
#include "flood_fill.cc"
#include "gemmi/symmetry.hpp"
#include "sites_to_vecs.cc"
#include "xyz_to_rlp.cc"
#include "refman_filter.cc"
#include "assign_indices.h"
#include "reflection_data.h"
#include "scanstaticpredictor.cc"
#include "combinations.cc"
#include <thread>
using Eigen::Matrix3d;
using Eigen::Vector3d;
using Eigen::Vector3i;
using json = nlohmann::json;

#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>

std::mutex mtx;

// Implement y = (x-5)^2
struct MyFunctor
{
    typedef float Scalar;

    typedef Eigen::VectorXf InputType;
    typedef Eigen::VectorXf ValueType;
    typedef Eigen::Matrix<Scalar,Eigen::Dynamic,Eigen::Dynamic> JacobianType;

    enum {
        InputsAtCompileTime = Eigen::Dynamic,
        ValuesAtCompileTime = Eigen::Dynamic
    };

    int operator()(const Eigen::VectorXf &x, Eigen::VectorXf &fvec) const
    {
        // We provide f(x) = x-5 because the algorithm will square this value internally
        fvec(0) = x(0) - 5.0;
        return 0;
    }

    int df(const Eigen::VectorXf &x, Eigen::MatrixXf &fjac) const {
      fjac(0,0) = 1;
      return 0;
    }

    int inputs() const { return 1; }// inputs is the dimension of x.
    int values() const { return 1; } // "values" is the number of f_i and
};

struct score_and_crystal {
    double score;
    Crystal crystal;
    double num_indexed;
    double rmsdxy;
};

std::map<int,score_and_crystal> results_map;
    
void calc_score(Crystal const &crystal,
  reflection_data const& obs, std::vector<Vector3d> const &rlp_select, std::vector<double> const &phi_select,
  Goniometer gonio, MonochromaticBeam beam, Panel panel, double width, int n){
  std::vector<Vector3i> miller_indices;
  int count;
  auto preassign = std::chrono::system_clock::now();
  std::tie(miller_indices, count) = assign_indices_global(crystal.get_A_matrix(), rlp_select, phi_select);
  auto t2 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = t2 - preassign;
  std::cout << "Time for assigning: " << elapsed_time.count() << " s" << std::endl;

  auto prefilter = std::chrono::system_clock::now();
  reflection_data sel_obs = reflection_filter_preevaluation(
      obs, miller_indices, gonio, crystal, beam, panel, width, 20
  );
  auto postfilter = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_timefilter = postfilter - prefilter;
  std::cout << "Time for reflection_filter: " << elapsed_timefilter.count() << " s" << std::endl;

  // Do the refinement


  // First make CrystalOrientationParameterisation, CrystalUnitCellParameterisation, BeamParameterisation,
  // DetectorParameterisationSinglePanel,
  //SimpleBParameterisation B_param {crystal.get_space_group()};
  // Then make SimplePredictionParam
  // Then set the crystal U, B in the expt object.
  // gradients = pred.get_gradients(obs)
  
  // Encapsulate those in a simple target, with a parameter vector.
  // That can calc resids and gradients which are input to a least-squares routine - use Eigen (lev mar)?
  // As part of residuals, it updates the parameterisation objects, updates the experiment objects
  // and runs the simple predictor on the reflection data.

  Eigen::VectorXf x(1);
  x(0) = 2;
  std::cout << "x: " << x << std::endl;
  MyFunctor myFunctor;
  Eigen::LevenbergMarquardt<MyFunctor, float> levenbergMarquardt(myFunctor);

  levenbergMarquardt.parameters.ftol = 1e-6;
  levenbergMarquardt.parameters.xtol = 1e-6;
  levenbergMarquardt.parameters.maxfev = 10; // Max iterations

  Eigen::VectorXf xmin = x; // initialize
  levenbergMarquardt.minimize(xmin);

  std::cout << "x that minimizes the function: " << xmin << std::endl;



  //write the score to the results map
  double xsum = 0;
  double ysum = 0;
  double zsum = 0;
  for (int i=0;i<sel_obs.flags.size();i++){
      //if (sel_obs.miller_indices[i] == null){
      //    continue;
      //}
      Vector3d xyzobs = sel_obs.xyzobs_mm[i];
      Vector3d xyzcal = sel_obs.xyzcal_mm[i];
      xsum += std::pow(xyzobs[0] - xyzcal[0],2);
      ysum += std::pow(xyzobs[1] - xyzcal[1],2);
      zsum += std::pow(xyzobs[2] - xyzcal[2],2);
  }
  double rmsdx = std::pow(xsum / sel_obs.xyzcal_mm.size(), 0.5);
  double rmsdy = std::pow(ysum / sel_obs.xyzcal_mm.size(), 0.5);
  double rmsdz = std::pow(zsum / sel_obs.xyzcal_mm.size(), 0.5);
  double xyrmsd = std::pow(std::pow(rmsdx, 2)+std::pow(rmsdy, 2), 0.5);

  // FIXME score the refined model
  score_and_crystal sac;
  sac.score = n;
  sac.crystal = crystal;
  sac.num_indexed = count;
  sac.rmsdxy = xyrmsd;
  mtx.lock();
  results_map[n] = sac;
  mtx.unlock();
  std::cout<< "Done from thread# " << std::this_thread::get_id() << std::endl;
}
constexpr double RAD2DEG = 180.0 / M_PI;

int main(int argc, char** argv) {
    // The purpose of an indexer is to determine the lattice model that best
    // explains the positions of the strong spots found during spot-finding.
    // The lattice model is a set of three vectors that define the crystal
    // lattice translations.
    // The experiment models (beam, detector) can also be refined during the
    // indexing process. The output is a set of models - a new crystal model that
    // describes the crystal lattice and an updated set of experiment models.
    auto t1 = std::chrono::system_clock::now();
    auto parser = argparse::ArgumentParser();
    parser.add_argument("-e", "--expt").help("Path to the DIALS expt file");
    parser.add_argument("-r", "--refl")
      .help("Path to the h5 reflection table file containing spotfinding results");
    parser.add_argument("--dmin")
      .help("The resolution limit of spots to use in the indexing process.")
      .scan<'f', float>();
    parser.add_argument("--max-cell")
      .help("The maximum possible cell length to consider during indexing")
      .scan<'f', float>();
    parser
      .add_argument(
        "--fft-npoints")  // mainly for testing, likely would always want to keep it as 256.
      .help(
        "The number of grid points to use for the fft. Powers of two are most "
        "efficient.")
      .default_value<uint32_t>(256)
      .scan<'u', uint32_t>();
    parser
      .add_argument("--nthreads")  // mainly for testing.
      .help(
        "The number of threads to use for the fft calculation."
        "Defaults to the value of std::thread::hardware_concurrency."
        "Better performance can typically be obtained with a higher number"
        "of threads than this.")
      .scan<'u', size_t>();
    parser.parse_args(argc, argv);

    if (!parser.is_used("expt")) {
        logger->error("Must specify experiment list file with --expt\n");
        std::exit(1);
    }
    if (!parser.is_used("refl")) {
        logger->error(
          "Must specify spotfinding results file (in DIALS HDF5 format) with --refl\n");
        std::exit(1);
    }
    // In DIALS, the max cell is automatically determined through a nearest
    // neighbour analysis that requires the annlib package. For now,
    // let's make this a required argument to help with testing/comparison
    // to DIALS.
    if (!parser.is_used("max-cell")) {
        logger->error("Must specify --max-cell\n");
        std::exit(1);
    }
    // FIXME use highest resolution by default to remove this requirement.
    if (!parser.is_used("dmin")) {
        logger->error("Must specify --dmin\n");
        std::exit(1);
    }
    std::string imported_expt = parser.get<std::string>("expt");
    std::string filename = parser.get<std::string>("refl");
    double max_cell = parser.get<float>("max-cell");
    double d_min = parser.get<float>("dmin");

    // Parse the experiment list (a json file) and load the models.
    // Will be moved to dx2.
    std::ifstream f(imported_expt);
    json elist_json_obj;
    try {
        elist_json_obj = json::parse(f);
    } catch (json::parse_error& ex) {
        logger->error("Unable to read {}; json parse error at byte {}",
                     imported_expt.c_str(),
                     ex.byte);
        std::exit(1);
    }
    Experiment<MonochromaticBeam> expt;
    try {
        expt = Experiment<MonochromaticBeam>(elist_json_obj);
    } catch (std::invalid_argument const& ex){
        logger->error("Unable to create MonochromaticBeam experiment: {}", ex.what());
        std::exit(1);
    }
    Scan scan = expt.scan();
    MonochromaticBeam beam = expt.beam();
    Goniometer gonio = expt.goniometer();
    Detector detector = expt.detector();
    assert(detector.panels().size()
           == 1);  // only considering single panel detectors initially.
    Panel panel = detector.panels()[0];

    // Read data from a reflection table. Again, this should be moved to
    // dx2 and only require the data array name (xyzobs.px.value) with some
    // logic to step through the directory structure
    std::string array_name = "/dials/processing/group_0/xyzobs.px.value";
    // Note, xyzobs_px is the flattened, on-disk representation of the array
    // i.e. if there are 100 spots, the length of xyzobs_px is 300, and
    // contains the elements [x0, y0, z0, x1, y1, z1, ..., x99, y99, z99]
    std::vector<double> xyzobs_px =
      read_array_from_h5_file<double>(filename, array_name);

    // The diffraction spots form a lattice in reciprocal space (if the experimental
    // geometry is accurate). So use the experimental models to transform the spot
    // coordinates on the detector into reciprocal space.
    std::vector<Vector3d> rlp = xyz_to_rlp(xyzobs_px, panel, beam, scan, gonio);
    logger->info("Number of reflections: {}", rlp.size());

    // b_iso is an isotropic b-factor used to weight the points when doing the fft.
    // i.e. high resolution (weaker) spots are downweighted by the expected
    // intensity fall-off as as function of resolution.
    double b_iso = -4.0 * std::pow(d_min, 2) * log(0.05);
    uint32_t n_points = parser.get<uint32_t>("fft-npoints");
    logger->info("Setting b_iso = {:.3f}", b_iso);

    // Create an array to store the fft result. This is a 3D grid of points, typically 256^3.
    std::vector<double> real_fft_result(n_points * n_points * n_points);

    // Do the fft of the reciprocal lattice coordinates.
    // the used in indexing array denotes whether a coordinate was used for the
    // fft (might not be if dmin filter was used for example). The used_in_indexing array
    // is sometimes used onwards in the dials indexing algorithms, so keep for now.
    size_t nthreads;
    if (parser.is_used("nthreads")) {
        nthreads = parser.get<size_t>("nthreads");
    } else {
        size_t max_threads = std::thread::hardware_concurrency();
        nthreads = max_threads ? max_threads : 1;
    }

    std::vector<bool> used_in_indexing =
      fft3d(rlp, real_fft_result, d_min, b_iso, n_points, nthreads);

    // The fft result is noisy. We want to extract the peaks, which may be spread over several
    // points on the fft grid. So we use a flood fill algorithm (https://en.wikipedia.org/wiki/Flood_fill)
    // to determine the connected regions in 3D. This is how it is done in DIALS, but I note that
    // perhaps this could be done with connected components analysis.
    // So do the flood fill, and extract the centres of mass of the peaks and the number of grid points
    // that contribute to each peak.
    std::vector<int> grid_points_per_void;
    std::vector<Vector3d> centres_of_mass_frac;
    // 15.0 is the DIALS 'rmsd_cutoff' parameter to filter out weak peaks.
    std::tie(grid_points_per_void, centres_of_mass_frac) =
      flood_fill(real_fft_result, 15.0, n_points);
    // Do some further filtering, 0.15 is the DIALS peak_volume_cutoff parameter.
    std::tie(grid_points_per_void, centres_of_mass_frac) =
      flood_fill_filter(grid_points_per_void, centres_of_mass_frac, 0.15);

    // Convert the peak centres from the fft grid into vectors in reciprocal space. These are our candidate
    // lattice vectors.
    // 3.0 is the min cell parameter.
    std::vector<Vector3d> candidate_lattice_vectors = sites_to_vecs(
      centres_of_mass_frac, grid_points_per_void, d_min, 3.0, max_cell, n_points);

    // Fix this inefficient selection with reflection-table-like struct.
    auto predata = std::chrono::system_clock::now();
    std::string flags_array_name = "/dials/processing/group_0/flags";
    std::vector<std::size_t> flags = read_array_from_h5_file<std::size_t>(filename, flags_array_name);
    // calculate s1 and xyzobsmm
    std::vector<Vector3d> s1(rlp.size());
    std::vector<Vector3d> xyzobs_mm(rlp.size());
    std::vector<Vector3d> xyzcal_mm(rlp.size());
    std::vector<double> phi(rlp.size());
    Vector3d s0 = beam.get_s0();
    Vector3d axis = gonio.get_rotation_axis();
    Matrix3d d_matrix = panel.get_d_matrix();
    std::array<double, 2> oscillation = scan.get_oscillation();
    double osc_width = oscillation[1];
    double osc_start = oscillation[0];
    int image_range_start = scan.get_image_range()[0];
    double DEG2RAD = M_PI / 180.0;
    double wl = beam.get_wavelength();
    for (int i = 0; i < rlp.size(); ++i) {
        int vec_idx= 3*i;
        double x1 = xyzobs_px[vec_idx];
        double x2 = xyzobs_px[vec_idx+1];
        double x3 = xyzobs_px[vec_idx+2];
        std::array<double, 2> xymm = panel.px_to_mm(x1,x2);
        double rot_angle = (((x3 + 1 - image_range_start) * osc_width) + osc_start) * DEG2RAD;
        phi[i] = rot_angle;
        Vector3d m = {xymm[0], xymm[1], 1.0};
        Vector3d s1_this = d_matrix * m;
        s1_this.normalize();
        s1[i] = s1_this / wl;
        xyzobs_mm[i] = {xymm[0], xymm[1], rot_angle};
    }

    // calculate entering array
    std::vector<bool> enterings(rlp.size());
    Vector3d vec = s0.cross(axis);
    for (int i=0;i<s1.size();i++){
        enterings[i] = ((s1[i].dot(vec)) < 0.0);
    }

    std::vector<double> phi_select(rlp.size());
    std::vector<Vector3d> rlp_select(rlp.size());
    std::vector<std::size_t> flags_select(rlp.size());
    std::vector<Vector3d> xyzobs_mm_select(rlp.size());
    std::vector<Vector3d> xyzcal_mm_select(rlp.size());
    std::vector<Vector3d> s1_select(rlp.size());
    std::vector<bool> entering_select(rlp.size());
    int selcount=0;
    // also select flags, xyzobs/cal, s1 and enterings
    for (int i=0;i<phi_select.size();i++){
        if ((1.0/rlp[i].norm()) > d_min && (phi[i]*RAD2DEG <= 360.0)){
            phi_select[selcount] = phi[i];
            rlp_select[selcount] = rlp[i];
            flags_select[selcount] = flags[i];
            xyzobs_mm_select[selcount] = xyzobs_mm[i];
            xyzcal_mm_select[selcount] = xyzcal_mm[i];
            s1_select[selcount] = s1[i];
            entering_select[selcount] = enterings[i];
            selcount++;
        }
    }
    rlp_select.resize(selcount);
    phi_select.resize(selcount);
    flags_select.resize(selcount);
    xyzobs_mm_select.resize(selcount);
    xyzcal_mm_select.resize(selcount);
    s1_select.resize(selcount);
    entering_select.resize(selcount);
    auto postdata = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_timedata = postdata - predata;
    std::cout << "Time for making data arrays: " << elapsed_timedata.count() << " s" << std::endl;
    Vector3i null{{0,0,0}};
    int n = 0;

    // FIXME check somewhere that there are solutions
    CandidateOrientationMatrices candidates(candidate_lattice_vectors, 1000); //quick (<1ms)
    // iterate over candidates; assign indices, refine, score.
    // need a map of scores for candidates: index to score and xtal. What about miller indices?
    int max_refine = 50;
    std::vector<Vector3i> miller_indices;
    
    int count;
    int n_images = scan.get_image_range()[1] - scan.get_image_range()[0] + 1;
    double width = scan.get_oscillation()[0] + (scan.get_oscillation()[1] * n_images);

    reflection_data obs;
    obs.flags = flags_select;
    obs.xyzobs_mm = xyzobs_mm_select;
    obs.xyzcal_mm = xyzcal_mm_select;
    obs.s1 = s1_select;
    obs.entering = entering_select;

    std::vector<std::thread> threads;
    
    while (candidates.has_next() && n < max_refine){
        // could do all this threaded.
        Crystal crystal = candidates.next(); //quick (<0.1ms)
        n++;
        //calc_score(crystal, obs, rlp_select, phi_select, gonio, beam, panel, width);
        threads.emplace_back(std::thread(calc_score, crystal, obs, rlp_select, phi_select,
         gonio, beam, panel, width, n));
    }
    for (auto &t : threads){
        t.join();
    }

    /*while (candidates.has_next() && n < max_refine){
        // could do all this threaded.
        Crystal crystal = candidates.next(); //quick (<0.1ms)
        n++;
        auto preassign = std::chrono::system_clock::now();
        std::tie(miller_indices, count) = assign_indices_global(crystal.get_A_matrix(), rlp_select, phi_select);
        auto t2 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_time = t2 - preassign;
        std::cout << "Time for assigning: " << elapsed_time.count() << " s" << std::endl;

        // make a reflection table like object
        reflection_data obs;
        obs.miller_indices = miller_indices;
        obs.flags = flags_select;
        obs.xyzobs_mm = xyzobs_mm_select;
        obs.xyzcal_mm = xyzcal_mm_select;
        obs.s1 = s1_select;
        obs.entering = entering_select;

        // get a filtered selection for refinement
        auto prefilter = std::chrono::system_clock::now();
        reflection_data sel_obs = reflection_filter_preevaluation(
            obs, gonio, crystal, beam, panel, width, 20
        );
        auto postfilter = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_timefilter = postfilter - prefilter;
        std::cout << "Time for reflection_filter: " << elapsed_timefilter.count() << " s" << std::endl;
        // do some refinement
        // FIXME implement in part3

        // now calculate the rmsd and model likelihood
        double xsum = 0;
        double ysum = 0;
        double zsum = 0;
        for (int i=0;i<sel_obs.flags.size();i++){
            //if (sel_obs.miller_indices[i] == null){
            //    continue;
            //}
            Vector3d xyzobs = sel_obs.xyzobs_mm[i];
            Vector3d xyzcal = sel_obs.xyzcal_mm[i];
            xsum += std::pow(xyzobs[0] - xyzcal[0],2);
            ysum += std::pow(xyzobs[1] - xyzcal[1],2);
            zsum += std::pow(xyzobs[2] - xyzcal[2],2);
        }
        double rmsdx = std::pow(xsum / sel_obs.xyzcal_mm.size(), 0.5);
        double rmsdy = std::pow(ysum / sel_obs.xyzcal_mm.size(), 0.5);
        double rmsdz = std::pow(zsum / sel_obs.xyzcal_mm.size(), 0.5);
        double xyrmsd = std::pow(std::pow(rmsdx, 2)+std::pow(rmsdy, 2), 0.5);

        // FIXME score the refined model
        score_and_crystal sac;
        sac.score = (double)n;
        sac.crystal = crystal;
        sac.num_indexed = count;
        sac.rmsdxy = xyrmsd;
        results_map[n] = sac;
    }*/
    std::cout << "Unit cell, #indexed, rmsd_xy" << std::endl;
    for (auto it=results_map.begin();it!=results_map.end();it++){
        gemmi::UnitCell cell = (*it).second.crystal.get_unit_cell();
        logger->info("{:>7.3f}, {:>7.3f}, {:>7.3f}, {:>7.3f}, {:>7.3f}, {:>7.3f}, {}, {:>7.4f}", cell.a,cell.b,cell.c,cell.alpha,cell.beta,cell.gamma,(*it).second.num_indexed, (*it).second.rmsdxy);
    }

    // find the best crystal from the map - lowest score
    auto it = *std::min_element(results_map.begin(), results_map.end(),
            [](const auto& l, const auto& r) { return l.second.score < r.second.score; });
    Crystal best_xtal = it.second.crystal;
    // at this point, we will test combinations of the candidate vectors, use those to index the spots, do some
    // refinement of the candidates and choose the best one. Then we will do some more refinement including extra
    // model parameters. At then end, we will have a list of refined experiment models (including a crystal)

    // For now, let's just write out the candidate vectors and write out the unrefined experiment models with the
    // first combination of candidate vectors as an example crystal, to demonstrate an example experiment list data
    // structure.
    
    // dump the candidate vectors to json
    std::string n_vecs = std::to_string(candidate_lattice_vectors.size() - 1);
    size_t n_zero = n_vecs.length();
    json vecs_out;
    for (int i = 0; i < candidate_lattice_vectors.size(); i++) {
        std::string s = std::to_string(i);
        auto pad_s = std::string(n_zero - std::min(n_zero, s.length()), '0') + s;
        vecs_out[pad_s] = candidate_lattice_vectors[i];
    }
    std::string outfile = "candidate_vectors.json";
    std::ofstream vecs_file(outfile);
    vecs_file << vecs_out.dump(4);
    logger->info("Saved candidate vectors to {}", outfile);

    // Now make a crystal and save an experiment list with the models.
    /*if (candidate_lattice_vectors.size() < 3) {
        logger->info(
          "Insufficient number of candidate vectors to make a crystal model.");
    } else {
        //gemmi::SpaceGroup space_group = *gemmi::find_spacegroup_by_name("P1");
        //Crystal best_xtal{candidate_lattice_vectors[0],
        //                  candidate_lattice_vectors[1],
        //                  candidate_lattice_vectors[2],
        //                  space_group};*/
    expt.set_crystal(best_xtal);
    json elist_out = expt.to_json();
    std::string efile_name = "elist.json";
    std::ofstream efile(efile_name);
    efile << elist_out.dump(4);
    logger->info("Saved experiment list to {}", efile_name);
    //}*/

    auto t2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_time = t2 - t1;
    logger->info("Total time for indexer: {:.4f}s", elapsed_time.count());
}
