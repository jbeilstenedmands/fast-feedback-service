#ifndef SCORE_CRYSTALS_H
#define SCORE_CRYSTALS_H

#include <chrono>
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/reflection.hpp>
#include <mutex>
#include <nlohmann/json.hpp>
#include <vector>
#include "assign_indices.hpp"
#include "ffs_logger.hpp"
#include "non_primitive_basis.hpp"
#include "reflection_filter.hpp"
#include "refine_candidate.hpp"

extern std::mutex score_and_crystal_mtx;

// A struct to score a candidate crystal model.
struct score_and_crystal {
    double score;
    Crystal crystal;
    MonochromaticBeam beam;
    Panel panel;
    double num_indexed;
    double rmsdxy;
    double fraction_indexed;
    double volume_score;
    double indexed_score;
    double rmsd_score;

    json to_json(){
        json data;
        data["score"] = score;
        data["num_indexed"] = num_indexed;
        data["rmsdxy"] = rmsdxy;
        data["fraction_indexed"] = fraction_indexed;
        data["volume_score"] = volume_score;
        data["indexed_score"] = indexed_score;
        data["rmsd_score"] = rmsd_score;
        data["crystal"] = crystal.to_json();
        return data;
    }
};
extern std::map<int, score_and_crystal> results_map;
/**
 * @brief Evaluate a crystal model by evaluating how well it describes the reflection data.
 * @param crystal The crystal model.
 * @param obs The reflection data.
 * @param gonio The goniometer model.
 * @param beam The beam model.
 * @param panel The panel from the detector model.
 * @param scan_width The scan width in degrees.
 * @param n The candidate number, starting at 1.
 */
void evaluate_crystal(Crystal crystal,
                      ReflectionTable const& obs,
                      Goniometer gonio,
                      MonochromaticBeam beam,
                      Panel panel,
                      double scan_width,
                      int n);

/**
 * @brief Determine a relative score for all solutions.
 * @param results_map A map of the candidate number to its score_and_crystal struct.
 */
void score_solutions(std::map<int, score_and_crystal>& results_map);
#endif
