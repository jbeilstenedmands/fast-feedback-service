// This program reads a nexus file and creates experiment models

#include <argparse/argparse.hpp>
#include "ffs_logger.hpp"
#include "import.hpp"
#include "h5read.h"
#include <dx2/beam.hpp>
#include <dx2/experiment.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <format>
#include <sstream>



int main(int argc, char **argv) {
    // This program creates dx2 experiment models from nxmx-format data.
    auto t1 = std::chrono::system_clock::now();
    auto parser = ImportArgumentParser("1.0");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::ostringstream oss;
        oss << parser;
        logger.info(oss.str()); // print help
        logger.error("Error: {}", err.what());
        return 1;
    }
    
    // Get the nexus file and create the reader.
    std::string nxs_file = parser.get<std::string>("file");
    std::unique_ptr<Reader> reader_ptr;
    reader_ptr = nxs_file.empty() ? std::make_unique<H5Read>()
                                    : std::make_unique<H5Read>(nxs_file);

    // Bind this as a reference
    Reader &reader = *reader_ptr;

    std::optional<reference_experiment> reference = std::nullopt;

#pragma region Reference experiment
    // Parse the reference experiment if provided.
    Experiment<MonochromaticBeam> reference_expt;
    bool use_reference_beam = false;
    bool use_reference_goniometer = false;
    bool use_reference_detector = false;
    if (parser.is_used("reference-geometry")){
        std::string reference_file = parser.get<std::string>("reference-geometry");
        std::ifstream f(reference_file);
        json reference_json;
        try {
            reference_json = json::parse(f);
        } catch (json::parse_error &ex) {
            logger.error("Unable to read {}; json parse error at byte {}",
                        reference_file.c_str(),
                        ex.byte);
            std::exit(1);
        }
        try {
            reference_expt = Experiment<MonochromaticBeam>(reference_json);
        } catch (std::invalid_argument const &ex) {
            logger.error("Unable to create MonochromaticBeam experiment: {}", ex.what());
            std::exit(1);
        }
        use_reference_beam = parser.get<bool>("reference.use-beam");
        use_reference_detector = parser.get<bool>("reference.use-detector");
        use_reference_goniometer = parser.get<bool>("reference.use-goniometer");
        reference = std::make_optional(
            reference_experiment(reference_expt, use_reference_beam, use_reference_detector, use_reference_goniometer));
    }
#pragma endregion

    Experiment<MonochromaticBeam> expt = make_experiment(reader, nxs_file, reference, std::make_optional(parser));
    
    json elist_out = expt.to_json();
    std::string efile_name = "imported.expt";
    std::ofstream efile(efile_name);
    efile << elist_out.dump(4);
    logger.info("Saved experiment list to {}", efile_name);
}
