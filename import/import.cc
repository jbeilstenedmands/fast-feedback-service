// This program reads a nexus file and creates experiment models

#include <argparse/argparse.hpp>
#include "ffs_logger.hpp"
#include "h5read.h"
#include <dx2/beam.hpp>
#include <dx2/crystal.hpp>
#include <dx2/detector.hpp>
#include <dx2/detector_attenuations.hpp>
#include <dx2/experiment.hpp>
#include <dx2/goniometer.hpp>
#include <dx2/imagesequence.hpp>
#include <dx2/scan.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

int main(int argc, char **argv) {
    // The purpose of an indexer is to determine the lattice model that best
    // explains the positions of the strong spots found during spot-finding.
    // The lattice model is a set of three vectors that define the crystal
    // lattice translations.
    // The experiment models (beam, detector) can also be refined during the
    // indexing process. The output is a set of models - a new crystal model that
    // describes the crystal lattice and an updated set of experiment models.
    auto t1 = std::chrono::system_clock::now();
    auto parser = argparse::ArgumentParser();
    parser.add_argument("-n", "--nxs").help("Path to the nexus file");
    parser.parse_args(argc, argv);
    

    if (!parser.is_used("nxs")) {
        logger.error("Must specify nexus file with --nxs\n");
        std::exit(1);
    }
    std::string nxs_file = parser.get<std::string>("nxs");
    std::unique_ptr<Reader> reader_ptr;
    //wait_for_ready_for_read(args.nxs, is_ready_for_read<H5Read>, wait_timeout);
    reader_ptr = nxs_file.empty() ? std::make_unique<H5Read>()
                                    : std::make_unique<H5Read>(nxs_file);

    // Bind this as a reference
    Reader &reader = *reader_ptr;

    auto wavelength_opt = reader.get_wavelength();
    if (!wavelength_opt) {
        fmt::print(
            "Error: No wavelength provided. Please pass wavelength using: "
            "--wavelength\n");
        std::exit(1);
    }
    double wavelength = wavelength_opt.value();
    logger.info("Got wavelength from file: {:.6f} Å", wavelength);
    auto [oscillation_start, oscillation_width] = reader.get_oscillation();
    if (oscillation_width > 0) {
        logger.info("Oscillation:  Start: {:.2f}°  Width: {:.2f}°",
                   oscillation_start, oscillation_width);
    }
    else {
        logger.info("Still-shot measurements");
    }
    int num_images = reader.get_number_of_images();
    logger.info("Number of images: {}", num_images);

    auto beam_center = reader.get_beam_center().value();
    auto pixel_size = reader.get_pixel_size().value();
    double distance = reader.get_detector_distance().value()*1000.0;
    double thickness = reader.get_detector_sensor_thickness().value()*1000.0;
    int height = reader.image_shape()[0];
    int width = reader.image_shape()[1];
    std::array<double, 2> beam_center_array = {
        static_cast<double>(beam_center[1]),
        static_cast<double>(beam_center[0])
    };
    std::array<double, 2> pixel_size_array = {
        static_cast<double>(pixel_size[0]) * 1000.0,
        static_cast<double>(pixel_size[1]) * 1000.0
    };
    // Need to extract these
    std::string fast_axis = "x";
    std::string slow_axis = "-y";
    // FIXME - need to extract sensor material and set it on the detector.
    std::string material = "Si";
    double mu = calculate_mu_for_material_at_wavelength(material, wavelength);

    Experiment<MonochromaticBeam> expt;
    expt.generate_identifier();

    MonochromaticBeam beam(wavelength);
    expt.set_beam(beam);
    Scan scan({1,num_images}, {oscillation_start, oscillation_width});
    expt.set_scan(scan);
    // FIXME get rotation axes and update gonio
    std::array<int, 2> image_size = {width, height};
    Panel panel(distance, beam_center_array, 
        pixel_size_array, image_size,
        fast_axis, slow_axis, thickness, mu);
    std::vector<Panel> panels = {panel};
    Detector detector(panels);
    expt.set_detector(detector);
    ImageSequence imagesequence(nxs_file, num_images);
    expt.set_imagesequence(imagesequence);

    json elist_out = expt.to_json();
    std::string efile_name = "imported.expt";
    std::ofstream efile(efile_name);
    efile << elist_out.dump(4);
    logger.info("Saved experiment list to {}", efile_name);
}
