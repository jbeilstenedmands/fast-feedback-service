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
#include <format>
#include <sstream>

int main(int argc, char **argv) {
    // This program creates dx2 experiment models from nxmx-format data.
    auto t1 = std::chrono::system_clock::now();
    auto parser = argparse::ArgumentParser();
    parser.add_argument("file").help("Path to the nexus file").metavar("FILE.nxs");
    // Beam options
    parser.add_argument("-w", "--wavelength", "--beam.wavelength")
        .help("Wavelength of the X-ray beam (Å)")
        .scan<'f', float>();
    parser.add_argument("--beam.direction")
        .help("Sample to source direction of the X-ray beam")
        .nargs(3)
        .scan<'f', float>();
    parser.add_argument("--beam.divergence")
        .help("Divergence of the X-ray beam") // FIXME - units?
        .scan<'f', float>();
    parser.add_argument("--beam.sigma-divergence")
        .help("Sigma divergence of the X-ray beam") // FIXME - what exactly is this, what are units?
        .scan<'f', float>();
    parser.add_argument("--beam.polarization-normal")
        .help("Polarization vector of the X-ray beam")
        .nargs(3)
        .scan<'f', float>();
    parser.add_argument("--beam.polarization-fraction")
        .help("Polarization fraction of the X-ray beam")
        .scan<'f', float>();
    parser.add_argument("--beam.flux")
        .help("Incident flux of the X-ray beam") // FIXME - units?
        .scan<'f', float>();
    parser.add_argument("--beam.transmission")
        .help("Transmission of the X-ray beam") // FIXME - units?
        .scan<'f', float>();
    // Scan options
    parser.add_argument("--image-range", "--scan.image-range")
        .help("The subset of images from the nxs file for processing")
        .nargs(2)
        .scan<'i', int>();
    parser.add_argument("--scan.oscillation-start")
        .help("The starting angle of the scan (°)")
        .scan<'f', float>();
    parser.add_argument("--scan.oscillation-width")
        .help("The rotation width of each image in the scan (°)")
        .scan<'f', float>();
    // Goniometer options
    parser.add_argument("--axis", "--goniometer.single.axis") // primary indicator of single axis gonio.
        .nargs(3)
        .scan<'g', double>();
    parser.add_argument("--fixed-rotation", "--goniometer.single.fixed-rotation") // defaults to identitiy
        .nargs(9)
        .scan<'g', double>();
    parser.add_argument("--setting-rotation", "--goniometer.single.setting-rotation") // defaults to identitiy
        .nargs(9)
        .scan<'g', double>();
    parser.add_argument("--axes", "--goniometer.multi.axes") // primary indicator of multi axis gonio (if single axis given, then equivalent to single axis gonio).
        .nargs(argparse::nargs_pattern::at_least_one)
        .scan<'g', double>();
    parser.add_argument("--angles", "--goniometer.multi.angles") // defaults to 0 for each axis
        .nargs(argparse::nargs_pattern::at_least_one)
        .scan<'g', double>();
    parser.add_argument("--names", "--goniometer.multi.names") // defaults to "" for each axis
        .nargs(argparse::nargs_pattern::at_least_one);
    parser.add_argument("--scan-axis", "--goniometer.multi.scan-axis") // defaults to 0
        .scan<'u', uint32_t>();

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
    //wait_for_ready_for_read(args.nxs, is_ready_for_read<H5Read>, wait_timeout);
    reader_ptr = nxs_file.empty() ? std::make_unique<H5Read>()
                                    : std::make_unique<H5Read>(nxs_file);

    // Bind this as a reference
    Reader &reader = *reader_ptr;

    // Initialise the experiment and generate an identifier.
    Experiment<MonochromaticBeam> expt;
    expt.generate_identifier();

    // Now create the experiment models. The general pattern is to instantiate from a json object, as
    // this gives a neat way to aggregate items and flexibility to add further in future. This is
    // filled with values from the argument parser if available, else from the reader if the item
    // is available from the reader.

#pragma region Beam
    
    json beam_data;
    float wavelength;
    
    // wavelength MUST be provided/found, as there is no sensible default.
    if (parser.is_used("beam.wavelength")){
        wavelength = parser.get<float>("beam.wavelength");
    }
    else if (reader.get_wavelength().has_value()){
        wavelength = reader.get_wavelength().value();
    }
    else {
        throw std::runtime_error("No wavelength value found in file, and not provided as an input option with --beam.wavelength");
    }
    beam_data["wavelength"] = wavelength;
    
    // other beam properties can be defaults if not specified.
    // First check if they are specified as arguments, else try the reader if this property exists.
    if (parser.is_used("beam.direction")){
        beam_data["direction"] = parser.get<std::vector<float>>("beam.direction");
    }
    if (parser.is_used("beam.divergence")){
        beam_data["divergence"] = parser.get<float>("beam.divergence");
    }
    if (parser.is_used("beam.sigma-divergence")){
        beam_data["sigma_divergence"] = parser.get<float>("beam.sigma-divergence");
    }
    if (parser.is_used("beam.polarization-normal")){
        beam_data["polarization_normal"] = parser.get<std::vector<float>>("beam.polarization-normal");
    }
    if (parser.is_used("beam.polarization-fraction")){
        beam_data["polarization_fraaction"] = parser.get<float>("beam.polarization-fraction");
    }
    if (parser.is_used("beam.flux")){
        beam_data["flux"] = parser.get<float>("beam.flux");
    }
    if (parser.is_used("beam.transmission")){
        beam_data["transmission"] = parser.get<float>("beam.transmission");
    }

    MonochromaticBeam beam(beam_data);
    expt.set_beam(beam);
    logger.info("Created a monochromatic beam model with wavelength {:.6f}Å", wavelength);

#pragma endregion

#pragma region Scan

    // Initialise the scan directly, rather than from json, as it is a simpler object.
    int num_images = reader.get_number_of_images(); // Can be used in scan and also imagesequence
    std::array<int, 2> image_range = {1, num_images};
    std::array<double, 2> oscillation;
    
    if (parser.is_used("scan.image-range")){
        std::vector<int> parsed_image_range =  parser.get<std::vector<int>>("scan.image-range");
        if (parsed_image_range[0] < 1){
            throw std::invalid_argument(std::format(
                "The start of the image range specified ({}) must be >= 1", parsed_image_range[0]));
        }
        if (parsed_image_range[1] > num_images){
            throw std::invalid_argument(std::format(
                "The end of the image range specified ({}) must be <= the number of images ({})", parsed_image_range[1], num_images));
        }
        image_range = {parsed_image_range[0], parsed_image_range[1]};
    }

    auto [oscillation_start, oscillation_width] = reader.get_oscillation();
    if (parser.is_used("scan.oscillation-start")){
        oscillation[0] = static_cast<double>(parser.get<float>("scan.oscillation-start"));
    }
    else {
        oscillation[0] = oscillation_start;
    }
    if (parser.is_used("scan.oscillation-width")){
        oscillation[1] = static_cast<double>(parser.get<float>("scan.oscillation-width"));
    }
    else {
        oscillation[1] = oscillation_width;
    }

    Scan scan(image_range, oscillation);
    expt.set_scan(scan);
    logger.info("Created a scan model with image range {}:{}, oscillation start {:.2f}° and oscillation width {:.2f}°",
        image_range[0], image_range[1], oscillation[0], oscillation[1]);

#pragma endregion

#pragma region ImageSequence

    // Initialise with the number of images in the nxs file, regardless of if a narrower range has been chosed for the scan.
    ImageSequence imagesequence(nxs_file, num_images); 
    expt.set_imagesequence(imagesequence);

#pragma endregion

#pragma region Goniometer

    Goniometer goniometer;

    if (parser.is_used("goniometer.single.axis")){
        auto vec = parser.get<std::vector<double>>("goniometer.single.axis");
        Vector3d axis = {vec[0], vec[1], vec[2]};
        Matrix3d setting_rotation;
        Matrix3d fixed_rotation;
        setting_rotation << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
        fixed_rotation << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
        if (parser.is_used("goniometer.single.setting-rotation")){
            auto s_r = parser.get<std::vector<double>>("goniometer.single.setting-rotation");
            setting_rotation << s_r[0], s_r[1], s_r[2], s_r[3], s_r[4], s_r[5], s_r[6], s_r[7], s_r[8];
        }
        if (parser.is_used("goniometer.single.fixed-rotation")){
            auto f_r = parser.get<std::vector<double>>("goniometer.single.fixed-rotation");
            fixed_rotation << f_r[0], f_r[1], f_r[2], f_r[3], f_r[4], f_r[5], f_r[6], f_r[7], f_r[8];
        }
        goniometer = Goniometer(fixed_rotation, axis, setting_rotation);
        expt.set_goniometer(goniometer);
        logger.info("Created a single-axis goniometer model with rotation axis ({:.3f}, {:.3f}, {:.3f})", axis[0], axis[1], axis[2]);
    }
    else if (parser.is_used("goniometer.multi.axes")){
        std::vector<Vector3d> axes;
        std::vector<double> angles;
        std::vector<std::string> names;
        std::size_t scan_axis = 0;
        std::vector<double> input_axes = parser.get<std::vector<double>>("goniometer.multi.axes");
        if (input_axes.size() % 3 != 0){
            throw std::invalid_argument("The number of values input to parameter goniometer.multi.axes must be a multiple of three, as it is a list of vectors");
        }
        for (int i=0;i<input_axes.size() / 3;++i){
            axes.push_back(Vector3d(input_axes[i*3], input_axes[i*3+1], input_axes[i*3+2]));
        }
        if (parser.is_used("goniometer.multi.angles")){
            std::vector<double> input_angles = parser.get<std::vector<double>>("goniometer.multi.angles");
            if (input_angles.size() != axes.size()){
                throw std::invalid_argument(std::format("The number of angles provided ({}) must match the number of axes ({})", input_angles.size(), axes.size()));
            }
            angles = input_angles;
        }
        else {
            for (int i=0;i<axes.size();++i){
                angles.push_back(0.0);
            }
        }
        if (parser.is_used("goniometer.multi.names")){
            std::vector<std::string> input_names = parser.get<std::vector<std::string>>("goniometer.multi.names");
            if (input_names.size() != axes.size()){
                throw std::invalid_argument(std::format("The number of names provided ({}) must match the number of axes ({})", input_names.size(), axes.size()));
            }
            names = input_names;
        }
        else {
            for (int i=0;i<axes.size();++i){
                names.push_back("");
            }
        }
        if (parser.is_used("goniometer.multi.scan-axis")){
            scan_axis = static_cast<std::size_t>(parser.get<uint32_t>("goniometer.multi.scan-axis"));
            if (scan_axis >= axes.size()){
                throw std::invalid_argument(std::format("The specified scan axis index ({}) must be lower than the number of axes ({}).", scan_axis, axes.size()));
            }
        }
        goniometer = Goniometer(axes, angles, names, scan_axis);
        expt.set_goniometer(goniometer);
        logger.info("Created a multi-axis goniometer model");
    }
    else {
        logger.info("Defaulting to single-axis goniometer with rotation axis (1,0,0)");
        // Don't need to set anything as will use the default gonio model.
    }

#pragma endregion

    // FIXME detector 

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

    
    
    // FIXME get rotation axes and update gonio
    std::array<int, 2> image_size = {width, height};
    Panel panel(distance, beam_center_array, 
        pixel_size_array, image_size,
        fast_axis, slow_axis, thickness, mu);
    std::vector<Panel> panels = {panel};
    Detector detector(panels);
    expt.set_detector(detector);
    

    json elist_out = expt.to_json();
    std::string efile_name = "imported.expt";
    std::ofstream efile(efile_name);
    efile << elist_out.dump(4);
    logger.info("Saved experiment list to {}", efile_name);
}
