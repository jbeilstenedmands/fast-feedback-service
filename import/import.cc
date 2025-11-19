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


#pragma region Argument Parser
class ImportArgumentParser : public argparse::ArgumentParser {
    public:
        ImportArgumentParser(std::string version) : ArgumentParser("", version, argparse::default_arguments::help) {
            add_importer_arguments();
        }
    private:
        void add_importer_arguments(){
            add_argument("file").help("Path to the nexus file").metavar("FILE.nxs");
            // Reference geometry options
            add_argument("--reference-geometry")
                .help("Path to a reference experiment geometry");
            add_argument("--reference.use-beam")
                .help("Control whether the beam model from the reference is used")
                .nargs(0, 1)
                .default_value(true)
                .implicit_value(true);
            add_argument("--reference.use-goniometer")
                .help("Control whether the goniometer model from the reference is used")
                .nargs(0, 1)
                .default_value(true)
                .implicit_value(true);
            add_argument("--reference.use-detector")
                .help("Control whether the detector model from the reference is used")
                .nargs(0, 1)
                .default_value(true)
                .implicit_value(true);
            // Beam options
            add_argument("-w", "--wavelength", "--beam.wavelength")
                .help("Wavelength of the X-ray beam (Å)")
                .scan<'f', float>();
            add_argument("--beam.direction")
                .help("Sample to source direction of the X-ray beam")
                .nargs(3)
                .scan<'f', float>();
            add_argument("--beam.divergence")
                .help("Divergence of the X-ray beam") // FIXME - units?
                .scan<'f', float>();
            add_argument("--beam.sigma-divergence")
                .help("Sigma divergence of the X-ray beam") // FIXME - what exactly is this, what are units?
                .scan<'f', float>();
            add_argument("--beam.polarization-normal")
                .help("Polarization vector of the X-ray beam")
                .nargs(3)
                .scan<'f', float>();
            add_argument("--beam.polarization-fraction")
                .help("Polarization fraction of the X-ray beam")
                .scan<'f', float>();
            add_argument("--beam.flux")
                .help("Incident flux of the X-ray beam") // FIXME - units?
                .scan<'f', float>();
            add_argument("--beam.transmission")
                .help("Transmission of the X-ray beam") // FIXME - units?
                .scan<'f', float>();
            // Scan options
            add_argument("--image-range", "--scan.image-range")
                .help("The subset of images from the nxs file for processing")
                .nargs(2)
                .scan<'i', int>();
            add_argument("--scan.oscillation-start")
                .help("The starting angle of the scan (°)")
                .scan<'f', float>();
            add_argument("--scan.oscillation-width")
                .help("The rotation width of each image in the scan (°)")
                .scan<'f', float>();
            // Goniometer options
            add_argument("--axis", "--goniometer.single.axis") // primary indicator of single axis gonio.
                .nargs(3)
                .scan<'g', double>();
            add_argument("--fixed-rotation", "--goniometer.single.fixed-rotation") // defaults to identitiy
                .nargs(9)
                .scan<'g', double>();
            add_argument("--setting-rotation", "--goniometer.single.setting-rotation") // defaults to identitiy
                .nargs(9)
                .scan<'g', double>();
            add_argument("--axes", "--goniometer.multi.axes") // primary indicator of multi axis gonio (if single axis given, then equivalent to single axis gonio).
                .nargs(argparse::nargs_pattern::at_least_one)
                .scan<'g', double>();
            add_argument("--angles", "--goniometer.multi.angles") // defaults to 0 for each axis
                .nargs(argparse::nargs_pattern::at_least_one)
                .scan<'g', double>();
            add_argument("--names", "--goniometer.multi.names") // defaults to "" for each axis
                .nargs(argparse::nargs_pattern::at_least_one);
            add_argument("--scan-axis", "--goniometer.multi.scan-axis") // defaults to 0
                .scan<'u', uint32_t>();
        }
};

#pragma endregion


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
    //wait_for_ready_for_read(args.nxs, is_ready_for_read<H5Read>, wait_timeout);
    reader_ptr = nxs_file.empty() ? std::make_unique<H5Read>()
                                    : std::make_unique<H5Read>(nxs_file);

    // Bind this as a reference
    Reader &reader = *reader_ptr;

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
        use_reference_goniometer = parser.get<bool>("reference.use-goniometer");
        use_reference_detector = parser.get<bool>("reference.use-detector");
    }
#pragma endregion

    // Initialise the experiment and generate an identifier.
    Experiment<MonochromaticBeam> expt;
    expt.generate_identifier();

    // Now create the experiment models. The general pattern is to instantiate from a json object, as
    // this gives a neat way to aggregate items and flexibility to add further in future. This is
    // filled with values from the argument parser if available, else from the reader if the item
    // is available from the reader.

#pragma region Beam
    json beam_data;
    beam_data["wavelength"] = 0.0;
    // If we have a reference geometry, use this to create a beam data json, else load as
    // much as we can from the reader.
    // These are then updated later with any custom options.
    if (use_reference_beam){
        beam_data = reference_expt.beam().to_json();
        logger.info("Using reference beam model as starting model.");
    }
    else {
        if (reader.get_wavelength().has_value()){
            beam_data["wavelength"] = reader.get_wavelength().value();
        }
        // FIXME add in the rest of the reader beam properties here.
    }
    
    // Now check parser options.
    if (parser.is_used("beam.wavelength")){
        beam_data["wavelength"] = parser.get<float>("beam.wavelength");
    }
    // wavelength MUST be provided/found, as there is no sensible default.
    if (beam_data["wavelength"] <= 0.0){
        throw std::runtime_error("No wavelength value found in file, and not provided as an input option with --beam.wavelength or --reference_geometry");
    }
    
    // other beam properties fall back to defaults if not specified.
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
    double wavelength = beam_data["wavelength"];
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
    json goniometer_data;
    // If we have a reference geometry, use this to create a goniometer data json, else load as
    // much as we can from the reader.
    // These are then updated later with any custom options.
    // Need to be slightly careful about single/multi axis goniometers - we only want one type of data to be
    // in the goniometer json.
    if (use_reference_goniometer){
        goniometer_data = reference_expt.goniometer().to_json();
        logger.info("Using reference goniometer model as starting model.");
    }
    /*else {
        //FIXME get reader to parse nxs file for goniometer options.
    }*/

    if (parser.is_used("goniometer.single.axis")){
        if (goniometer_data.contains("axes")){
            throw std::invalid_argument("A multi-axis reference goniometer has been provided alongside single-axis options.");
        }
        std::vector<double> axis = parser.get<std::vector<double>>("goniometer.single.axis");
        goniometer_data["rotation_axis"] = axis;
        std::vector<double> setting_rotation = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        std::vector<double> fixed_rotation = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        if (parser.is_used("goniometer.single.setting-rotation")){
            setting_rotation = parser.get<std::vector<double>>("goniometer.single.setting-rotation");
        }
        // FIXME else get from reader?
        goniometer_data["setting_rotation"] = setting_rotation;
        if (parser.is_used("goniometer.single.fixed-rotation")){
            fixed_rotation = parser.get<std::vector<double>>("goniometer.single.fixed-rotation");
        }
        // FIXME else get from reader?
        goniometer_data["fixed_rotation"] = fixed_rotation;
        
        goniometer = Goniometer(goniometer_data);
        expt.set_goniometer(goniometer);
        logger.info("Created a single-axis goniometer model with rotation axis ({:.3f}, {:.3f}, {:.3f})", axis[0], axis[1], axis[2]);
    }
    else if (parser.is_used("goniometer.multi.axes")){
        if (goniometer_data.contains("axis")){
            throw std::invalid_argument("A single-axis reference goniometer has been provided alongside multi-axis options.");
        }
        std::vector<double> input_axes = parser.get<std::vector<double>>("goniometer.multi.axes");
        if (input_axes.size() % 3 != 0){
            throw std::invalid_argument("The number of values input to parameter goniometer.multi.axes must be a multiple of three, as it is a list of vectors");
        }
        std::vector<Vector3d> axes; 
        int n_axes = input_axes.size() / 3;
        for (int i=0;i<n_axes; ++i){
            axes.push_back(Vector3d{input_axes[i*3],input_axes[i*3+1],input_axes[i*3+2]});
        }
        goniometer_data["axes"] = axes;
    
        if (parser.is_used("goniometer.multi.angles")){
            std::vector<double> input_angles = parser.get<std::vector<double>>("goniometer.multi.angles");
            if (input_angles.size() != n_axes){
                throw std::invalid_argument(std::format("The number of angles provided ({}) must match the number of axes ({})", input_angles.size(), n_axes));
            }
            goniometer_data["angles"] = input_angles;
        }
        else if (!(goniometer_data.contains("angles"))){
            std::vector<double> angles;
            for (int i=0;i<n_axes;++i){
                angles.push_back(0.0);
            }
            goniometer_data["angles"] = angles;
        }
        if (parser.is_used("goniometer.multi.names")){
            std::vector<std::string> input_names = parser.get<std::vector<std::string>>("goniometer.multi.names");
            if (input_names.size() != n_axes){
                throw std::invalid_argument(std::format("The number of names provided ({}) must match the number of axes ({})", input_names.size(), n_axes));
            }
            goniometer_data["names"] = input_names;
        }
        else if (!(goniometer_data.contains("names"))){
            std::vector<std::string> names;
            for (int i=0;i<n_axes;++i){
                names.push_back("");
            }
            goniometer_data["names"] = names;
        }
        if (parser.is_used("goniometer.multi.scan-axis")){
            std::size_t scan_axis = static_cast<std::size_t>(parser.get<uint32_t>("goniometer.multi.scan-axis"));
            if (scan_axis >= n_axes){
                throw std::invalid_argument(std::format("The specified scan axis index ({}) must be lower than the number of axes ({}).", scan_axis, n_axes));
            }
            goniometer_data["scan_axis"] = scan_axis;
        }
        else if (!(goniometer_data.contains("scan_axis"))){
            std::size_t scan_axis = 0;
            goniometer_data["scan_axis"] = scan_axis;
        }
        goniometer = Goniometer(goniometer_data);
        expt.set_goniometer(goniometer);
        logger.info("Created a multi-axis goniometer model");
    }
    else {
        logger.info("Defaulting to single-axis goniometer with rotation axis (1,0,0)");
        // Don't need to set anything as will use the default gonio model.
    }

#pragma endregion

    // FIXME detector
    // The simplest way to configure a detector is with distance and beam centre, then assuming

#pragma region Detector
    json panel_data;
    json detector_data;
    // full spec - fast, slow, origin, pixel_size, image_size, trusted_range, type, name, thickness, mu,
    // raw_image_offset, pedestal, pxmm strategy.
    
    if (use_reference_detector){
        detector_data = reference_expt.detector().to_json();
        if (detector_data["panels"].size() > 1){
            throw std::invalid_argument("The reference detector is multi-panel, only single panel detectors are currently supported.");
        }
        logger.info("Constructing detector model from reference experiment.");
    }
    else {
        // Get from the reader.
        // Assuming single panel.
        // FIXME - need to extract sensor material and set it on the detector.
        std::string material = "Si";
        double mu = calculate_mu_for_material_at_wavelength(material, beam_data["wavelength"]);
        panel_data["mu"] = mu;
        double thickness = reader.get_detector_sensor_thickness().value()*1000.0;
        panel_data["thickness"] = thickness;

        auto beam_center = reader.get_beam_center().value();
        auto pixel_size = reader.get_pixel_size().value();
        double distance = reader.get_detector_distance().value()*1000.0;
        
        panel_data["distance"] = distance;
        //double thickness = reader.get_detector_sensor_thickness().value()*1000.0;
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
        panel_data["beam_center"] = beam_center_array;
        panel_data["pixel_size"] = pixel_size_array;
        // Need to extract these
        //std::string fast_axis = "x";
        //std::string slow_axis = "-y";
        panel_data["fast_axis"] = std::vector<double>{1.0,0.0,0.0};
        panel_data["slow_axis"] = std::vector<double>{0.0,-1.0,0.0};
        std::array<int, 2> image_size = {width, height};
        panel_data["image_size"] = image_size;
        panel_data["trusted_range"] = std::array<double, 2>{0.0, 65536.0};
        panel_data["type"] = "SENSOR_PAD";
        panel_data["name"] = "module";
        panel_data["raw_image_offset"] = std::array<int, 2>{0,0};
        panel_data["gain"] = 1.0;
        panel_data["pedestal"] = 0.0;
        panel_data["px_mm_strategy"] = {{"type", "ParallaxCorrectedPxMmStrategy"}};
        std::vector<json> panels_array = {panel_data};
        detector_data["panels"] = panels_array;
        logger.info("Constructing detector model from reader");
    }

    
    /*Panel panel(distance, beam_center_array, 
        pixel_size_array, image_size,
        fast_axis, slow_axis, thickness, mu);
    std::vector<Panel> panels = {panel};*/
    Detector detector(detector_data);
    expt.set_detector(detector);

#pragma endregion

    json elist_out = expt.to_json();
    std::string efile_name = "imported.expt";
    std::ofstream efile(efile_name);
    efile << elist_out.dump(4);
    logger.info("Saved experiment list to {}", efile_name);
}
