import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

import ffs.index # for make_panel
import ffs.integrate

def run(args=None):
    st = time.time()
    parser = argparse.ArgumentParser(
        prog="index",
        description="Runs standalone indexing of serial data using the GPU fast-feedback-indexer",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("-r", "--reflections", help="Path to the indexed.refl h5 file")
    parser.add_argument("-e", "--experiments", help="Path to the indexed.expt json")
    parser.add_argument("--test", action="store_true", help="Run in test mode")

    parsed = parser.parse_args(args)
    #min_spots = parsed.min_spots
    if not parsed.experiments:
        print("No imported experiment list provided.")
        return
    with open(parsed.experiments, "r") as f:
        expts = json.load(f)
    wavelength = expts["beam"][0]["wavelength"]
    crystals = expts["crystal"]
    s0 = np.asarray([0.0, 0.0, -1.0 / wavelength], dtype=np.float64)
    detector_dict = expts["detector"][0]["hierarchy"]
    # only single panel detectors for now.
    panel_dict = expts["detector"][0]["panels"][0]
    detector = {
        "distance": -1.0 * (detector_dict["origin"][2] + panel_dict["origin"][2]),
        "beam_center_x": -1.0
        * (detector_dict["origin"][0] + panel_dict["origin"][0])
        / panel_dict["pixel_size"][0],
        "beam_center_y": (detector_dict["origin"][1] + panel_dict["origin"][1])
        / panel_dict["pixel_size"][1],
        "pixel_size_x": panel_dict["pixel_size"][0],
        "pixel_size_y": panel_dict["pixel_size"][1],
        "image_size_x": panel_dict["image_size"][0],
        "image_size_y": panel_dict["image_size"][1],
        "thickness": panel_dict["thickness"],
        "mu": panel_dict["mu"],
    }

    if not parsed.reflections:
        print("No strong reflections h5 file provided.")
        return
    try:
        with h5py.File(parsed.reflections, "r") as refls:
            processing_group = refls["dials"]["processing"]["group_0"]
            xyz_obs = processing_group["xyzobs.px.value"][:]
            xyz_cal = processing_group["xyzcal.px"][:]
            spot_covars = processing_group["spot_covariance"][:]
            spot_mobs = processing_group["spot_mobs"][:]
            intensities = processing_group["intensity.sum.value"][:]
            miller_index = processing_group["miller_index"][:]
            ids = processing_group["id"][:]
            experiment_ids = processing_group.attrs["experiment_ids"]
            identifiers = processing_group.attrs["identifiers"]
            identifiers_map = dict(zip(experiment_ids, identifiers))
    except Exception as e:
        print(
            f"Unable to interpret the reflection file - please check input.\n Error: {e}"
        )
        return

    try:
        panel = ffs.index.make_panel(
            detector["distance"],
            detector["beam_center_x"],
            detector["beam_center_y"],
            detector["pixel_size_x"],
            detector["pixel_size_y"],
            detector["image_size_x"],
            detector["image_size_y"],
            detector["thickness"],
            detector["mu"],
        )
    except Exception as e:
        print(
            f"Unable to compose a detector panel model from the detector json.\n Error: {e}"
        )
        return

    tables = []
    id_values = []

    ## Note this assumes ids are in ascending order, which is the
    ## expected form of the output from spotfinding.
    unique_ids, start_indices = np.unique(ids, return_index=True)
    end_indices = np.append(start_indices[1:], len(ids))
    for id_, start, end in zip(unique_ids, start_indices, end_indices):
        xyz_obs_this = xyz_obs[start:end]
        xyz_cal_this = xyz_cal[start:end]
        covars_this = spot_covars[start:end]
        mobs_this = spot_mobs[start:end]
        intensities_this = intensities[start:end]
        miller_index_this = miller_index[start:end]
        if xyz_obs_this.any():
            tables.append((xyz_obs_this, xyz_cal_this, covars_this, intensities_this, miller_index_this, mobs_this))
            id_values.append(id_)

    ## Initialise the GPU integrator.
    '''try:
        indexer = GPUIndexer()
    except ModuleNotFoundError as e:  # if ffbidx not sourced
        print(f"ModuleNotFoundError: {e}")
        print(
            "ffbidx not found, has the fast-feedback-indexer module been built and sourced?"
        )
        return
    except ImportError as e:
        print(f"ImportError: {e}")
        print(
            "Potential source of this error: has the CUDA Runtime Library been loaded?"
        )
        return'''
    '''indexer.panel = panel
    indexer.cell = input_cell
    indexer.wavelength = wavelength

    # Quantities to log for log output
    n_indexed_images = 0
    n_total = len(tables)'''
    n_considered = 0

    t1 = time.time()

    for t, i, xtal in zip(tables, id_values, crystals):
        # data already has an initial prediction.
        xyzobs_this, xyzcal_this, covars_this, intensities_this, miller_index_this, mobs_this = t
        obs_flat = xyzobs_this.flatten()
        cal_flat = xyzcal_this.flatten()
        covars_flat = covars_this.flatten()
        #if t.shape[0] < min_spots:
        #    continue
        n_considered += 1
        
        covars = covars_flat.reshape(-1,3)
        sigma_b = np.mean((covars[0,:] + covars[1,:]) / 2)**0.5

        st = time.time()
        a = xtal["real_space_a"]
        b = xtal["real_space_b"]
        c = xtal["real_space_c"]
        A_inv = np.array([a,b,c], dtype="float64")
        A = np.linalg.inv(A_inv)
        ffs.integrate.ssx_integrate(xyzcal_this,
            xyzobs_this, covars_this, intensities_this, miller_index_this, mobs_this, s0, panel, A
        )
        end = time.time()
        print(f"{end-st:8f}s")


    t2 = time.time()
    print(f"Mosaicity models refined on {n_considered} images")


if __name__ == "__main__":
    st = time.time()
    run(sys.argv[1:])
    print(f"Program time: {time.time() - st:.3f}s")
