import argparse

import os
import time
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

from CLAM.wsi_core.batch_process_utils import initialize_df
from CLAM.wsi_core.WholeSlideImage import WholeSlideImage
from CLAM.wsi_core.wsi_utils import StitchCoords


def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(
        file_path,
        wsi_object,
        downscale=downscale,
        bg_color=(0, 0, 0),
        alpha=-1,
        draw_grid=False,
    )
    total_time = time.time() - start

    return heatmap, total_time


def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None):
    ### Start Seg Timer
    start_time = time.time()
    # Use segmentation file
    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    # Segment
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(
    source,
    save_dir,
    patch_save_dir,
    mask_save_dir,
    stitch_save_dir,
    patch_size=256,
    step_size=256,
    seg_params={
        "seg_level": -1,
        "sthresh": 8,
        "mthresh": 7,
        "close": 4,
        "use_otsu": False,
        "keep_ids": "none",
        "exclude_ids": "none",
    },
    filter_params={"a_t": 100, "a_h": 16, "max_n_holes": 8},
    vis_params={"vis_level": -1, "line_thickness": 500},
    patch_params={"use_padding": True, "contour_fn": "four_pt"},
    patch_level=0,
    use_default_params=False,
    seg=False,
    save_mask=True,
    stitch=False,
    patch=False,
    auto_skip=True,
    process_list=None,
):

    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df["process"] == 1
    process_stack = df[mask]

    total = len(process_stack)

    legacy_support = "a" in df.keys()
    if legacy_support:
        logger.debug("Detected legacy segmentation csv file, legacy support enabled")
        df = df.assign(
            **{
                "a_t": np.full((len(df)), int(filter_params["a_t"]), dtype=np.uint32),
                "a_h": np.full((len(df)), int(filter_params["a_h"]), dtype=np.uint32),
                "max_n_holes": np.full(
                    (len(df)), int(filter_params["max_n_holes"]), dtype=np.uint32
                ),
                "line_thickness": np.full(
                    (len(df)), int(vis_params["line_thickness"]), dtype=np.uint32
                ),
                "contour_fn": np.full((len(df)), patch_params["contour_fn"]),
            }
        )

    seg_times = 0.0
    patch_times = 0.0
    stitch_times = 0.0

    for i in tqdm(range(total), disable=os.environ.get("DISABLE_PROGRESS_BAR", False)):
        df.to_csv(os.path.join(save_dir, "process_list_autogen.csv"), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, "slide_id"]
        logger.info("Processing ({}/{}) {}".format(i+1, total, slide))

        df.loc[idx, "process"] = 0
        slide_id, _ = os.path.splitext(slide)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + ".h5")):
            logger.info("{} already exist in destination location, skipped".format(slide_id))
            df.loc[idx, "status"] = "already_exist"
            continue

        # Inialize WSI
        full_path = os.path.join(source, slide)
        WSI_object = WholeSlideImage(full_path)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()

        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                if legacy_support and key == "vis_level":
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == "a_t":
                    old_area = df.loc[idx, "a"]
                    seg_level = df.loc[idx, "seg_level"]
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == "seg_level":
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params["vis_level"] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params["vis_level"] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params["vis_level"] = best_level

        if current_seg_params["seg_level"] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params["seg_level"] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params["seg_level"] = best_level

        keep_ids = str(current_seg_params["keep_ids"])
        if keep_ids != "none" and len(keep_ids) > 0:
            str_ids = current_seg_params["keep_ids"]
            current_seg_params["keep_ids"] = np.array(str_ids.split(",")).astype(int)
        else:
            current_seg_params["keep_ids"] = []

        exclude_ids = str(current_seg_params["exclude_ids"])
        if exclude_ids != "none" and len(exclude_ids) > 0:
            str_ids = current_seg_params["exclude_ids"]
            current_seg_params["exclude_ids"] = np.array(str_ids.split(",")).astype(int)
        else:
            current_seg_params["exclude_ids"] = []

        w, h = WSI_object.level_dim[current_seg_params["seg_level"]]
        if w * h > 1e8:
            logger.debug(
                "level_dim {} x {} is likely too large for successful segmentation, aborting".format(
                    w, h
                )
            )
            df.loc[idx, "status"] = "failed_seg"
            continue

        df.loc[idx, "vis_level"] = current_vis_params["vis_level"]
        df.loc[idx, "seg_level"] = current_seg_params["seg_level"]

        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(
                WSI_object, current_seg_params, current_filter_params
            )

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id + ".jpg")
            mask.save(mask_path)

        patch_time_elapsed = -1  # Default time
        if patch:
            current_patch_params.update(
                {
                    "patch_level": patch_level,
                    "patch_size": patch_size,
                    "step_size": step_size,
                    "save_path": patch_save_dir,
                }
            )
            file_path, patch_time_elapsed = patching(
                WSI_object=WSI_object,
                **current_patch_params,
            )

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id + ".h5")
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(
                    file_path, WSI_object, downscale=64
                )
                stitch_path = os.path.join(stitch_save_dir, slide_id + ".jpg")
                heatmap.save(stitch_path)

        logger.debug("Segmentation took {} seconds".format(seg_time_elapsed))
        logger.debug("Patching took {} seconds".format(patch_time_elapsed))
        logger.debug("Stitching took {} seconds".format(stitch_time_elapsed))
        df.loc[idx, "status"] = "processed"

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, "process_list_autogen.csv"), index=False)
    logger.info("Average segmentation time in s per slide: {}".format(seg_times))
    logger.info("Average patching time in s per slide: {}".format(patch_times))
    logger.info("Average stiching time in s per slide: {}".format(stitch_times))

    return seg_times, patch_times


# --- New preprocessing function ---
def _preprocess_seg_patch_args(
    source: str,
    step_size: int,
    patch_size: int,
    patch: bool,
    seg: bool,
    stitch: bool,
    auto_skip: bool,  # This is the raw argparse value
    save_dir: str,
    preset: str = None,
    patch_level: int = 0,
    process_list: str = None,
):
    """
    Preprocesses the raw arguments from UI/argparse into the format
    expected by the seg_and_patch function.
    """
    patch_save_dir = os.path.join(save_dir, "patches")
    mask_save_dir = os.path.join(save_dir, "masks")
    stitch_save_dir = os.path.join(save_dir, "stitches")

    # process_list needs to be absolute path if provided
    if process_list:
        # Assuming process_list is a filename within save_dir based on your original script
        _process_list_path = os.path.join(save_dir, process_list)
    else:
        _process_list_path = None

    directories = {
        "source": source,
        "save_dir": save_dir,
        "patch_save_dir": patch_save_dir,
        "mask_save_dir": mask_save_dir,
        "stitch_save_dir": stitch_save_dir,
    }

    # Ensure directories exist (except source)
    for key, val in directories.items():
        if key not in ["source"]:
            os.makedirs(val, exist_ok=True)

    # Default parameters
    seg_params = {
        "seg_level": -1,
        "sthresh": 8,
        "mthresh": 7,
        "close": 4,
        "use_otsu": False,
        "keep_ids": "none",
        "exclude_ids": "none",
    }
    filter_params = {"a_t": 100, "a_h": 16, "max_n_holes": 8}
    vis_params = {"vis_level": -1, "line_thickness": 250}
    patch_params = {"use_padding": True, "contour_fn": "four_pt"}

    # Override with preset if provided
    if preset:
        try:
            preset_df = pd.read_csv(preset)
            for key in seg_params.keys():
                seg_params[key] = preset_df.loc[0, key]
            for key in filter_params.keys():
                filter_params[key] = preset_df.loc[0, key]
            for key in vis_params.keys():
                vis_params[key] = preset_df.loc[0, key]
            for key in patch_params.keys():
                patch_params[key] = preset_df.loc[0, key]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Preset file '{preset}' not found in 'presets/' directory."
            )
        except Exception as e:
            raise ValueError(f"Error reading preset file '{preset}': {e}")

    parameters = {
        "seg_params": seg_params,
        "filter_params": filter_params,
        "patch_params": patch_params,
        "vis_params": vis_params,
    }

    return {
        "directories": directories,
        "parameters": parameters,
        "patch_size": patch_size,
        "step_size": step_size,
        "seg": seg,
        "use_default_params": False,  # Hardcoded in original script
        "save_mask": True,  # Hardcoded in original script
        "stitch": stitch,
        "patch_level": patch_level,
        "patch": patch,
        "process_list": _process_list_path,
        "auto_skip": auto_skip,
    }


parser = argparse.ArgumentParser(description="seg and patch")
parser.add_argument(
    "--source", type=str, help="path to folder containing raw wsi image files"
)
parser.add_argument("--step_size", type=int, default=256, help="step_size")
parser.add_argument("--patch_size", type=int, default=256, help="patch_size")
parser.add_argument("--patch", default=False, action="store_true")
parser.add_argument("--seg", default=False, action="store_true")
parser.add_argument("--stitch", default=False, action="store_true")
parser.add_argument(
    "--auto_skip",
    default=False,
    action="store_true",
    help="If enabled, automatically skip already processed files.",
)
parser.add_argument("--save_dir", type=str, help="directory to save processed data")
parser.add_argument(
    "--preset",
    default=None,
    type=str,
    help="predefined profile of default segmentation and filter parameters (.csv)",
)
parser.add_argument(
    "--patch_level", type=int, default=0, help="downsample level at which to patch"
)
parser.add_argument(
    "--process_list",
    type=str,
    default=None,
    help="name of list of images to process with parameters (.csv)",
)
parser.add_argument(
    "--log_level",
    type=str,
    default="INFO",
    help="log level",
)
parser.add_argument(
    "--disable_progress_bar",
    default=False,
    action="store_true",
    help="Disable the progress bar."
)

# This part is for standalone execution if you run this file directly
if __name__ == "__main__":
    args = parser.parse_args()
    
    logger.remove()

    logger.add(sys.stderr, level=args.log_level.upper(), format="{time:YYYY-MM-DD HH:mm:ss} | {message}") 

    # Call the preprocessing function
    processed_args = _preprocess_seg_patch_args(
        source=args.source,
        step_size=args.step_size,
        patch_size=args.patch_size,
        patch=args.patch,
        seg=args.seg,
        stitch=args.stitch,
        auto_skip=args.auto_skip,
        save_dir=args.save_dir,
        preset=args.preset,
        patch_level=args.patch_level,
        process_list=args.process_list,
    )
    
    os.environ["DISABLE_PROGRESS_BAR"] = str(args.disable_progress_bar)

    logger.debug("Directories:")
    logger.debug(processed_args["directories"])

    logger.debug("Parameters:")
    logger.debug(processed_args["parameters"])

    # Call seg_and_patch with the processed arguments
    seg_times, patch_times = seg_and_patch(
        **processed_args["directories"],
        **processed_args["parameters"],
        patch_size=processed_args["patch_size"],
        step_size=processed_args["step_size"],
        seg=processed_args["seg"],
        use_default_params=processed_args["use_default_params"],
        save_mask=processed_args["save_mask"],
        stitch=processed_args["stitch"],
        patch_level=processed_args["patch_level"],
        patch=processed_args["patch"],
        process_list=processed_args["process_list"],
        auto_skip=processed_args["auto_skip"],
    )
    logger.info(f"Segmentation Times: {seg_times}, Patching Times: {patch_times}")
