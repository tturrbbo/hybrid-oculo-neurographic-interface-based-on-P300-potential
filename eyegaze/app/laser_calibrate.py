from __future__ import annotations

import argparse

from eyegaze.utils.config import load_config
from eyegaze.utils.video import camera
from eyegaze.calibration.laser_calibration import run_laser_calibration


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/experiment.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    laser_cfg = cfg.get("laser", {})
    cam_cfg = laser_cfg.get("camera", {})
    with camera(
        index=cam_cfg.get("index", 0),
        name=cam_cfg.get("name"),
        fallback_index=cam_cfg.get("fallback_index", 0),
        width=cam_cfg.get("width"),
        height=cam_cfg.get("height"),
        fps=cam_cfg.get("fps"),
    ) as cap:
        run_laser_calibration(cap, cfg)


if __name__ == "__main__":
    main()
