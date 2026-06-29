from __future__ import annotations

import argparse

from eyegaze.utils.config import load_config
from eyegaze.utils.screen import get_screen_size
from eyegaze.utils.video import camera, fullscreen
from eyegaze.tracking.gaze_estimator import GazeEstimator
from eyegaze.calibration.screen_calibration import run_screen_calibration
from eyegaze.tiles_stimulus.gui.gui import StimulusApp


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--participant", required=True)
    p.add_argument("--config", default="config/experiment.yaml")
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--auto", action="store_true", help="запустить автоматические random trials без ручной панели")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    screen_w, screen_h = get_screen_size()
    cam_cfg = cfg.get("camera", {})
    screen_cfg = cfg.get("screen", {})
    head_angle_cfg = cfg.get("head_angle", {})

    gaze = GazeEstimator(
        baseline_distance_cm=float(screen_cfg.get("baseline_distance_cm", 60.0)),
        tolerance_cm=float(screen_cfg.get("distance_tolerance_cm", 7.0)),
        head_angle_cfg=head_angle_cfg,
    )

    with camera(
        index=cam_cfg.get("index", 0),
        name=cam_cfg.get("name"),
        fallback_index=cam_cfg.get("fallback_index"),
        width=cam_cfg.get("width"),
        height=cam_cfg.get("height"),
        fps=cam_cfg.get("fps"),
    ) as cap:
        # Используем старую штатную калибровку экрана из eyegaze.
        with fullscreen("GazeScreenCalibration"):
            run_screen_calibration(
                gaze,
                cap,
                screen_w,
                screen_h,
                cfg,
                "GazeScreenCalibration",
            )

        # После калибровки запускаем именно модуль плиток из плтки.zip,
        # но с уже обученным GazeEstimator.
        app = StimulusApp(
            # По умолчанию оставляем оригинальную панель управления из плтки.zip:
            # цель, вспышка, пауза, число последовательностей и т.д.
            # Если нужен автоматический режим, запускай с --auto.
            auto_random_trials=bool(args.auto),
            auto_max_trials=int(args.trials),
            gaze_estimator=gaze,
            gaze_cap=cap,
            eyegaze_screen_w=screen_w,
            eyegaze_screen_h=screen_h,
            participant_id=args.participant,
            eyegaze_config=cfg,
        )
        app.run()


if __name__ == "__main__":
    main()
