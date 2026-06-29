import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

from psychopy import core, event, visual

from eyegaze.tiles_stimulus import config
from eyegaze.tiles_stimulus.core.grid import Grid
from eyegaze.tiles_stimulus.core.stimulus_controller import StimulusController

try:
    from psychopy.visual.textbox2 import TextBox2
except Exception:  # разные пути в сборках PsychoPy
    TextBox2 = getattr(visual, "TextBox2", None)  # type: ignore[assignment]
if TextBox2 is None:
    raise ImportError("Требуется psychopy.visual.textbox2.TextBox2 (установите psychopy >= 3)")


def _parse_float(
    s: str,
    default: float,
    lo: float,
    hi: float,
) -> float:
    try:
        v = float(str(s).strip().replace(",", "."))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _parse_int(s: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(float(str(s).strip().replace(",", ".")))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _clamp_int(v: int, lo: int, hi: int, *, default: int) -> int:
    try:
        x = int(v)
    except (TypeError, ValueError):
        x = int(default)
    return max(int(lo), min(int(hi), x))


class StimulusApp:
    def __init__(
        self,
        *,
        auto_random_trials: bool = False,
        inter_trial_s: float = 1.0,
        auto_plan_trials: int = 15,
        auto_plan_target_tile_id: int = 4,
        auto_plan_target_repeats: int = 0,
        auto_plan_target_epochs: int = 12,
        sequences_override: int | None = None,
        auto_max_trials: int | None = None,
        stim_control_dir: str | Path | None = None,
        gaze_estimator=None,
        gaze_cap=None,
        eyegaze_screen_w: int | None = None,
        eyegaze_screen_h: int | None = None,
        participant_id: str = "P01",
        eyegaze_config: dict | None = None,
    ) -> None:
        self.auto_random_trials = bool(auto_random_trials)
        self.stim_control_dir = Path(stim_control_dir) if stim_control_dir else None

        # Интеграция с существующей калибровкой eyegaze.
        # ВАЖНО: здесь НЕ создаётся новая калибровка взгляда.
        # Используется уже обученный GazeEstimator после run_screen_calibration().
        self.gaze_estimator = gaze_estimator
        self.gaze_cap = gaze_cap
        self.eyegaze_screen_w = eyegaze_screen_w
        self.eyegaze_screen_h = eyegaze_screen_h
        self.participant_id = str(participant_id)
        self.eyegaze_config = eyegaze_config or {}
        self._gaze_writer = None
        self._gaze_log = None
        self._gaze_text = None
        self._gaze_dot = None
        self._gaze_last_result = None
        self._gaze_last_predicted_tile = None
        self._gaze_smooth_xy = None
        self._gaze_last_good_xy = None
        self._gaze_lost_frames = 0

        # Размер красной точки/жёлтого пятна задаётся через зрительный угол.
        # Базовый радиус считается из:
        # angle_deg + reference_distance_cm + физический размер экрана.
        self._angular_spot_base_radius_px = None
        self._angular_spot_current_radius_px = None
        self._angular_spot_reference_cm = None
        self._angular_spot_last_applied_cm = None
        self._stim_control_wait_trial = False
        self._stim_control_last_ms = 0
        self.inter_trial_s = max(0.0, float(inter_trial_s))
        self.auto_plan_trials = max(0, int(auto_plan_trials))
        self.auto_plan_target_tile_id = int(auto_plan_target_tile_id)
        self.auto_plan_target_repeats = max(0, int(auto_plan_target_repeats))
        self.auto_plan_target_epochs = max(0, int(auto_plan_target_epochs))
        self._auto_trials_started = 0
        self._overlay_exp_index = 0
        self._overlay_exp_total = 0
        self._overlay_exp_calib = False
        self._auto_target_plan: list[int] = []
        self._auto_pause_until: float | None = None
        self._auto_next_target: int | None = None
        self._auto_max_trials = int(auto_max_trials) if auto_max_trials is not None else None
        self._sequences_override = int(sequences_override) if sequences_override is not None else None
        self.win = visual.Window(
            screen=int(getattr(config, "SCREEN_INDEX", 1)),
            size=config.WINDOW_SIZE,
            color=config.WINDOW_COLOR,
            units="pix",
            fullscr=bool(getattr(config, "FULLSCREEN", False)),
        )
        self.grid = Grid(size=config.GRID_SIZE)
        self.controller = StimulusController(
            self.grid,
            flash_duration=config.DEFAULT_FLASH_DURATION,
            isi=config.DEFAULT_ISI,
            cue_duration=config.DEFAULT_CUE_S,
            ready_duration=config.DEFAULT_READY_S,
            inter_block_s=config.DEFAULT_INTER_BLOCK_S,
            cue_color=config.CUE_COLOR,
            stim_color=config.STIM_COLOR,
        )
        # Автопротокол: без клика START сразу идём в полный экран и циклы trial с случайной целью.
        self.show_controls = not self.auto_random_trials
        # С stim_control первый trial даёт протокол, не автостарт.
        self._auto_pending_first_trial = bool(self.auto_random_trials) and self.stim_control_dir is None
        self._tiles_visual: list = []
        self._tile_texts: list = []
        self._build_visual_grid()
        self.start_button = visual.Rect(
            self.win,
            width=config.BUTTON_WIDTH,
            height=config.BUTTON_HEIGHT,
            pos=config.START_BUTTON_POS,
            fillColor="green",
        )
        self.stop_button = visual.Rect(
            self.win,
            width=config.BUTTON_WIDTH,
            height=config.BUTTON_HEIGHT,
            pos=config.STOP_BUTTON_POS,
            fillColor="red",
        )
        self.start_text = visual.TextStim(
            self.win, text="START", pos=config.START_BUTTON_POS, color="black", height=18
        )
        self.stop_text = visual.TextStim(
            self.win, text="STOP", pos=config.STOP_BUTTON_POS, color="black", height=18
        )
        self.mouse = event.Mouse(win=self.win)
        if self.auto_random_trials:
            self.win.mouseVisible = False

        px = self._right_panel_x()
        y = config.PANEL_FIRST_ROW_Y
        dy = config.PANEL_ROW_DY
        label_x = px + config.PANEL_LABEL_OFFSET

        def row() -> float:
            nonlocal y
            cur = y
            y -= dy
            return cur

        self._labels: List[visual.TextStim] = []
        self._tbs: List[TextBox2] = []

        def add_row(caption: str, val: str) -> TextBox2:
            ry = row()
            self._labels.append(
                visual.TextStim(
                    self.win,
                    text=caption,
                    pos=(label_x, ry),
                    color="white",
                    height=14,
                    alignText="left",
                )
            )
            tb: TextBox2 = TextBox2(
                self.win,
                text=val,
                pos=(px, ry),
                size=(config.PANEL_TB_W, config.PANEL_TB_H),
                units="pix",
                color="white",
                fillColor="#1e1e1e",
                borderColor="#666666",
                font="Arial",
                letterHeight=config.PANEL_LETTER_H,
                editable=True,
            )
            self._tbs.append(tb)
            return tb

        self.tb_cue = add_row("Показ цели (с)", f"{config.DEFAULT_CUE_S:g}")
        self.tb_ready = add_row("Пауза+крест (с)", f"{config.DEFAULT_READY_S:g}")
        self.tb_isi = add_row("ISI (с)", f"{config.DEFAULT_ISI:.2f}")
        self.tb_flash = add_row("Вспышка (с)", f"{config.DEFAULT_FLASH_DURATION:.2f}")
        self.tb_inter = add_row("Между рядами (с)", f"{config.DEFAULT_INTER_BLOCK_S:.2f}")
        self.tb_seq = add_row("Раунды", f"{config.DEFAULT_SEQUENCES}")
        self.tb_target = add_row("Цель 0–8", f"{config.DEFAULT_TARGET_ID}")

        # Apply sequences override for auto-protocol so operator changes in protocol_runner_gui take effect.
        if self._sequences_override is not None:
            seq = _clamp_int(
                self._sequences_override,
                config.SEQUENCES_MIN,
                config.SEQUENCES_MAX,
                default=config.DEFAULT_SEQUENCES,
            )
            self.tb_seq.text = str(seq)

        self.hint = visual.TextStim(
            self.win,
            text="Space — остановка сессии  ·  Esc — выход из программы",
            pos=(0, config.OPERATOR_HINT_Y),
            color="#888888",
            height=config.OPERATOR_HINT_H,
        )

        # В авто-режиме строим план целей только после того, как инициализированы tb_seq и др. поля.
        if self.auto_random_trials:
            self._auto_target_plan = self._build_auto_target_plan()

        self.fixation_cross = visual.ShapeStim(
            self.win,
            vertices=(
                (0, -config.FIXATION_CROSS_SIZE / 2),
                (0, config.FIXATION_CROSS_SIZE / 2),
                (0, 0),
                (-config.FIXATION_CROSS_SIZE / 2, 0),
                (config.FIXATION_CROSS_SIZE / 2, 0),
            ),
            closeShape=False,
            lineWidth=2,
            lineColor=config.FIXATION_CROSS_COLOR,
            pos=(0, 0),
        )

        # Overlay между trial (как 3a0a997): серый fullscreen + номер плитки.
        w, h = float(self.win.size[0]), float(self.win.size[1])
        title_h = max(32, int(h * 0.10 / 1.5))
        sub_h = max(24, int(h * 0.085 / 1.5))
        self._overlay_force_show = False
        self._overlay_title = ""
        self._overlay_sub = ""
        self._overlay_target_id: int | None = None
        self._auto_overlay_title = visual.TextStim(
            self.win,
            text="",
            pos=(0, int(h * 0.18)),
            color="white",
            height=title_h,
            wrapWidth=w * 0.9,
        )
        self._auto_overlay_sub = visual.TextStim(
            self.win,
            text="",
            pos=(0, int(h * -0.02)),
            color="white",
            height=sub_h,
            wrapWidth=w * 0.9,
        )
        self._auto_overlay_target_num = visual.TextStim(
            self.win,
            text="",
            pos=(0, int(h * -0.22)),
            color="yellow",
            height=max(28, int((h * 0.44) / 3.0)),
            bold=True,
        )
        self._auto_overlay_bg = visual.Rect(
            self.win,
            width=w,
            height=h,
            pos=(0, 0),
            fillColor="#202020",
            lineColor=None,
            opacity=1.0,
        )

    def _right_panel_x(self) -> float:
        return config.PANEL_X_FRACTION * (self.win.size[0] * 0.5)

    def _build_visual_grid(self) -> None:
        """
        Плитки из вашего gui/gui.py остаются теми же PsychoPy Rect/TextStim.
        Изменено только размещение: теперь они стоят по окантовке широкого экрана.
        """
        tile_size = float(config.TILE_SIZE_PX) * float(config.TILE_DISPLAY_SCALE)

        w, h = float(self.win.size[0]), float(self.win.size[1])
        half_w, half_h = w / 2.0, h / 2.0

        edge = float(getattr(config, "PERIMETER_EDGE_MARGIN_PX", 70))
        corner = float(getattr(config, "PERIMETER_CORNER_MARGIN_PX", 120))

        top_count = int(getattr(config, "PERIMETER_TOP_COUNT", 7))
        bottom_count = int(getattr(config, "PERIMETER_BOTTOM_COUNT", 7))
        right_count = int(getattr(config, "PERIMETER_RIGHT_COUNT", 3))
        left_count = int(getattr(config, "PERIMETER_LEFT_COUNT", 3))

        def linspace(a: float, b: float, n: int) -> list[float]:
            if n <= 0:
                return []
            if n == 1:
                return [(a + b) / 2.0]
            return [a + (b - a) * i / (n - 1) for i in range(n)]

        positions: list[tuple[float, float]] = []
        left_x = -half_w + corner
        right_x = half_w - corner
        top_y = half_h - edge
        bottom_y = -half_h + edge

        # top: left -> right
        for x in linspace(left_x, right_x, top_count):
            positions.append((x, top_y))

        # right: top -> bottom
        for y in linspace(top_y - corner, bottom_y + corner, right_count):
            positions.append((right_x, y))

        # bottom: right -> left
        for x in reversed(linspace(left_x, right_x, bottom_count)):
            positions.append((x, bottom_y))

        # left: bottom -> top
        for y in reversed(linspace(top_y - corner, bottom_y + corner, left_count)):
            positions.append((left_x, y))

        # если плиток в Grid больше, чем позиций, лишние не рисуем
        self.grid.tiles = self.grid.tiles[:len(positions)]

        for tile, (x, y) in zip(self.grid.tiles, positions):
            tile.x = float(x)
            tile.y = float(y)
            # normalized screen coords: 0..1, y вниз как в gaze_xy
            tile.x_norm = (float(x) + half_w) / max(1.0, w)
            tile.y_norm = (half_h - float(y)) / max(1.0, h)

            rect = visual.Rect(
                self.win,
                width=tile_size,
                height=tile_size,
                pos=(x, y),
                fillColor=config.TILE_DEFAULT_COLOR,
                lineColor=config.TILE_LINE_COLOR,
            )
            tile_text = visual.TextStim(
                self.win,
                text=str(tile.id),
                pos=(x, y),
                color="white",
                height=tile_size * 0.25,
            )
            self._tiles_visual.append(rect)
            self._tile_texts.append(tile_text)


    def _read_settings(self) -> Tuple[int, int]:
        self.controller.cue_duration = _parse_float(
            self.tb_cue.text,
            config.DEFAULT_CUE_S,
            config.CUE_MIN,
            config.CUE_MAX,
        )
        self.tb_cue.text = f"{self.controller.cue_duration:.3g}"
        self.controller.ready_duration = _parse_float(
            self.tb_ready.text,
            config.DEFAULT_READY_S,
            config.READY_MIN,
            config.READY_MAX,
        )
        self.tb_ready.text = f"{self.controller.ready_duration:.3g}"
        self.controller.isi = _parse_float(
            self.tb_isi.text, config.DEFAULT_ISI, config.ISI_MIN, config.ISI_MAX
        )
        self.tb_isi.text = f"{self.controller.isi:.2f}"
        self.controller.flash_duration = _parse_float(
            self.tb_flash.text,
            config.DEFAULT_FLASH_DURATION,
            config.FLASH_MIN,
            config.FLASH_MAX,
        )
        self.tb_flash.text = f"{self.controller.flash_duration:.2f}"
        self.controller.inter_block_s = _parse_float(
            self.tb_inter.text,
            config.DEFAULT_INTER_BLOCK_S,
            config.INTER_BLOCK_MIN,
            config.INTER_BLOCK_MAX,
        )
        self.tb_inter.text = f"{self.controller.inter_block_s:.2f}"
        seq = _parse_int(
            self.tb_seq.text,
            config.DEFAULT_SEQUENCES,
            config.SEQUENCES_MIN,
            config.SEQUENCES_MAX,
        )
        self.tb_seq.text = str(seq)
        tgt = _parse_int(
            self.tb_target.text,
            config.DEFAULT_TARGET_ID,
            0,
            len(self.grid.tiles) - 1,
        )
        self.tb_target.text = str(tgt)
        return seq, tgt

    def _start_trial_with_target(self, target_tile_id: int) -> None:
        """Один trial: trial_config в LSL (для логов) + start_experiment с фиксированной целью."""
        sequences, _ = self._read_settings()
        if sequences <= 0:
            return
        tgt = int(target_tile_id)
        tgt = max(0, min(len(self.grid.tiles) - 1, tgt))
        self.tb_target.text = str(tgt)
        cfg_payload = (
            f"trial_config|target={tgt};sequences={sequences};"
            f"isi_s={self.controller.isi:.3f};flash_s={self.controller.flash_duration:.3f};"
            f"cue_s={self.controller.cue_duration:.3f};ready_s={self.controller.ready_duration:.3f};"
            f"inter_block_s={self.controller.inter_block_s:.3f};grid={self.grid.size}x{self.grid.size}"
        )
        self.win.callOnFlip(self.controller.lsl.send, -3, cfg_payload)
        self.show_controls = False
        self.win.mouseVisible = False
        self._clear_waiting_overlay()
        self.controller.start_experiment(sequences, target_tile_id=tgt)
        core.wait(0.2)

    def _start_trial_random_target(self) -> None:
        """Авто-цель на каждый trial.

        В первые auto_plan_trials trial используем план целей, чтобы нужная плитка встретилась
        auto_plan_target_repeats раз, но не подряд (для набора шаблона без монотонности).
        После окончания плана — обычный рандом.
        """
        tgt: int
        if self._auto_trials_started < len(self._auto_target_plan):
            tgt = int(self._auto_target_plan[self._auto_trials_started])
        else:
            prev = None
            if self._auto_trials_started > 0:
                prev = int(self._auto_target_plan[-1]) if self._auto_target_plan else None
            tgt = self._rand_target_avoid(prev=prev)
        self._auto_trials_started += 1
        self._start_trial_with_target(tgt)

    def _schedule_auto_trial(self, *, target_tile_id: int) -> None:
        """Show overlay during pause, then start trial with given target."""
        tgt = max(0, min(len(self.grid.tiles) - 1, int(target_tile_id)))
        self._auto_next_target = tgt
        if self.inter_trial_s <= 0:
            self._auto_pause_until = None
            self._auto_next_target = None
            self._start_trial_with_target(tgt)
            return
        self._auto_pause_until = float(core.getTime()) + float(self.inter_trial_s)

    def _auto_pause_active(self) -> bool:
        if self._auto_pause_until is None:
            return False
        now = float(core.getTime())
        if now < float(self._auto_pause_until):
            return True
        # pause finished
        self._auto_pause_until = None
        if self._auto_next_target is not None:
            tgt = int(self._auto_next_target)
            self._auto_next_target = None
            self._start_trial_with_target(tgt)
        return False

    def _in_stim_control_phase(self) -> bool:
        """С протоколом v2 все P300 (включая калибровку) — только по stim_control.json."""
        return self.stim_control_dir is not None

    def _show_waiting_overlay(self, *, line1: str, line2: str = "") -> None:
        self._overlay_force_show = True
        self._overlay_title = str(line1)
        self._overlay_sub = str(line2)
        self._overlay_target_id = None

    def _clear_waiting_overlay(self) -> None:
        self._overlay_force_show = False

    def _apply_stim_control_meta(self, cmd: dict | None) -> None:
        if not cmd:
            return
        self._overlay_exp_index = int(cmd.get("experiment_index") or 0)
        self._overlay_exp_total = int(cmd.get("experiment_total") or 0)
        label = str(cmd.get("label") or "")
        self._overlay_exp_calib = "калибр" in label.lower()

    def _subject_experiment_title(self) -> str:
        # В оригинальной зипке тут был импорт:
        # from experiment_protocol.subject_display import format_subject_experiment_title
        # В eyegaze такого модуля нет, поэтому оставляем простой заголовок.
        if self._overlay_exp_total > 0:
            idx = self._overlay_exp_index or max(1, int(self._auto_trials_started))
            return f"Эксперимент {idx}/{self._overlay_exp_total}"
        return "Подготовьтесь"

    def _draw_auto_overlay(self) -> None:
        """Серый fullscreen между trial (3a0a997) или ожидание протокола."""
        if not self.auto_random_trials:
            return
        title = ""
        sub = ""
        target_txt: str | None = None
        if self._overlay_force_show:
            title = self._overlay_title
            sub = self._overlay_sub
            if self._overlay_target_id is not None:
                target_txt = str(int(self._overlay_target_id))
        elif self._auto_pause_until is not None and self._auto_next_target is not None:
            title = self._subject_experiment_title()
            sub = "Смотрите на указанную плитку"
            target_txt = str(int(self._auto_next_target))
        else:
            return
        self._auto_overlay_title.text = title
        self._auto_overlay_sub.text = sub
        self._auto_overlay_target_num.text = target_txt or ""
        self._auto_overlay_bg.draw()
        self._auto_overlay_title.draw()
        self._auto_overlay_sub.draw()
        if target_txt:
            self._auto_overlay_target_num.draw()

    def _poll_stim_control(self) -> bool:
        """True — кадр отрисован, основной цикл должен continue."""
        if not self._in_stim_control_phase():
            return False
        from experiment_protocol import stim_control as sc

        cmd = sc.read_control(self.stim_control_dir)  # type: ignore[arg-type]
        self._apply_stim_control_meta(cmd)
        if cmd is None:
            self._show_waiting_overlay(
                line1="Ожидание",
                line2="",
            )
            self._draw()
            self._draw_auto_overlay()
            self.win.flip()
            return True
        state = str(cmd.get("state") or "paused")
        if state == "done":
            self._clear_waiting_overlay()
            return False
        if state == "paused":
            self._stim_control_wait_trial = False
            title = self._subject_experiment_title()
            self._show_waiting_overlay(
                line1=title,
                line2=str(cmd.get("message") or "Подготовьтесь"),
            )
            self._draw()
            self._draw_auto_overlay()
            self.win.flip()
            return True
        if state == "trial":
            tid = int(cmd.get("target_tile_id") or 0)
            cmd_ms = int(cmd.get("unix_ms") or 0)
            if not self._stim_control_wait_trial and cmd_ms != int(self._stim_control_last_ms):
                self._stim_control_last_ms = cmd_ms
                self._clear_waiting_overlay()
                self._auto_trials_started += 1
                self._schedule_auto_trial(target_tile_id=tid)
                self._stim_control_wait_trial = True
            if self._auto_pause_active():
                self._draw()
                self._draw_auto_overlay()
                self.win.flip()
                return True
            # Пауза закончилась — идёт прогон с плитками (не перекрывать ожиданием).
            return False
        self._show_waiting_overlay(line1="Ожидание", line2="")
        self._draw()
        self._draw_auto_overlay()
        self.win.flip()
        return True

    def _rand_target_avoid(self, *, prev: int | None) -> int:
        n = len(self.grid.tiles)
        if n <= 1:
            return 0
        tries = 0
        while True:
            x = random.randint(0, n - 1)
            if prev is None or x != int(prev):
                return int(x)
            tries += 1
            if tries > 50:
                # fallback: just pick next different
                return int((int(prev) + 1) % n)

    def _build_auto_target_plan(self) -> list[int]:
        """Plan for first trials: target repeats spaced out, never adjacent.

        Produces a list of length <= auto_plan_trials.
        """
        n_tiles = len(self.grid.tiles)
        if n_tiles <= 0 or self.auto_plan_trials <= 0:
            return []
        trials = int(self.auto_plan_trials)
        tgt = _clamp_int(self.auto_plan_target_tile_id, 0, n_tiles - 1, default=4)
        # Протокол v2: калибровка — все прогоны на одной плитке (3×12 эпох → эталон).
        if self.stim_control_dir is not None:
            return [int(tgt)] * trials
        # How many repeats do we need for a reliable template?
        # Each trial contributes approx. `sequences` target epochs (target flashes once per sequence).
        seq = _parse_int(
            self.tb_seq.text,
            config.DEFAULT_SEQUENCES,
            config.SEQUENCES_MIN,
            config.SEQUENCES_MAX,
        )
        max_non_adjacent = (trials + 1) // 2
        if self.auto_plan_target_repeats > 0:
            reps_needed = int(self.auto_plan_target_repeats)
        else:
            # auto: ensure target_epochs collected within non-adjacent constraint
            target_epochs = int(self.auto_plan_target_epochs)
            if target_epochs <= 0:
                reps_needed = 0
            else:
                # if sequences too small, raise it so we can satisfy both epochs and non-adjacent constraint
                # need: reps <= max_non_adjacent and reps*seq >= target_epochs  => seq >= ceil(target_epochs/max_non_adjacent)
                if max_non_adjacent > 0:
                    min_seq = int(math.ceil(target_epochs / float(max_non_adjacent)))
                else:
                    min_seq = target_epochs
                if seq < min_seq:
                    seq = _clamp_int(min_seq, config.SEQUENCES_MIN, config.SEQUENCES_MAX, default=config.DEFAULT_SEQUENCES)
                    self.tb_seq.text = str(seq)
                reps_needed = int(math.ceil(target_epochs / float(max(1, seq))))
        reps = max(0, min(int(reps_needed), trials, max_non_adjacent))
        if reps <= 0:
            # no special plan
            plan: list[int] = []
            prev: int | None = None
            for _ in range(trials):
                x = self._rand_target_avoid(prev=prev)
                plan.append(x)
                prev = x
            return plan

        # Choose positions for target: spread roughly evenly, avoid adjacency.
        positions: set[int] = set()
        if reps == 1:
            positions.add(trials // 2)
        else:
            step = (trials - 1) / float(reps - 1)
            for i in range(reps):
                positions.add(int(round(i * step)))
        # Fix adjacency if any
        pos_sorted = sorted(positions)
        fixed: list[int] = []
        for p in pos_sorted:
            if fixed and p == fixed[-1] + 1:
                # try shift right, else left
                pr = p + 1
                pl = p - 1
                if pr < trials and pr not in positions and (not fixed or pr != fixed[-1] + 1):
                    p = pr
                elif pl >= 0 and pl not in positions and (not fixed or p != fixed[-1] + 1):
                    p = pl
            fixed.append(p)
        positions = set(fixed)

        # Build plan
        plan: list[int] = []
        prev: int | None = None
        for i in range(trials):
            if i in positions:
                x = tgt
                if prev is not None and int(prev) == int(x):
                    # shouldn't happen; pick non-target
                    x = self._rand_target_avoid(prev=prev)
                plan.append(int(x))
                prev = int(x)
            else:
                # pick random, avoid repeating prev and avoid target if prev was target to prevent adjacency
                avoid_prev = prev
                tries = 0
                while True:
                    x = self._rand_target_avoid(prev=avoid_prev)
                    if prev is not None and int(prev) == int(tgt) and int(x) == int(tgt):
                        tries += 1
                        if tries > 50:
                            x = int((int(tgt) + 1) % n_tiles)
                        else:
                            continue
                    plan.append(int(x))
                    prev = int(x)
                    break
        return plan

    def _draw(self) -> None:
        for (tile, rect, tile_text) in zip(
                self.grid.tiles, self._tiles_visual, self._tile_texts
        ):

            if tile.active and not self.show_controls:
                # Любая мигающая плитка становится чёрной
                rect.fillColor = self.controller.get_stim_color()
                tile_text.color = "white"
            else:
                # Все остальные плитки белые
                rect.fillColor = config.TILE_DEFAULT_COLOR
                tile_text.color = "black"

            rect.draw()
            tile_text.draw()

        if self.show_controls:
            self.start_button.draw()
            self.stop_button.draw()
            self.start_text.draw()
            self.stop_text.draw()
            for lab in self._labels:
                lab.draw()
            for tb in self._tbs:
                tb.draw()
            self.hint.draw()
        else:
            self.fixation_cross.draw()

    def _handle_buttons(self) -> None:
        if not self.show_controls:
            return
        if self.mouse.getPressed()[0]:
            if self.mouse.isPressedIn(self.start_button):
                sequences, target_tile_id = self._read_settings()
                if sequences <= 0:
                    return
                self._start_trial_with_target(target_tile_id)
            if self.mouse.isPressedIn(self.stop_button):
                self.controller.stop()
                self.show_controls = True
                self.win.mouseVisible = True
                core.wait(0.2)


    def _screen_px_per_cm_for_spot(self) -> float:
        cfg = self.eyegaze_config or {}
        gaze_cfg = cfg.get("gaze_tiles", {}) if isinstance(cfg, dict) else {}
        screen_width_cm = float(gaze_cfg.get("screen_width_cm", getattr(config, "SCREEN_WIDTH_CM", 60.0)))
        screen_height_cm = float(gaze_cfg.get("screen_height_cm", getattr(config, "SCREEN_HEIGHT_CM", 34.0)))
        w_px = float(self.eyegaze_screen_w or self.win.size[0])
        h_px = float(self.eyegaze_screen_h or self.win.size[1])
        return ((w_px / max(1e-6, screen_width_cm)) + (h_px / max(1e-6, screen_height_cm))) / 2.0

    def _compute_angular_spot_radius_px(self, distance_cm: float) -> float:
        angle_deg = float(getattr(config, "YELLOW_SPOT_ANGLE_DEG", 8.0))
        d = max(1.0, float(distance_cm))
        radius_cm = math.tan(math.radians(angle_deg)) * d
        radius_px = radius_cm * self._screen_px_per_cm_for_spot()
        min_px = float(getattr(config, "YELLOW_SPOT_MIN_RADIUS_PX", 8.0))
        max_px = float(getattr(config, "YELLOW_SPOT_MAX_RADIUS_PX", 260.0))
        return max(min_px, min(max_px, radius_px))

    def _update_angular_spot_size_by_distance(self, distance_cm) -> None:
        try:
            if distance_cm is None:
                return
            d = float(distance_cm)
        except Exception:
            return
        if d <= 1.0 or self._gaze_dot is None:
            return
        if not bool(getattr(config, "YELLOW_SPOT_DISTANCE_SIZE_ENABLED", True)):
            return
        threshold_cm = float(getattr(config, "YELLOW_SPOT_UPDATE_THRESHOLD_CM", 2.5))
        if self._angular_spot_last_applied_cm is not None:
            if abs(d - float(self._angular_spot_last_applied_cm)) < threshold_cm:
                return
        new_radius = self._compute_angular_spot_radius_px(d)
        if self._angular_spot_current_radius_px is not None:
            if abs(new_radius - float(self._angular_spot_current_radius_px)) < 1.0:
                return
        self._angular_spot_current_radius_px = new_radius
        self._angular_spot_last_applied_cm = d
        try:
            self._gaze_dot.radius = new_radius
            self._gaze_dot.opacity = float(getattr(config, "YELLOW_SPOT_OPACITY", 0.45))
        except Exception:
            try:
                self._gaze_dot.size = (new_radius * 2.0, new_radius * 2.0)
            except Exception:
                pass

    def _angular_spot_status_text(self, distance_cm) -> str:
        if not bool(getattr(config, "YELLOW_SPOT_DISTANCE_SIZE_ENABLED", True)):
            return ""
        try:
            d = float(distance_cm)
        except Exception:
            return ""
        angle_deg = float(getattr(config, "YELLOW_SPOT_ANGLE_DEG", 8.0))
        r_px = self._angular_spot_current_radius_px
        if r_px is None:
            return f"Пятно: {angle_deg:.1f}°"
        return f"Пятно: {angle_deg:.1f}° | дистанция {d:.1f} см | радиус {r_px:.0f}px"

    def _setup_existing_gaze_logger_if_needed(self) -> None:
        if self.gaze_estimator is None or self.gaze_cap is None or self._gaze_writer is not None:
            return

        import csv
        import time
        from pathlib import Path

        log_dir = Path("data/logs/gaze_tiles_existing_calibration")
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{self.participant_id}_gaze_tiles_existing_calibration_{ts}.csv"

        self._gaze_log = log_path.open("w", newline="", encoding="utf-8")
        self._gaze_writer = csv.DictWriter(self._gaze_log, fieldnames = [
            "timestamp_unix",
            "target_tile_id",
            "predicted_tile_id",
            "p300_tile_id",
            "hybrid_final_tile_id",
            "hybrid_decision_source",
            "tile_hit",
        ])
        self._gaze_writer.writeheader()
        self._gaze_log.flush()

        self._gaze_text = visual.TextStim(
            self.win,
            text="",
            pos=getattr(config, "GAZE_INFO_POS", (-640, 315)),
            color="white",
            height=float(getattr(config, "GAZE_INFO_HEIGHT", 32)),
            wrapWidth=float(getattr(config, "GAZE_INFO_WRAP_WIDTH", 1100)),
            alignText="left",
            anchorHoriz="left",
        )
        initial_dot_radius = self._compute_angular_spot_radius_px(
            distance_cm=float(getattr(config, "YELLOW_SPOT_REFERENCE_DISTANCE_CM", 80.0))
        )
        self._angular_spot_base_radius_px = initial_dot_radius
        self._angular_spot_current_radius_px = initial_dot_radius
        self._angular_spot_reference_cm = float(getattr(config, "YELLOW_SPOT_REFERENCE_DISTANCE_CM", 80.0))
        self._angular_spot_last_applied_cm = self._angular_spot_reference_cm

        self._gaze_dot = visual.Circle(
            self.win,
            radius=initial_dot_radius,
            fillColor=getattr(config, "YELLOW_SPOT_FILL_COLOR", "red"),
            lineColor=getattr(config, "YELLOW_SPOT_LINE_COLOR", "white"),
            lineWidth=2,
            opacity=float(getattr(config, "YELLOW_SPOT_OPACITY", 0.45)),
            pos=(0, 0),
        )

    def _tile_screen_xy_px(self, tile) -> tuple[float, float]:
        w = float(self.eyegaze_screen_w or self.win.size[0])
        h = float(self.eyegaze_screen_h or self.win.size[1])
        return float(getattr(tile, "x_norm", 0.5)) * w, float(getattr(tile, "y_norm", 0.5)) * h

    def _gaze_xy_px_to_psychopy(self, x_px: float, y_px: float) -> tuple[float, float]:
        w = float(self.eyegaze_screen_w or self.win.size[0])
        h = float(self.eyegaze_screen_h or self.win.size[1])
        return (float(x_px) / max(1.0, w) - 0.5) * float(self.win.size[0]), (0.5 - float(y_px) / max(1.0, h)) * float(self.win.size[1])

    def _screen_xy_to_angles(self, x_px: float, y_px: float) -> tuple[float, float]:
        cfg = self.eyegaze_config or {}
        gaze_cfg = cfg.get("gaze_tiles", {}) if isinstance(cfg, dict) else {}
        screen_cfg = cfg.get("screen", {}) if isinstance(cfg, dict) else {}

        screen_width_cm = float(gaze_cfg.get("screen_width_cm", 60.0))
        screen_height_cm = float(gaze_cfg.get("screen_height_cm", 34.0))
        distance_cm = float(gaze_cfg.get("screen_distance_cm", screen_cfg.get("baseline_distance_cm", 66.0)))

        w = float(self.eyegaze_screen_w or self.win.size[0])
        h = float(self.eyegaze_screen_h or self.win.size[1])

        dx_cm = (float(x_px) / max(1.0, w) - 0.5) * screen_width_cm
        dy_cm = (0.5 - float(y_px) / max(1.0, h)) * screen_height_cm

        import math
        yaw = math.degrees(math.atan2(dx_cm, max(1e-6, distance_cm)))
        pitch = math.degrees(math.atan2(dy_cm, max(1e-6, distance_cm)))
        return float(yaw), float(pitch)

    def _nearest_tile_by_gaze_xy(self, gaze_xy) :
        if gaze_xy is None:
            return None
        gx, gy = float(gaze_xy[0]), float(gaze_xy[1])
        best = None
        best_d = 10**18
        for tile in self.grid.tiles:
            tx, ty = self._tile_screen_xy_px(tile)
            d = (tx - gx) ** 2 + (ty - gy) ** 2
            if d < best_d:
                best_d = d
                best = tile
        return best

    def _current_target_tile(self):
        target_id = self.controller.get_target_id()
        if target_id is None:
            return None
        for tile in self.grid.tiles:
            if int(tile.id) == int(target_id):
                return tile
        return None

    def _format_gaze_direction_ru(self, yaw, pitch, predicted_tile=None) -> str:
        try:
            yaw = float(yaw)
            pitch = float(pitch)
        except Exception:
            return "Взгляд: нет данных"

        parts = []
        if abs(yaw) < 1.0:
            parts.append("по центру")
        elif yaw > 0:
            parts.append(f"вправо {abs(yaw):.1f}°")
        else:
            parts.append(f"влево {abs(yaw):.1f}°")

        if abs(pitch) >= 1.0:
            if pitch > 0:
                parts.append(f"вверх {abs(pitch):.1f}°")
            else:
                parts.append(f"вниз {abs(pitch):.1f}°")

        tile_text = "" if predicted_tile is None else f" | плитка {predicted_tile.id}"
        return "Взгляд: " + "; ".join(parts) + tile_text

    def _read_p300_bridge_decision(self) -> dict:
        try:
            if not bool(getattr(config, "P300_BRIDGE_ENABLED", True)):
                return {}
            import json
            import time
            p = Path(str(getattr(config, "P300_DECISION_PATH", "data/p300_bridge/latest_decision.json")))
            if not p.exists():
                return {}
            data = json.loads(p.read_text(encoding="utf-8"))
            max_age_s = float(getattr(config, "P300_DECISION_MAX_AGE_S", 5.0))
            ts = data.get("timestamp_unix")
            if ts is not None and (time.time() - float(ts)) > max_age_s:
                return {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _resolve_hybrid_tile_decision(self, target_tile, predicted_tile) -> dict:
        target_id = None if target_tile is None else int(target_tile.id)
        gaze_id = None if predicted_tile is None else int(predicted_tile.id)

        p300 = self._read_p300_bridge_decision()
        p300_id = p300.get("winner_digit")
        try:
            p300_id = None if p300_id is None else int(p300_id)
        except Exception:
            p300_id = None

        if target_id is not None and gaze_id == target_id:
            return {
                "final_tile_id": gaze_id,
                "decision_source": "eye_tracking",
                "reason": "gaze_matches_target",
                "p300_tile_id": p300_id,
                "p300_available": bool(p300),
            }

        if p300_id is not None:
            return {
                "final_tile_id": p300_id,
                "decision_source": "p300",
                "reason": "gaze_mismatch",
                "p300_tile_id": p300_id,
                "p300_available": True,
            }

        return {
            "final_tile_id": gaze_id,
            "decision_source": "eye_tracking_fallback",
            "reason": "gaze_mismatch_but_no_p300",
            "p300_tile_id": None,
            "p300_available": False,
        }

    def _update_and_draw_existing_gaze(self) -> None:
        if self.gaze_estimator is None or self.gaze_cap is None:
            return

        self._setup_existing_gaze_logger_if_needed()

        import time

        ok, frame = self.gaze_cap.read()
        if not ok or frame is None:
            return

        result = self.gaze_estimator.process_frame(frame)
        self._gaze_last_result = result

        # Берём gaze_xy из уже откалиброванного GazeEstimator,
        # но дополнительно сглаживаем точку и не сбрасываем её мгновенно при моргании.
        raw_gaze_xy = result.gaze_xy

        gaze_cfg = {}
        try:
            gaze_cfg = (self.eyegaze_config or {}).get("gaze_tiles", {})
        except Exception:
            gaze_cfg = {}

        smooth_alpha = float(gaze_cfg.get("existing_gaze_smoothing_alpha", 0.35))
        hold_frames = int(gaze_cfg.get("existing_gaze_hold_frames", 8))
        ignore_blink = bool(gaze_cfg.get("existing_gaze_ignore_blink", True))

        gaze_xy = raw_gaze_xy
        if ignore_blink and bool(getattr(result, "blink", False)):
            gaze_xy = None

        if gaze_xy is not None:
            gx, gy = float(gaze_xy[0]), float(gaze_xy[1])
            if self._gaze_smooth_xy is None:
                self._gaze_smooth_xy = (gx, gy)
            else:
                sx_old, sy_old = self._gaze_smooth_xy
                a = max(0.0, min(1.0, smooth_alpha))
                self._gaze_smooth_xy = (
                    a * gx + (1.0 - a) * sx_old,
                    a * gy + (1.0 - a) * sy_old,
                )
            self._gaze_last_good_xy = self._gaze_smooth_xy
            self._gaze_lost_frames = 0
            gaze_xy = self._gaze_smooth_xy
            # Растягиваем координаты взгляда от центра экрана.
            stretch_x = float(gaze_cfg.get("gaze_stretch_x", 1.0))
            stretch_y = float(gaze_cfg.get("gaze_stretch_y", 1.0))

            w = float(self.eyegaze_screen_w or self.win.size[0])
            h = float(self.eyegaze_screen_h or self.win.size[1])

            cx = w / 2.0
            cy = h / 2.0

            gx, gy = gaze_xy

            gx = cx + (gx - cx) * stretch_x
            gy = cy + (gy - cy) * stretch_y

            gx = max(0.0, min(w - 1.0, gx))
            gy = max(0.0, min(h - 1.0, gy))

            gaze_xy = (gx, gy)
        else:
            self._gaze_lost_frames += 1
            if self._gaze_last_good_xy is not None and self._gaze_lost_frames <= hold_frames:
                gaze_xy = self._gaze_last_good_xy

        predicted_tile = self._nearest_tile_by_gaze_xy(gaze_xy)
        self._gaze_last_predicted_tile = predicted_tile

        target_tile = self._current_target_tile()

        target_yaw = target_pitch = None
        gaze_yaw = gaze_pitch = None
        target_x = target_y = None

        if target_tile is not None:
            target_x, target_y = self._tile_screen_xy_px(target_tile)
            target_yaw, target_pitch = self._screen_xy_to_angles(target_x, target_y)

        if gaze_xy is not None:
            gaze_yaw, gaze_pitch = self._screen_xy_to_angles(float(gaze_xy[0]), float(gaze_xy[1]))

        yaw_error = ""
        pitch_error = ""
        tile_hit = ""
        if target_yaw is not None and gaze_yaw is not None:
            yaw_error = abs(float(target_yaw) - float(gaze_yaw))
        if target_pitch is not None and gaze_pitch is not None:
            pitch_error = abs(float(target_pitch) - float(gaze_pitch))
        if target_tile is not None and predicted_tile is not None:
            tile_hit = int(int(target_tile.id) == int(predicted_tile.id))

        meta = result.meta or {}
        head_distance = meta.get("distance_cm")
        try:
            self._update_angular_spot_size_by_distance(head_distance)
        except Exception:
            pass

        hybrid = self._resolve_hybrid_tile_decision(target_tile, predicted_tile)

        if self._gaze_writer is not None:
            self._gaze_writer.writerow({
                "timestamp_unix": f"{time.time():.6f}",

                "target_tile_id": "" if target_tile is None else target_tile.id,

                "predicted_tile_id": "" if predicted_tile is None else predicted_tile.id,

                "p300_tile_id": "" if hybrid.get("p300_tile_id") is None
                else int(hybrid.get("p300_tile_id")),

                "hybrid_final_tile_id": "" if hybrid.get("final_tile_id") is None
                else int(hybrid.get("final_tile_id")),

                "hybrid_decision_source": str(
                    hybrid.get("decision_source", "")
                ),

                "tile_hit": tile_hit,
            })
            self._gaze_log.flush()

        # white frame around predicted tile
        if predicted_tile is not None:
            try:
                idx = int(predicted_tile.id)
                if 0 <= idx < len(self._tiles_visual):
                    rect = self._tiles_visual[idx]
                    old_line = rect.lineColor
                    old_width = rect.lineWidth
                    rect.lineColor = "white"
                    rect.lineWidth = 6
                    rect.draw()
                    rect.lineColor = old_line
                    rect.lineWidth = old_width
            except Exception:
                pass

        # red gaze point
        if gaze_xy is not None and self._gaze_dot is not None:
            gx, gy = self._gaze_xy_px_to_psychopy(float(gaze_xy[0]), float(gaze_xy[1]))
            self._gaze_dot.pos = (gx, gy)
            self._gaze_dot.draw()

        if self._gaze_text is not None:
            if gaze_xy is None:
                txt = "Взгляд: нет данных"
            else:
                txt = self._format_gaze_direction_ru(gaze_yaw, gaze_pitch, predicted_tile)
            spot_txt = self._angular_spot_status_text(head_distance)
            if spot_txt:
                txt += "\n" + spot_txt
            try:
                txt += f"\nГибрид: {hybrid.get('decision_source', '')} | итог={hybrid.get('final_tile_id')} | P300={hybrid.get('p300_tile_id')}"
            except Exception:
                pass
            self._gaze_text.text = txt
            self._gaze_text.draw()

    def _close_existing_gaze_logger(self) -> None:
        try:
            if self._gaze_log is not None:
                self._gaze_log.flush()
                self._gaze_log.close()
        except Exception:
            pass

    def run(self) -> None:
        while True:
            if self._poll_stim_control():
                keys = event.getKeys()
                if "escape" in keys:
                    break
                continue
            # Без stim_control — лимит trial; с протоколом — только state=done в stim_control.json.
            if (
                self.auto_random_trials
                and not self._in_stim_control_phase()
                and self._auto_max_trials is not None
                and self._auto_trials_started >= int(self._auto_max_trials)
            ):
                break
            if self._in_stim_control_phase():
                cmd = None
                try:
                    from experiment_protocol import stim_control as sc

                    cmd = sc.read_control(self.stim_control_dir)  # type: ignore[arg-type]
                except Exception:
                    pass
                if cmd and str(cmd.get("state")) == "done":
                    break
            if self._auto_pending_first_trial:
                self._auto_pending_first_trial = False
                # Show instruction before the very first auto trial as well
                if self._auto_trials_started < len(self._auto_target_plan):
                    tgt0 = int(self._auto_target_plan[self._auto_trials_started])
                else:
                    tgt0 = self._rand_target_avoid(prev=None)
                self._auto_trials_started += 1
                self._schedule_auto_trial(target_tile_id=int(tgt0))
            # If we're between trials in auto mode, just render overlay and wait
            if self._auto_pause_active():
                self._draw()
                self._draw_auto_overlay()
                self.win.flip()
                keys = event.getKeys()
                if "escape" in keys:
                    break
                continue
            self._handle_buttons()
            event_data = self.controller.update()
            if event_data:
                if "tile_id" in event_data:
                    self.win.callOnFlip(
                        self.controller.lsl.send, event_data["tile_id"], event_data["event"]
                    )
                    print(event_data)
                elif event_data.get("event") == "trial_start":
                    target = event_data.get("target")
                    self.win.callOnFlip(
                        self.controller.lsl.send, -1, f"trial_start|target={target}"
                    )
                    print(f"TRIAL START: target={target}")
                elif event_data.get("event") == "trial_end":
                    self.win.callOnFlip(self.controller.lsl.send, -2, "trial_end")
                    print("TRIAL END")
                    if self.auto_random_trials:
                        self.controller.stop()
                        if self._in_stim_control_phase():
                            self._stim_control_wait_trial = False
                        elif self._auto_trials_started < len(self._auto_target_plan):
                            nxt = int(self._auto_target_plan[self._auto_trials_started])
                            self._auto_trials_started += 1
                            self._schedule_auto_trial(target_tile_id=int(nxt))
                        elif not self.stim_control_dir:
                            prev = int(self._auto_target_plan[-1]) if self._auto_target_plan else None
                            nxt = int(self._rand_target_avoid(prev=prev))
                            self._auto_trials_started += 1
                            self._schedule_auto_trial(target_tile_id=int(nxt))
                    else:
                        self.controller.stop()
                        self.show_controls = True
                        self.win.mouseVisible = True
            self._draw()
            if self.gaze_estimator is not None and self.gaze_cap is not None:
                self._update_and_draw_existing_gaze()
            # Серый оверлей только между trial, не во время мигания плиток.
            if self._auto_pause_active() or self._overlay_force_show:
                self._draw_auto_overlay()
            self.win.flip()
            keys = event.getKeys()
            if "escape" in keys:
                break
            if "space" in keys and not self.show_controls:
                self.controller.stop()
                self.show_controls = True
                self.win.mouseVisible = True
        self.controller.stop()
        self._close_existing_gaze_logger()
        self.win.close()
        core.quit()
