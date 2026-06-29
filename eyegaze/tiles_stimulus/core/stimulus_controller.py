from __future__ import annotations

import random
from psychopy import core


class _LSLMarkerSender:
    """
    Отправка LSL-маркеров для P300 Analyzer.

    Поток:
        name="BCI_StimMarkers"
        type="Markers"

    Для фотодатчика и continuous CSV номера плиток кодируются как 100..108,
    а P300 marker_parsing.py декодирует их обратно в 0..8.

    Примеры:
        "100|on"  -> плитка 0 включилась
        "100|off" -> плитка 0 выключилась
        "-1|trial_start|target=5"
        "-2|trial_end"
        "-3|trial_config|..."
    """

    def __init__(self):
        self.outlet = None
        try:
            from pylsl import StreamInfo, StreamOutlet

            info = StreamInfo(
                name="BCI_StimMarkers",
                type="Markers",
                channel_count=1,
                nominal_srate=0,
                channel_format="string",
                source_id="stimulus-controller-001",
            )
            self.outlet = StreamOutlet(info)
            print("[LSL] Marker stream created: BCI_StimMarkers")
        except Exception as e:
            self.outlet = None
            print(f"[LSL] Marker stream disabled: {e}")

    def send(self, tile_id, event):
        if self.outlet is None:
            return

        try:
            tile_id = int(tile_id)
        except Exception:
            tile_id = -999

        event = str(event)
        if event == "flash_on":
            event = "on"
        elif event == "flash_off":
            event = "off"

        marker_id = (100 + tile_id) if tile_id >= 0 else tile_id
        marker = f"{marker_id}|{event}"
        try:
            self.outlet.push_sample([marker])
            print("[LSL MARKER]", marker)
        except Exception as e:
            print(f"[LSL] marker send error: {e}")


class StimulusController:
    def __init__(
        self,
        grid,
        *,
        flash_duration=0.10,
        isi=0.10,
        cue_duration=2.0,
        ready_duration=1.5,
        inter_block_s=0.8,
        cue_color="white",
        stim_color="white",
    ):
        self.grid = grid
        self.flash_duration = float(flash_duration)
        self.isi = float(isi)
        self.cue_duration = float(cue_duration)
        self.ready_duration = float(ready_duration)
        self.inter_block_s = float(inter_block_s)
        self.cue_color = cue_color
        self.stim_color = stim_color

        self.lsl = _LSLMarkerSender()

        self.running = False
        self.target_tile_id = None
        self.sequences = 0
        self._events = []
        self._start_time = 0.0
        self._idx = 0

    def start_experiment(self, sequences: int, target_tile_id: int = 0):
        self.stop()
        self.running = True
        self.target_tile_id = int(target_tile_id)
        self.sequences = int(sequences)
        self._start_time = core.getTime()
        self._idx = 0

        self._events = [{"t": 0.0, "event": "trial_start", "target": self.target_tile_id}]

        t = self.ready_duration
        ids = [int(tile.id) for tile in self.grid.tiles]
        for _ in range(max(1, self.sequences)):
            order = ids[:]
            random.shuffle(order)
            for tid in order:
                self._events.append({"t": t, "event": "flash_on", "tile_id": tid})
                t += self.flash_duration
                self._events.append({"t": t, "event": "flash_off", "tile_id": tid})
                t += self.isi
            t += self.inter_block_s

        self._events.append({"t": t, "event": "trial_end"})

    def stop(self):
        for tile in self.grid.tiles:
            tile.active = False
        self.running = False
        self._events = []

    def get_target_id(self):
        return self.target_tile_id

    def get_target_color(self):
        return self.cue_color

    def get_stim_color(self):
        return self.stim_color

    def update(self):
        if not self.running:
            return None
        now = core.getTime() - self._start_time
        if self._idx >= len(self._events):
            return None

        ev = self._events[self._idx]
        if now < ev["t"]:
            return None

        self._idx += 1
        name = ev.get("event")

        if name == "trial_start":
            return ev

        if name == "flash_on":
            tid = int(ev["tile_id"])
            for tile in self.grid.tiles:
                tile.active = (int(tile.id) == tid)
            return {"event": "flash_on", "tile_id": tid}

        if name == "flash_off":
            for tile in self.grid.tiles:
                tile.active = False
            return {"event": "flash_off", "tile_id": int(ev["tile_id"])}

        if name == "trial_end":
            self.stop()
            return {"event": "trial_end"}

        return None
