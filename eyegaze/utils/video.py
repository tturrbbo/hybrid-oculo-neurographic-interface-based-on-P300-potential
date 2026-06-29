from __future__ import annotations

from contextlib import contextmanager
import time
import cv2


def list_cameras() -> list[str]:
    try:
        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()
        return list(graph.get_input_devices())
    except Exception:
        return []


def _set_if_present(cap: cv2.VideoCapture, prop: int, value):
    if value is None or value == "":
        return
    try:
        cap.set(prop, int(value))
    except Exception:
        pass


def _backend_name(cap: cv2.VideoCapture) -> str:
    try:
        return cap.getBackendName()
    except Exception:
        return "unknown"


def open_camera(
    index: int = 0,
    *,
    name: str | None = None,
    fallback_index: int | None = None,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
) -> cv2.VideoCapture:
    """
    Strict camera opening for Windows.

    LIST_CAMERAS.bat uses DirectShow order through pygrabber.
    So this function also opens cameras through cv2.CAP_DSHOW.

    No silent fallback is used.
    If index=1 is requested, only camera 1 is opened.
    """

    if index is None:
        if fallback_index is None:
            raise RuntimeError("camera index is None and no fallback_index was provided")
        index = fallback_index

    index = int(index)

    devices = list_cameras()
    if devices:
        print("[camera] DirectShow devices:")
        for i, dev in enumerate(devices):
            print(f"[camera]   {i}: {dev}")

    if name:
        print(f"[camera] Requested camera: index={index}, name={name}")
    else:
        print(f"[camera] Requested camera: index={index}")

    print(f"[camera] Opening STRICT DirectShow: cv2.VideoCapture({index}, cv2.CAP_DSHOW)")
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"cannot open requested camera index={index}. "
            f"No automatic fallback was used. Check LIST_CAMERAS.bat and config/experiment.yaml."
        )

    _set_if_present(cap, cv2.CAP_PROP_FRAME_WIDTH, width)
    _set_if_present(cap, cv2.CAP_PROP_FRAME_HEIGHT, height)
    _set_if_present(cap, cv2.CAP_PROP_FPS, fps)

    last_frame = None
    for _ in range(30):
        ok, frame = cap.read()
        if ok and frame is not None:
            last_frame = frame
        time.sleep(0.02)

    if last_frame is None:
        cap.release()
        raise RuntimeError(f"camera index={index} opened, but no frames received")

    h, w = last_frame.shape[:2]
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    backend = _backend_name(cap)

    print(
        f"[camera] Camera OK: index={index}, backend={backend}, "
        f"frame={w}x{h}, fps={real_fps:.1f}, mean={float(last_frame.mean()):.2f}"
    )

    return cap


@contextmanager
def fullscreen(name: str, screen_index: int = 1):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    # Координаты мониторов в режиме "Расширить".
    # 0 обычно ноут: x=0
    # 1 обычно внешний монитор: x=1920 или -1920
    if screen_index == 1:
        cv2.moveWindow(name, 1920, 0)
    else:
        cv2.moveWindow(name, 0, 0)

    cv2.waitKey(100)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        yield
    finally:
        cv2.destroyWindow(name)


@contextmanager
def camera(
    index: int = 0,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    name: str | None = None,
    fallback_index: int | None = None,
):
    cap = open_camera(
        index=index,
        name=name,
        fallback_index=fallback_index,
        width=width,
        height=height,
        fps=fps,
    )
    try:
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()


def iter_frames(cap: cv2.VideoCapture):
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        yield frame
