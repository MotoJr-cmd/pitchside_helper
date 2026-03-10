from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional

import cv2
import mss
import numpy as np
import torch
from ultralytics import YOLO


TARGET_CLASSES: List[int] = [0, 32]  # 0 = person, 32 = sports ball

DEFAULT_MONITOR: Dict[str, int] = {"top": 40, "left": 0, "width": 800, "height": 640}


def get_device() -> str:
    """
    Select the best available device for inference.

    On Apple Silicon, prefer Metal Performance Shaders (MPS) when available;
    otherwise, fall back to CPU.
    """
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return "mps"
    return "cpu"


def load_model(weights: str, device: Optional[str] = None) -> YOLO:
    """
    Load a YOLO11 model and move it to the requested device.

    Parameters
    ----------
    weights:
        Path or name of the YOLO weights file (e.g. 'yolo11n.pt').
    device:
        Device string understood by PyTorch ('mps', 'cpu', 'cuda', ...).
        If None, uses get_device().
    """
    model = YOLO(weights)
    dev = device or get_device()
    model.to(dev)
    print(f"[vision] Loaded model '{weights}' on device: {dev}")
    return model


def _update_fps(prev_time: Optional[float]) -> tuple[float, float]:
    """
    Compute instantaneous FPS given the previous timestamp.

    Returns (fps, now_timestamp).
    """
    now = time.perf_counter()
    if prev_time is None:
        return 0.0, now

    dt = now - prev_time
    if dt <= 0.0:
        return 0.0, now

    fps = 1.0 / dt
    return fps, now


def run_screen_capture(
    model: YOLO,
    monitor: Dict[str, int],
    imgsz: Optional[int] = None,
    classes: Optional[List[int]] = None,
    conf: float = 0.25,
    max_frames: Optional[int] = None,
) -> None:
    """
    Run a realtime screen-region inference loop using the given YOLO model.

    Parameters
    ----------
    model:
        A YOLO model already moved to the desired device.
    monitor:
        Dictionary describing the capture region, e.g.
        {'top': 40, 'left': 0, 'width': 800, 'height': 640}.
    imgsz:
        Optional input size hint for the model. If None, process frames at native
        capture resolution. For speed, pass e.g. 640.
    classes:
        List of class IDs to keep (e.g. [0, 32] for person + sports ball).
    conf:
        Confidence threshold for detections.
    max_frames:
        Optional safety limit mainly for automated testing; when set, the loop
        exits after processing this many frames.
    """
    print(
        f"[vision] Starting screen capture loop on monitor={monitor} "
        f"with imgsz={imgsz or 'native'}, classes={classes or 'all'}"
    )

    frame_count = 0
    prev_time: Optional[float] = None
    device = getattr(model, "device", None)
    if device is not None:
        try:
            device_str = str(device)
        except Exception:
            device_str = "unknown"
    else:
        device_str = "unknown"

    try:
        with mss.mss() as sct:
            while True:
                raw = sct.grab(monitor)

                frame = np.array(raw)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                fps, prev_time = _update_fps(prev_time)

                predict_kwargs = {
                    "source": frame,
                    "conf": conf,
                    "classes": classes,
                    "verbose": False,
                }
                if imgsz is not None:
                    predict_kwargs["imgsz"] = imgsz
                if device_str != "unknown":
                    predict_kwargs["device"] = device_str

                results = model.predict(**predict_kwargs)
                rendered = results[0].plot()

                hud_text = f"Pitchside Helper | FPS: {fps:.1f}"
                cv2.putText(
                    rendered,
                    hud_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Pitchside Screen Analytics", rendered)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[vision] 'q' pressed, exiting loop.")
                    break

                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    print(f"[vision] Reached max_frames={max_frames}, exiting loop.")
                    break
    finally:
        cv2.destroyAllWindows()
        print("[vision] Destroyed all OpenCV windows.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pitchside Helper - YOLO11 Nano screen-region analytics (HUD)."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11n.pt",
        help="Path to YOLO11 Nano weights file (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help=(
            "Optional inference image size. "
            "If omitted, process frames at native capture resolution. "
            "For speed, try --imgsz 640."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO detections (default: 0.25).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Optional maximum number of frames to process before exiting. "
            "Primarily useful for automated testing."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    model = load_model(args.weights, device=device)

    run_screen_capture(
        model=model,
        monitor=DEFAULT_MONITOR,
        imgsz=args.imgsz,
        classes=TARGET_CLASSES,
        conf=args.conf,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()

