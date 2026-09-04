#!/usr/bin/env python3
"""Live viewer for a locally connected Thorlabs scientific camera."""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets


pg.setConfigOptions(imageAxisOrder="row-major")


@dataclass(frozen=True)
class CameraInfo:
    camera_id: str
    model: str
    name: str
    serial_number: str
    firmware_version: str
    sensor_shape: tuple[int, int]
    image_shape: tuple[int, int]
    bit_depth: int
    sensor_type: str
    exposure_time_us: int
    exposure_range_us: tuple[int, int]
    gain: int
    gain_range: tuple[int, int]
    gain_db: float | None
    communication_interface: str
    usb_port_type: str
    sdk_version: str


@dataclass(frozen=True)
class CameraSettings:
    exposure_time_us: int
    gain: int
    gain_db: float | None


@dataclass(frozen=True)
class FramePacket:
    image: np.ndarray
    frame_count: int
    camera_timestamp_ns: int | None
    received_at: float
    acquisition_fps: float
    sdk_fps: float
    skipped_frames: int


@dataclass(frozen=True)
class ROIStatistics:
    bounds: tuple[int, int, int, int]
    pixel_count: int
    total: float
    mean: float
    minimum: float
    maximum: float


def clipped_roi_bounds(
    position: tuple[float, float],
    size: tuple[float, float],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return pixel bounds as ``(x0, x1, y0, y1)`` inside an image."""

    image_height, image_width = image_shape
    x0 = min(max(0, int(round(position[0]))), max(0, image_width - 1))
    y0 = min(max(0, int(round(position[1]))), max(0, image_height - 1))
    width = max(1, int(round(size[0])))
    height = max(1, int(round(size[1])))
    x1 = min(image_width, x0 + width)
    y1 = min(image_height, y0 + height)
    return x0, x1, y0, y1


def roi_statistics(
    image: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> ROIStatistics:
    """Calculate simple statistics from the raw camera counts in an ROI."""

    x0, x1, y0, y1 = bounds
    pixels = image[y0:y1, x0:x1]
    return ROIStatistics(
        bounds=bounds,
        pixel_count=int(pixels.size),
        total=float(np.sum(pixels, dtype=np.float64)),
        mean=float(np.mean(pixels)),
        minimum=float(np.min(pixels)),
        maximum=float(np.max(pixels)),
    )


def _simple_colormap(
    name: str,
    stops: int = 6,
    red_at_top: bool = False,
) -> pg.ColorMap:
    """Make a colour map with only a few editable histogram handles."""

    if name == "grey":
        colours = np.asarray([(0, 0, 0), (255, 255, 255)], dtype=np.uint8)
    else:
        base = pg.colormap.get(name)
        colours = base.getLookupTable(0.0, 1.0, stops)[:, :3]

    positions = np.linspace(0.0, 1.0, len(colours))
    if red_at_top:
        # Keep the normal map almost all the way to its upper limit. Pixels at
        # that limit, or clipped above it, then use the final bright-red stop.
        positions[-1] = 0.999
        positions = np.append(positions, 1.0)
        colours = np.vstack([colours, (255, 0, 0)])

    return pg.ColorMap(positions, colours)


def _enum_name(value: object) -> str:
    return str(getattr(value, "name", value))


def _gain_db(camera, gain: int) -> float | None:
    try:
        return float(camera.convert_gain_to_decibels(gain))
    except Exception:
        return None


class CameraWorker(QtCore.QThread):
    """Own the SDK and camera while continuously draining incoming frames."""

    camera_opened = QtCore.pyqtSignal(object)
    settings_changed = QtCore.pyqtSignal(object)
    message = QtCore.pyqtSignal(str)
    failed = QtCore.pyqtSignal(str, str)

    def __init__(self, camera_id: str | None, parent=None):
        super().__init__(parent)
        self.camera_id = camera_id
        self._stop_requested = threading.Event()
        self._commands: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self._frame_lock = threading.Lock()
        self._latest_frame: FramePacket | None = None

    def stop(self) -> None:
        self._stop_requested.set()

    def set_camera_settings(self, exposure_time_us: int, gain: int) -> None:
        self._commands.put(("settings", (exposure_time_us, gain)))

    def take_latest_frame(self) -> FramePacket | None:
        """Give the GUI the newest frame and release the worker's reference."""

        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
        return frame

    def _store_latest_frame(self, frame: FramePacket) -> None:
        with self._frame_lock:
            self._latest_frame = frame

    def run(self) -> None:
        try:
            self._run_camera()
        except Exception as error:
            details = traceback.format_exc()
            print(details, file=sys.stderr)
            self.failed.emit(str(error), details)

    def _run_camera(self) -> None:
        # Import here so importing dnamic-toolkit never requires the proprietary SDK.
        from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
        from thorlabs_tsi_sdk.tl_camera_enums import OPERATION_MODE
        from thorlabs_tsi_sdk.version import version_number

        self.message.emit("Opening Thorlabs camera SDK")
        with TLCameraSDK() as sdk:
            camera_ids = list(sdk.discover_available_cameras())
            self.message.emit(f"Discovered cameras: {camera_ids or 'none'}")
            if not camera_ids:
                raise RuntimeError("No Thorlabs scientific cameras were discovered")

            camera_id = self.camera_id or camera_ids[0]
            if camera_id not in camera_ids:
                raise RuntimeError(
                    f"Camera {camera_id!r} was not found; discovered {camera_ids}"
                )

            with sdk.open_camera(camera_id) as camera:
                info = self._read_camera_info(camera_id, camera, version_number)
                self.camera_opened.emit(info)

                camera.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
                camera.frames_per_trigger_zero_for_unlimited = 0
                camera.image_poll_timeout_ms = 50
                camera.arm(2)
                camera.issue_software_trigger()
                self.message.emit("Continuous software-triggered acquisition started")

                self._poll_camera(camera)

        self.message.emit("Camera and SDK closed")

    @staticmethod
    def _read_camera_info(camera_id: str, camera, sdk_version: str) -> CameraInfo:
        exposure_range = camera.exposure_time_range_us
        gain_range = camera.gain_range
        gain = int(camera.gain)
        return CameraInfo(
            camera_id=str(camera_id),
            model=str(camera.model),
            name=str(camera.name),
            serial_number=str(camera.serial_number),
            firmware_version=str(camera.firmware_version),
            sensor_shape=(
                int(camera.sensor_height_pixels),
                int(camera.sensor_width_pixels),
            ),
            image_shape=(
                int(camera.image_height_pixels),
                int(camera.image_width_pixels),
            ),
            bit_depth=int(camera.bit_depth),
            sensor_type=_enum_name(camera.camera_sensor_type),
            exposure_time_us=int(camera.exposure_time_us),
            exposure_range_us=(int(exposure_range.min), int(exposure_range.max)),
            gain=gain,
            gain_range=(int(gain_range.min), int(gain_range.max)),
            gain_db=_gain_db(camera, gain),
            communication_interface=_enum_name(camera.communication_interface),
            usb_port_type=_enum_name(camera.usb_port_type),
            sdk_version=str(sdk_version),
        )

    def _poll_camera(self, camera) -> None:
        frames_seen = 0
        rate_started = time.monotonic()
        acquisition_fps = 0.0
        sdk_fps = 0.0
        last_sdk_rate_read = 0.0
        last_frame_count: int | None = None
        skipped_frames = 0

        # The GUI does not need every camera frame. Limiting display updates keeps
        # Qt responsive while this loop continues to drain the camera buffer.
        last_display_frame = 0.0
        display_interval = 1.0 / 30.0

        while not self._stop_requested.is_set():
            self._apply_commands(camera)
            frame = camera.get_pending_frame_or_null()
            if frame is None:
                continue

            now = time.monotonic()
            frames_seen += 1
            frame_count = int(frame.frame_count)
            if last_frame_count is not None and frame_count > last_frame_count:
                skipped_frames += max(0, frame_count - last_frame_count - 1)
            last_frame_count = frame_count

            elapsed = now - rate_started
            if elapsed >= 1.0:
                acquisition_fps = frames_seen / elapsed
                frames_seen = 0
                rate_started = now

            if now - last_sdk_rate_read >= 1.0:
                try:
                    sdk_fps = float(camera.get_measured_frame_rate_fps())
                except Exception as error:
                    self.message.emit(f"Could not read SDK frame rate: {error}")
                last_sdk_rate_read = now

            if now - last_display_frame < display_interval:
                continue

            # SDK-owned image memory becomes invalid on the next poll.
            image = np.copy(frame.image_buffer)
            packet = FramePacket(
                image=image,
                frame_count=frame_count,
                camera_timestamp_ns=frame.time_stamp_relative_ns_or_null,
                received_at=now,
                acquisition_fps=acquisition_fps,
                sdk_fps=sdk_fps,
                skipped_frames=skipped_frames,
            )
            self._store_latest_frame(packet)
            last_display_frame = now

    def _apply_commands(self, camera) -> None:
        while True:
            try:
                command, value = self._commands.get_nowait()
            except queue.Empty:
                return

            if command != "settings":
                continue

            exposure_time_us, gain = value
            try:
                camera.exposure_time_us = int(exposure_time_us)
                if camera.gain_range.max > 0:
                    camera.gain = int(gain)
                actual_gain = int(camera.gain)
                settings = CameraSettings(
                    exposure_time_us=int(camera.exposure_time_us),
                    gain=actual_gain,
                    gain_db=_gain_db(camera, actual_gain),
                )
                self.settings_changed.emit(settings)
                self.message.emit(
                    "Applied camera settings: "
                    f"exposure={settings.exposure_time_us / 1000:g} ms, "
                    f"gain={settings.gain}"
                )
            except Exception as error:
                self.message.emit(f"Could not apply camera settings: {error}")


class CameraViewer(QtWidgets.QMainWindow):
    def __init__(self, camera_id: str | None = None):
        super().__init__()
        self.camera_id = camera_id
        self.worker: CameraWorker | None = None
        self.image: np.ndarray | None = None
        self.camera_info: CameraInfo | None = None
        self.camera_settings: CameraSettings | None = None
        self.roi: pg.RectROI | None = None
        self.markers: list[pg.TargetItem] = []
        self.profile_lines: list[pg.LineSegmentROI | None] = [None, None]
        self.trace_times: deque[float] = deque()
        self.trace_counts: deque[float] = deque()
        self.trace_started: float | None = None
        self.display_frames = 0
        self.display_rate_started = time.monotonic()
        self.display_fps = 0.0
        self.last_displayed_frame_count: int | None = None
        self.undisplayed_frames = 0
        self.last_diagnostics_update = 0.0
        self.last_trace_plot_update = 0.0
        self.connection_failed = False

        self.setWindowTitle("Thorlabs camera viewer")
        self.resize(1550, 900)
        self._build_window()
        self._set_connected_controls(False)
        self.disconnect_button.setEnabled(False)

        # The GUI asks for the newest image at its own pace. A single shared frame
        # prevents acquisition from filling Qt's event queue with large arrays.
        self.frame_timer = QtCore.QTimer(self)
        self.frame_timer.setInterval(33)
        self.frame_timer.timeout.connect(self._display_latest_frame)
        self.frame_timer.start()
        QtCore.QTimer.singleShot(0, self.connect_camera)

    def _build_window(self) -> None:
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.setCentralWidget(splitter)

        self.image_view = pg.ImageView()
        self.image_view.getView().setAspectLocked(True)
        self.image_view.getView().invertY(True)
        if getattr(self.image_view.ui, "menuBtn", None):
            self.image_view.ui.menuBtn.hide()
        if getattr(self.image_view.ui, "roiBtn", None):
            self.image_view.ui.roiBtn.hide()
        self.image_view.setColorMap(_simple_colormap("magma"))

        splitter.addWidget(self.image_view)

        panel = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(8, 8, 8, 8)
        panel_layout.addWidget(self._build_connection_group())
        panel_layout.addWidget(self._build_settings_group())
        panel_layout.addWidget(self._build_roi_group())
        panel_layout.addWidget(self._build_marker_group())
        panel_layout.addWidget(self._build_trace_group(), stretch=2)
        panel_layout.addWidget(self._build_details_tabs(), stretch=1)
        self.panel_scroll = QtWidgets.QScrollArea()
        self.panel_scroll.setWidgetResizable(True)
        self.panel_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.panel_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.panel_scroll.setMinimumWidth(440)
        self.panel_scroll.setMaximumWidth(640)
        self.panel_scroll.setWidget(panel)
        splitter.addWidget(self.panel_scroll)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)

        self.cursor_label = QtWidgets.QLabel("Move over the image for pixel values")
        self.statusBar().addPermanentWidget(self.cursor_label, 1)

        self.mouse_proxy = pg.SignalProxy(
            self.image_view.getView().scene().sigMouseMoved,
            rateLimit=60,
            slot=self._mouse_moved,
        )
        self.image_view.getView().scene().sigMouseClicked.connect(self._image_clicked)
        self.connect_button.setFocus()

    def _build_connection_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Camera")
        layout = QtWidgets.QVBoxLayout(group)

        status_row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Disconnected")
        self.status_label.setStyleSheet("font-weight: bold; color: #b03030")
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_camera)
        self.disconnect_button = QtWidgets.QPushButton("Disconnect")
        self.disconnect_button.clicked.connect(self.disconnect_camera)
        status_row.addWidget(self.status_label, stretch=1)
        status_row.addWidget(self.connect_button)
        status_row.addWidget(self.disconnect_button)
        layout.addLayout(status_row)

        serial_row = QtWidgets.QHBoxLayout()
        serial_row.addWidget(QtWidgets.QLabel("Camera ID"))
        self.camera_id_edit = QtWidgets.QLineEdit(self.camera_id or "")
        self.camera_id_edit.setPlaceholderText("blank uses first camera found")
        serial_row.addWidget(self.camera_id_edit, stretch=1)
        layout.addLayout(serial_row)

        return group

    def _build_settings_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Acquisition and display")
        grid = QtWidgets.QGridLayout(group)

        grid.addWidget(QtWidgets.QLabel("Exposure"), 0, 0)
        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setDecimals(3)
        self.exposure_spin.setSuffix(" ms")
        self.exposure_spin.setKeyboardTracking(False)
        self.exposure_spin.setStepType(
            QtWidgets.QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        )
        grid.addWidget(self.exposure_spin, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Gain index"), 1, 0)
        self.gain_spin = QtWidgets.QSpinBox()
        grid.addWidget(self.gain_spin, 1, 1)
        self.apply_settings_button = QtWidgets.QPushButton("Apply")
        self.apply_settings_button.clicked.connect(self.apply_camera_settings)
        grid.addWidget(self.apply_settings_button, 0, 2, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Colour map"), 2, 0)
        self.colour_map_combo = QtWidgets.QComboBox()
        self.colour_map_combo.addItems(
            ["magma", "grey", "viridis", "plasma", "inferno", "cividis"]
        )
        self.colour_map_combo.currentTextChanged.connect(self._set_colour_map)
        grid.addWidget(self.colour_map_combo, 2, 1, 1, 2)

        self.auto_levels_check = QtWidgets.QCheckBox("Autoscale every frame")
        self.auto_levels_check.setChecked(True)
        self.auto_levels_check.toggled.connect(self._autoscale_toggled)
        grid.addWidget(self.auto_levels_check, 3, 0, 1, 2)
        self.auto_levels_button = QtWidgets.QPushButton("Autoscale once")
        self.auto_levels_button.clicked.connect(self._auto_levels_once)
        grid.addWidget(self.auto_levels_button, 3, 2)

        self.red_at_top_check = QtWidgets.QCheckBox(
            "Show colour-scale maximum in red"
        )
        self.red_at_top_check.setToolTip(
            "Make pixels at or above the upper colour-bar handle bright red"
        )
        self.red_at_top_check.toggled.connect(self._red_at_top_toggled)
        grid.addWidget(self.red_at_top_check, 4, 0, 1, 3)
        return group

    def _build_roi_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Measurement ROI")
        layout = QtWidgets.QVBoxLayout(group)
        button_row = QtWidgets.QHBoxLayout()
        self.reset_roi_button = QtWidgets.QPushButton("Reset ROI")
        self.reset_roi_button.clicked.connect(self._reset_roi)
        self.pause_trace_check = QtWidgets.QCheckBox("Pause trace")
        self.clear_trace_button = QtWidgets.QPushButton("Clear trace")
        self.clear_trace_button.clicked.connect(self._clear_trace)
        button_row.addWidget(self.reset_roi_button)
        button_row.addWidget(self.pause_trace_check)
        button_row.addWidget(self.clear_trace_button)
        layout.addLayout(button_row)

        self.roi_info_text = QtWidgets.QPlainTextEdit()
        self.roi_info_text.setReadOnly(True)
        self.roi_info_text.setMaximumHeight(100)
        self.roi_info_text.setStyleSheet("font-family: monospace")
        layout.addWidget(self.roi_info_text)
        return group

    def _build_marker_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Markers and line profiles")
        layout = QtWidgets.QVBoxLayout(group)
        button_row = QtWidgets.QHBoxLayout()
        self.add_marker_button = QtWidgets.QPushButton("Place marker")
        self.add_marker_button.setCheckable(True)
        self.clear_markers_button = QtWidgets.QPushButton("Clear markers")
        self.clear_markers_button.clicked.connect(self._clear_markers)
        button_row.addWidget(self.add_marker_button)
        button_row.addWidget(self.clear_markers_button)
        layout.addLayout(button_row)

        self.marker_info_text = QtWidgets.QPlainTextEdit()
        self.marker_info_text.setReadOnly(True)
        self.marker_info_text.setMaximumHeight(48)
        self.marker_info_text.setPlaceholderText(
            "Press Place marker, then click on the image"
        )
        self.marker_info_text.setStyleSheet("font-family: monospace")
        layout.addWidget(self.marker_info_text)

        profile_row = QtWidgets.QHBoxLayout()
        profile_row.addWidget(QtWidgets.QLabel("Show"))
        self.profile_line_checks = [
            QtWidgets.QCheckBox("Line 1"),
            QtWidgets.QCheckBox("Line 2"),
        ]
        self.profile_line_checks[0].setChecked(True)
        for index, checkbox in enumerate(self.profile_line_checks):
            checkbox.toggled.connect(
                lambda checked, line_index=index: self._toggle_profile_line(
                    line_index, checked
                )
            )
            profile_row.addWidget(checkbox)
        self.reset_profile_lines_button = QtWidgets.QPushButton("Reset lines")
        self.reset_profile_lines_button.clicked.connect(self._reset_profile_lines)
        profile_row.addWidget(self.reset_profile_lines_button)
        layout.addLayout(profile_row)
        return group

    def _build_trace_group(self) -> QtWidgets.QTabWidget:
        tabs = QtWidgets.QTabWidget()
        roi_page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(roi_page)
        layout.setContentsMargins(4, 4, 4, 4)

        history_row = QtWidgets.QHBoxLayout()
        history_row.addWidget(QtWidgets.QLabel("History"))
        self.history_spin = QtWidgets.QSpinBox()
        self.history_spin.setRange(5, 3600)
        self.history_spin.setValue(120)
        self.history_spin.setSuffix(" s")
        history_row.addWidget(self.history_spin)
        history_row.addStretch()
        layout.addLayout(history_row)

        self.trace_plot = pg.PlotWidget()
        self.trace_plot.showGrid(x=True, y=True, alpha=0.25)
        self.trace_plot.setLabel("bottom", "Time", units="s")
        self.trace_plot.setLabel("left", "Summed counts")
        self.trace_curve = self.trace_plot.plot(
            pen=pg.mkPen("#ffd43b", width=2),
            antialias=True,
        )
        layout.addWidget(self.trace_plot)
        tabs.addTab(roi_page, "ROI sum over time")

        self.profile_plot = pg.PlotWidget()
        self.profile_plot.showGrid(x=True, y=True, alpha=0.25)
        self.profile_plot.setLabel("bottom", "Distance along line", units="px")
        self.profile_plot.setLabel("left", "Raw pixel value")
        self.profile_plot.addLegend()
        self.profile_curves = [
            self.profile_plot.plot(pen=pg.mkPen("#00d5ff", width=2), name="Line 1"),
            self.profile_plot.plot(pen=pg.mkPen("#ff9f43", width=2), name="Line 2"),
        ]
        self.profile_saturation_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen("#ff3030", width=2, style=QtCore.Qt.PenStyle.DashLine),
        )
        self.profile_plot.addItem(self.profile_saturation_line, ignoreBounds=True)
        tabs.addTab(self.profile_plot, "Line profiles")
        return tabs

    def _build_details_tabs(self) -> QtWidgets.QTabWidget:
        tabs = QtWidgets.QTabWidget()

        self.live_info_text = QtWidgets.QPlainTextEdit()
        self.live_info_text.setReadOnly(True)
        self.live_info_text.setStyleSheet("font-family: monospace")
        tabs.addTab(self.live_info_text, "Live diagnostics")

        self.camera_info_text = QtWidgets.QPlainTextEdit()
        self.camera_info_text.setReadOnly(True)
        self.camera_info_text.setPlaceholderText("Camera information will appear here")
        self.camera_info_text.setStyleSheet("font-family: monospace")
        tabs.addTab(self.camera_info_text, "Camera details")

        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.document().setMaximumBlockCount(400)
        self.log_text.setStyleSheet("font-family: monospace")
        tabs.addTab(self.log_text, "Activity log")
        tabs.setMinimumHeight(150)
        return tabs

    @QtCore.pyqtSlot()
    def connect_camera(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return

        self._prepare_new_session()
        requested_id = self.camera_id_edit.text().strip() or None
        self.connection_failed = False
        self.worker = CameraWorker(requested_id, self)
        self.worker.camera_opened.connect(self._camera_opened)
        self.worker.settings_changed.connect(self._settings_changed)
        self.worker.message.connect(self._log)
        self.worker.failed.connect(self._camera_failed)
        self.worker.finished.connect(self._camera_closed)

        self.status_label.setText("Connecting…")
        self.status_label.setStyleSheet("font-weight: bold; color: #b07000")
        self.connect_button.setEnabled(False)
        self.disconnect_button.setEnabled(True)
        self.camera_id_edit.setEnabled(False)
        self._log("Starting camera connection")
        self.worker.start()

    def _prepare_new_session(self) -> None:
        if self.roi is not None:
            self.image_view.getView().removeItem(self.roi)
            self.roi = None
        for marker in self.markers:
            self.image_view.getView().removeItem(marker)
        self.markers.clear()
        for line in self.profile_lines:
            if line is not None:
                self.image_view.getView().removeItem(line)
        self.profile_lines = [None, None]

        self.image = None
        self.camera_info = None
        self.camera_settings = None
        self.last_displayed_frame_count = None
        self.undisplayed_frames = 0
        self.camera_info_text.clear()
        self.live_info_text.clear()
        self.roi_info_text.clear()
        self.marker_info_text.clear()
        for curve in self.profile_curves:
            curve.setData([], [])
        self._clear_trace()

    @QtCore.pyqtSlot()
    def disconnect_camera(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        self.status_label.setText("Disconnecting…")
        self.disconnect_button.setEnabled(False)
        self._log("Stopping acquisition")
        self.worker.stop()

    @QtCore.pyqtSlot(object)
    def _camera_opened(self, info: CameraInfo) -> None:
        self.camera_info = info
        self.camera_settings = CameraSettings(
            exposure_time_us=info.exposure_time_us,
            gain=info.gain,
            gain_db=info.gain_db,
        )
        self.status_label.setText("Live")
        self.status_label.setStyleSheet("font-weight: bold; color: #14833b")
        self.setWindowTitle(
            f"{info.model} {info.serial_number} — Thorlabs camera viewer"
        )
        self.camera_id_edit.setText(info.camera_id)
        self._set_connected_controls(True)

        self.camera_info_text.setPlainText(
            "\n".join(
                [
                    f"Model:       {info.model}",
                    f"Name:        {info.name}",
                    f"Serial:      {info.serial_number}",
                    f"Firmware:    {info.firmware_version}",
                    f"Sensor:      {info.sensor_shape[1]} x {info.sensor_shape[0]}",
                    f"Image:       {info.image_shape[1]} x {info.image_shape[0]}",
                    f"Bit depth:   {info.bit_depth}",
                    f"Sensor type: {info.sensor_type}",
                    f"Interface:   {info.communication_interface} / {info.usb_port_type}",
                    f"SDK:         {info.sdk_version}",
                    "Exposure:    "
                    f"{info.exposure_range_us[0] / 1000:g}–"
                    f"{info.exposure_range_us[1] / 1000:g} ms",
                    f"Gain range:  {info.gain_range[0]}–{info.gain_range[1]}",
                ]
            )
        )

        self.exposure_spin.setRange(
            info.exposure_range_us[0] / 1000.0,
            info.exposure_range_us[1] / 1000.0,
        )
        self.exposure_spin.setValue(info.exposure_time_us / 1000.0)
        self.exposure_spin.setToolTip(
            "Allowed range: "
            f"{info.exposure_range_us[0] / 1000:g}–"
            f"{info.exposure_range_us[1] / 1000:g} ms"
        )
        self.gain_spin.setRange(*info.gain_range)
        self.gain_spin.setValue(info.gain)
        self.gain_spin.setEnabled(info.gain_range[1] > 0)
        self.profile_saturation_line.setValue((1 << info.bit_depth) - 1)
        self._log(
            f"Opened {info.model} with serial number {info.serial_number}; "
            f"{info.image_shape[1]} x {info.image_shape[0]}, {info.bit_depth}-bit"
        )
        if info.sensor_type == "BAYER":
            self._log(
                "This is a Bayer colour sensor. The viewer currently shows its "
                "raw mosaic, which can look like alternating rows."
            )
        self.panel_scroll.verticalScrollBar().setValue(0)

    @QtCore.pyqtSlot(object)
    def _settings_changed(self, settings: CameraSettings) -> None:
        self.camera_settings = settings
        self.exposure_spin.setValue(settings.exposure_time_us / 1000.0)
        self.gain_spin.setValue(settings.gain)

    @QtCore.pyqtSlot(str, str)
    def _camera_failed(self, summary: str, details: str) -> None:
        self.connection_failed = True
        self.status_label.setText("Error")
        self.status_label.setStyleSheet("font-weight: bold; color: #b03030")
        self._log(f"Camera error: {summary}")
        self._log(details.rstrip())

    @QtCore.pyqtSlot()
    def _camera_closed(self) -> None:
        if not self.connection_failed:
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("font-weight: bold; color: #b03030")
        self._set_connected_controls(False)
        self.connect_button.setEnabled(True)
        self.disconnect_button.setEnabled(False)
        self.camera_id_edit.setEnabled(True)

    def _set_connected_controls(self, connected: bool) -> None:
        self.exposure_spin.setEnabled(connected)
        self.gain_spin.setEnabled(connected)
        self.apply_settings_button.setEnabled(connected)
        self.reset_roi_button.setEnabled(connected)
        self.add_marker_button.setEnabled(connected)
        self.clear_markers_button.setEnabled(connected)
        for checkbox in self.profile_line_checks:
            checkbox.setEnabled(connected)
        self.reset_profile_lines_button.setEnabled(connected)
        self.red_at_top_check.setEnabled(connected)

    @QtCore.pyqtSlot()
    def apply_camera_settings(self) -> None:
        if self.worker is None or not self.worker.isRunning():
            return
        exposure_time_us = int(round(self.exposure_spin.value() * 1000.0))
        self.worker.set_camera_settings(exposure_time_us, self.gain_spin.value())

    @QtCore.pyqtSlot()
    def _display_latest_frame(self) -> None:
        if self.worker is None:
            return
        packet = self.worker.take_latest_frame()
        if packet is not None:
            self._frame_received(packet)

    @QtCore.pyqtSlot(object)
    def _frame_received(self, packet: FramePacket) -> None:
        first_frame = self.image is None
        self.image = packet.image
        if (
            self.last_displayed_frame_count is not None
            and packet.frame_count > self.last_displayed_frame_count
        ):
            self.undisplayed_frames += max(
                0, packet.frame_count - self.last_displayed_frame_count - 1
            )
        self.last_displayed_frame_count = packet.frame_count
        self.display_frames += 1
        now = time.monotonic()
        elapsed = now - self.display_rate_started
        if elapsed >= 1.0:
            self.display_fps = self.display_frames / elapsed
            self.display_frames = 0
            self.display_rate_started = now

        self.image_view.setImage(
            self.image,
            autoRange=first_frame,
            autoLevels=False,
            autoHistogramRange=first_frame,
        )
        if first_frame or self.auto_levels_check.isChecked():
            self._apply_image_levels()
        if first_frame:
            self._reset_roi(clear_trace=False)
            self._ensure_profile_lines()

        refresh_text = now - self.last_diagnostics_update >= 0.2
        self._update_roi(packet.received_at, refresh_text)
        if refresh_text:
            self._update_markers()
            self._update_line_profiles()
            self._update_live_diagnostics(packet)
            self.last_diagnostics_update = now

    def _update_live_diagnostics(self, packet: FramePacket) -> None:
        if self.image is None:
            return
        bit_depth = self.camera_info.bit_depth if self.camera_info else 16
        saturation_value = (1 << bit_depth) - 1
        saturated = int(np.count_nonzero(self.image >= saturation_value))
        saturated_percent = 100.0 * saturated / self.image.size
        display_levels = self.image_view.getImageItem().getLevels()
        if display_levels is None:
            display_levels_text = "unavailable"
        else:
            display_levels_text = f"{display_levels[0]:g} / {display_levels[1]:g}"
        timestamp = (
            "unavailable"
            if packet.camera_timestamp_ns is None
            else f"{packet.camera_timestamp_ns / 1e9:.6f} s"
        )
        settings = self.camera_settings
        if settings is None:
            exposure_text = "unavailable"
            gain_text = "unavailable"
        else:
            exposure_text = f"{settings.exposure_time_us / 1000:g} ms"
            gain_text = str(settings.gain)
            if settings.gain_db is not None:
                gain_text += f" ({settings.gain_db:.2f} dB)"
        text = "\n".join(
            [
                f"Frame number:    {packet.frame_count}",
                f"Camera time:     {timestamp}",
                f"SDK frame rate:  {packet.sdk_fps:.2f} fps",
                f"Polled rate:     {packet.acquisition_fps:.2f} fps",
                f"Displayed rate:  {self.display_fps:.2f} fps",
                f"Camera gaps:     {packet.skipped_frames}",
                f"Not displayed:   {self.undisplayed_frames}",
                f"Exposure:        {exposure_text}",
                f"Gain:            {gain_text}",
                f"Image dtype:     {self.image.dtype}",
                f"Image min/max:   {np.min(self.image):g} / {np.max(self.image):g}",
                f"Image mean:      {np.mean(self.image):.2f}",
                f"Display levels:  {display_levels_text}",
                f"Digital maximum: {saturation_value}",
                f"At digital max:  {saturated} ({saturated_percent:.4f}%)",
            ]
        )
        self._set_live_diagnostics_text(text)

    def _set_live_diagnostics_text(self, text: str) -> None:
        vertical = self.live_info_text.verticalScrollBar()
        horizontal = self.live_info_text.horizontalScrollBar()
        vertical_position = vertical.value()
        horizontal_position = horizontal.value()
        was_at_bottom = bool(self.live_info_text.toPlainText()) and (
            vertical_position == vertical.maximum()
        )

        self.live_info_text.setPlainText(text)

        vertical.setValue(vertical.maximum() if was_at_bottom else vertical_position)
        horizontal.setValue(horizontal_position)

    def _ensure_roi(self) -> None:
        if self.image is None or self.roi is not None:
            return
        height, width = self.image.shape[:2]
        roi_width = max(1, width // 4)
        roi_height = max(1, height // 4)
        self.roi = pg.RectROI(
            ((width - roi_width) // 2, (height - roi_height) // 2),
            (roi_width, roi_height),
            sideScalers=True,
            rotatable=False,
            scaleSnap=True,
            translateSnap=True,
            snapSize=1.0,
            pen=pg.mkPen((255, 255, 255, 230), width=2),
            hoverPen=pg.mkPen((0, 210, 255), width=3),
        )
        self.roi.setZValue(10)
        self.roi.sigRegionChangeFinished.connect(self._roi_moved)
        self.image_view.getView().addItem(self.roi)

    def _roi_bounds(self) -> tuple[int, int, int, int] | None:
        if self.image is None or self.roi is None:
            return None
        return clipped_roi_bounds(
            (self.roi.pos().x(), self.roi.pos().y()),
            (self.roi.size().x(), self.roi.size().y()),
            self.image.shape[:2],
        )

    def _update_roi(
        self, received_at: float, refresh_text: bool = True
    ) -> ROIStatistics | None:
        self._ensure_roi()
        bounds = self._roi_bounds()
        if self.image is None or bounds is None:
            return None

        statistics = roi_statistics(self.image, bounds)
        x0, x1, y0, y1 = statistics.bounds
        if refresh_text:
            self.roi_info_text.setPlainText(
                "\n".join(
                    [
                        f"Bounds: x={x0}:{x1}, y={y0}:{y1}",
                        f"Pixels: {statistics.pixel_count}",
                        f"Sum:    {statistics.total:.0f}",
                        f"Mean:   {statistics.mean:.3f}",
                        f"Min/max:{statistics.minimum:g} / {statistics.maximum:g}",
                    ]
                )
            )

        if not self.pause_trace_check.isChecked():
            if self.trace_started is None:
                self.trace_started = received_at
            elapsed = received_at - self.trace_started
            self.trace_times.append(elapsed)
            self.trace_counts.append(statistics.total)
            self._trim_trace(elapsed)
            if received_at - self.last_trace_plot_update >= 0.1:
                self.trace_curve.setData(
                    np.asarray(self.trace_times), np.asarray(self.trace_counts)
                )
                self.last_trace_plot_update = received_at
        return statistics

    @QtCore.pyqtSlot()
    def _roi_moved(self) -> None:
        bounds = self._roi_bounds()
        if bounds is None or self.roi is None:
            return
        x0, x1, y0, y1 = bounds
        self.roi.setPos((x0, y0), finish=False)
        self.roi.setSize((x1 - x0, y1 - y0), finish=False)
        self._clear_trace()
        self._log(f"ROI moved to x={x0}:{x1}, y={y0}:{y1}")

    @QtCore.pyqtSlot()
    def _reset_roi(self, clear_trace: bool = True) -> None:
        if self.image is None:
            return
        self._ensure_roi()
        if self.roi is None:
            return
        height, width = self.image.shape[:2]
        roi_width = max(1, width // 4)
        roi_height = max(1, height // 4)
        self.roi.setPos(
            ((width - roi_width) // 2, (height - roi_height) // 2), finish=False
        )
        self.roi.setSize((roi_width, roi_height), finish=False)
        if clear_trace:
            self._clear_trace()
            self._log("ROI reset to the centre of the image")

    def _trim_trace(self, elapsed: float) -> None:
        cutoff = elapsed - self.history_spin.value()
        while self.trace_times and self.trace_times[0] < cutoff:
            self.trace_times.popleft()
            self.trace_counts.popleft()

    @QtCore.pyqtSlot()
    def _clear_trace(self) -> None:
        self.trace_times.clear()
        self.trace_counts.clear()
        self.trace_started = None
        self.last_trace_plot_update = 0.0
        self.trace_curve.setData([], [])

    @QtCore.pyqtSlot(str)
    def _set_colour_map(self, name: str) -> None:
        self.image_view.setColorMap(
            _simple_colormap(
                name,
                red_at_top=self.red_at_top_check.isChecked(),
            )
        )
        self._log(f"Display colour map changed to {name}")

    @QtCore.pyqtSlot(bool)
    def _red_at_top_toggled(self, _enabled: bool) -> None:
        self._set_colour_map(self.colour_map_combo.currentText())

    @QtCore.pyqtSlot(bool)
    def _autoscale_toggled(self, enabled: bool) -> None:
        if enabled:
            self._apply_image_levels()

    @QtCore.pyqtSlot()
    def _auto_levels_once(self) -> None:
        if self._apply_image_levels():
            self.auto_levels_check.setChecked(False)

    def _apply_image_levels(self) -> bool:
        if self.image is None:
            return False
        low = float(np.nanmin(self.image))
        high = float(np.nanmax(self.image))
        if not math.isfinite(low) or not math.isfinite(high) or low == high:
            return False

        self._set_image_levels(low, high)
        return True

    def _set_image_levels(self, low: float, high: float) -> None:
        self.image_view.getImageItem().setLevels((low, high))
        histogram = getattr(self.image_view.ui, "histogram", None)
        if histogram is not None:
            try:
                histogram.setLevels(low, high)
            except Exception:
                # Older pyqtgraph releases expose only the region object here.
                histogram.region.setRegion((low, high))

    def _ensure_profile_lines(self) -> None:
        if self.image is None:
            return
        for index, checkbox in enumerate(self.profile_line_checks):
            if checkbox.isChecked() and self.profile_lines[index] is None:
                self._create_profile_line(index)

    def _create_profile_line(self, index: int) -> None:
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        if index == 0:
            points = [(0.15 * width, 0.5 * height), (0.85 * width, 0.5 * height)]
            colour = "#00d5ff"
        else:
            points = [(0.5 * width, 0.15 * height), (0.5 * width, 0.85 * height)]
            colour = "#ff9f43"

        line = pg.LineSegmentROI(
            points,
            pen=pg.mkPen(colour, width=2),
            hoverPen=pg.mkPen("white", width=3),
        )
        line.setZValue(12)
        line.sigRegionChanged.connect(lambda *_args: self._update_line_profiles())
        self.image_view.getView().addItem(line, ignoreBounds=True)
        self.profile_lines[index] = line

    def _toggle_profile_line(self, index: int, visible: bool) -> None:
        line = self.profile_lines[index]
        if visible:
            if line is None:
                self._create_profile_line(index)
        elif line is not None:
            self.image_view.getView().removeItem(line)
            self.profile_lines[index] = None
            self.profile_curves[index].setData([], [])
        self._update_line_profiles()

    @QtCore.pyqtSlot()
    def _reset_profile_lines(self) -> None:
        for index, line in enumerate(self.profile_lines):
            if line is not None:
                self.image_view.getView().removeItem(line)
                self.profile_lines[index] = None
        self._ensure_profile_lines()
        self._update_line_profiles()

    def _update_line_profiles(self) -> None:
        if self.image is None:
            return
        image_item = self.image_view.getImageItem()
        for line, curve in zip(self.profile_lines, self.profile_curves, strict=True):
            if line is None:
                curve.setData([], [])
                continue
            values = line.getArrayRegion(
                self.image,
                image_item,
                axes=(1, 0),
                order=0,
            )
            if values is None:
                curve.setData([], [])
                continue
            values = np.asarray(values).reshape(-1)
            curve.setData(np.arange(values.size), values)

    @QtCore.pyqtSlot(object)
    def _image_clicked(self, event) -> None:
        if not self.add_marker_button.isChecked() or self.image is None:
            return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        view = self.image_view.getView()
        point = view.mapSceneToView(event.scenePos())
        x = int(round(point.x()))
        y = int(round(point.y()))
        height, width = self.image.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return

        marker_number = len(self.markers) + 1
        marker = pg.TargetItem(
            pos=(x, y),
            size=14,
            symbol="crosshair",
            movable=True,
            pen=pg.mkPen("#00d5ff", width=2),
            label=f"M{marker_number}",
            labelOpts={"color": "#00d5ff"},
        )
        marker.setZValue(20)
        marker.sigPositionChangeFinished.connect(self._marker_moved)
        view.addItem(marker)
        self.markers.append(marker)
        self.add_marker_button.setChecked(False)
        self._update_markers()
        self._log(f"Placed marker M{marker_number} at x={x}, y={y}")
        event.accept()

    @QtCore.pyqtSlot()
    def _marker_moved(self) -> None:
        self._update_markers()

    def _update_markers(self) -> None:
        if self.image is None:
            return
        height, width = self.image.shape[:2]
        lines = []
        for index, marker in enumerate(self.markers, start=1):
            x = min(max(0, int(round(marker.pos().x()))), width - 1)
            y = min(max(0, int(round(marker.pos().y()))), height - 1)
            value = self.image[y, x]
            lines.append(f"M{index}: x={x:4d}, y={y:4d}, value={value}")
        self.marker_info_text.setPlainText("\n".join(lines))

    @QtCore.pyqtSlot()
    def _clear_markers(self) -> None:
        view = self.image_view.getView()
        for marker in self.markers:
            view.removeItem(marker)
        self.markers.clear()
        self.marker_info_text.clear()
        self._log("Cleared image markers")

    def _mouse_moved(self, event) -> None:
        if self.image is None:
            self.cursor_label.setText("No image")
            return
        scene_position = event[0]
        view = self.image_view.getView()
        if not view.sceneBoundingRect().contains(scene_position):
            return
        point = view.mapSceneToView(scene_position)
        x = int(math.floor(point.x()))
        y = int(math.floor(point.y()))
        height, width = self.image.shape[:2]
        if 0 <= x < width and 0 <= y < height:
            self.cursor_label.setText(f"Cursor: x={x}, y={y}, value={self.image[y, x]}")

    @QtCore.pyqtSlot(str)
    def _log(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{stamp}] {message}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(3000):
                self._log("Camera thread did not stop within three seconds")
                event.ignore()
                return
        event.accept()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="View and measure a locally connected Thorlabs camera"
    )
    parser.add_argument(
        "--camera-id",
        help="camera ID or serial number to open (default: first camera found)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("dnamic Thorlabs camera viewer")
    window = CameraViewer(camera_id=args.camera_id)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
