"""
EmotionVision — main.py
=======================
"""

import sys
import logging
import datetime
from pathlib import Path

import cv2
import numpy as np
import time

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QButtonGroup
from PyQt5.QtCore import QDate, QTimer, Qt, pyqtSlot, QThread, pyqtSignal, QTime
from PyQt5.QtGui import QPixmap, QImage

from face_emotion_ui import Ui_MainWindow
from facemodel import detect_face_emotion, rest_emotions
import config

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# VideoThread
# ─────────────────────────────────────────────────────────────────────────────
class VideoThread(QThread):
    """Captures frames in a background thread and emits them to the main window.

    Signals
    -------
    frame_updated(raw_frame, annotated_frame, fps, emotion_index)
    """

    frame_updated = pyqtSignal(np.ndarray, np.ndarray, int, int)

    def __init__(self, camera_index: int = 0) -> None:
        super().__init__()
        self._camera_index = camera_index
        self._running = False

    # ── Public API ────────────────────────────────────────────────────────────
    def stop(self) -> None:
        """Ask the thread to finish after the current frame."""
        self._running = False

    # ── Thread body ───────────────────────────────────────────────────────────
    def run(self) -> None:
        self._running = True
        cap = cv2.VideoCapture(self._camera_index)

        if not cap.isOpened():
            log.error("Cannot open camera index %d", self._camera_index)
            return

        frames_to_count: int = config.FPS_SAMPLE_FRAMES
        cnt: int = 0
        fps: int = 0
        st: float = time.time()

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            #print(f"Frame shape is: {frame.shape}")
            if not ret:
                log.warning("Empty frame received — skipping.")
                continue

            raw_frame = np.copy(frame)
            annotated_frame, emotion_index = detect_face_emotion(frame)

            cnt += 1
            if cnt == frames_to_count:
                try:
                    fps = round(frames_to_count / (time.time() - st))
                except ZeroDivisionError:
                    fps = 0
                st = time.time()
                cnt = 0

            self.frame_updated.emit(raw_frame, annotated_frame, fps, emotion_index)

        cap.release()
        log.info("VideoThread stopped cleanly.")


# ─────────────────────────────────────────────────────────────────────────────
# FacialApp  — main window
# ─────────────────────────────────────────────────────────────────────────────
class FacialApp(QMainWindow, Ui_MainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)

        # ── State ─────────────────────────────────────────────────────────────
        self.started: bool = False
        self.checkbox_status: bool = False   # user ticked emotion before starting
        self.checked: bool = False           # at least one emotion btn is active
        self.selfie_label: str = ""          # text of currently selected emotion btn
        self.xx: str | None = None           # emotion we're waiting to capture
        self.domin: int | None = None
        self.temp: np.ndarray | None = None
        self.emotions: list[int] = []

        # Latest raw frame stored so the selfie button always has fresh data
        self._latest_raw: np.ndarray | None = None
        self._latest_index: int = -1

        # ── Date / time labels ────────────────────────────────────────────────
        now = QDate.currentDate()
        self.date_label.setText(now.toString("ddd dd MMM yyyy"))
        self.time_label.setText(datetime.datetime.now().strftime("%I:%M %p"))

        # ── Optional splash images (graceful if files absent) ─────────────────
        self._try_set_pixmap(self.img_label,  config.SPLASH_IMAGE)
        self._try_set_pixmap(self.selfie,     config.SELFIE_PLACEHOLDER)

        # ── Session timer ─────────────────────────────────────────────────────
        self._elapsed = QTime(0, 0)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick_timer)

        # ── Video thread ──────────────────────────────────────────────────────
        self.video_thread = VideoThread(camera_index=config.CAMERA_INDEX)
        self.video_thread.frame_updated.connect(self._on_frame)

        # ── Emotion button group (replaces 5 separate stateChanged handlers) ──
        self._emotion_group = QButtonGroup(self)
        self._emotion_group.setExclusive(True)
        for btn in (
            self.checkBox,
            self.checkBox_2,
            self.checkBox_3,
            self.checkBox_4,
            self.checkBox_5,
        ):
            self._emotion_group.addButton(btn)
        self._emotion_group.buttonClicked.connect(self._on_emotion_selected)

        # ── Button connections ────────────────────────────────────────────────
        self.Start.clicked.connect(self._on_start_clicked)
        self.Log_out_Button.clicked.connect(self._on_stop_clicked)
        self.selfie_button.clicked.connect(self._on_selfie_clicked)
        self.save_button.clicked.connect(self._on_save_clicked)
    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _try_set_pixmap(label: QtWidgets.QLabel, path: str) -> None:
        """Set a pixmap on a label if the file exists; silently skip otherwise."""
        p = Path(path)
        if p.exists():
            label.setPixmap(QPixmap(str(p)))
        else:
            log.debug("Image not found, skipping: %s", path)

    def _notify(self, message: str) -> None:
        QMessageBox.information(self, "EmotionVision", message)

    # ─────────────────────────────────────────────────────────────────────────
    # Button handlers
    # ─────────────────────────────────────────────────────────────────────────
    def _on_start_clicked(self) -> None:
        """Start live stream on first press; ignore subsequent presses."""
        self.Start.setChecked(True)
        self.Log_out_Button.setChecked(False)

        if not self.started:
            self.started = True
            log.info("Stream started.")
            for btn in self._emotion_group.buttons():
                btn.setCheckable(True)
            self._elapsed = QTime(0, 0)
            self._timer.start(1000)
            self.video_thread.start()

            # Show live badge if the UI exposes one
            if hasattr(self, "live_badge"):
                self.live_badge.setVisible(True)

    def _on_stop_clicked(self) -> None:
        """Stop the stream and reset UI state."""
        self.Log_out_Button.setChecked(True)
        self.Start.setChecked(False)

        self.started = False
        self._timer.stop()
        self.video_thread.stop()
        # Give the thread up to 2 s to finish cleanly before forcing it
        if not self.video_thread.wait(2000):
            log.warning("VideoThread did not stop in time — terminating.")
            self.video_thread.terminate()

        self.selfie_button.setCheckable(False)
        for btn in self._emotion_group.buttons():
            btn.setCheckable(False)
            btn.setChecked(False)

        self.checked = False
        self.selfie_label = ""
        self.xx = None

        if hasattr(self, "live_badge"):
            self.live_badge.setVisible(False)

        log.info("Stream stopped.")

    def _on_selfie_clicked(self) -> None:
        """Capture a selfie frame when the detected emotion matches the selection."""
        if not self.started:
            self._notify(
                'Press "Start" to begin the live stream before taking a selfie.'
            )
            return

        if not self.checked:
            self._notify(
                "Please select a target emotion before capturing a selfie."
            )
            return

        # The actual capture happens inside _on_frame whenever conditions are met.
        self.selfie_button.setCheckable(True)
        self.selfie_button.setChecked(True)
        self.xx = self.selfie_label
        log.debug("Waiting to capture: target emotion = %s", self.xx)

    def _on_emotion_selected(self, btn: QtWidgets.QPushButton) -> None:
        """Called whenever one of the five emotion buttons is toggled."""
        self.checked = True
        # Strip leading whitespace added in the UI for padding
        self.selfie_label = btn.text().strip().lower()
        log.debug("Emotion selected: %s", self.selfie_label)
    def _on_save_clicked(self) -> None:
        if self.selfie.pixmap() is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save image", "selfie.png", "Images (*.png *.jpg)"
        )
        if path:
            if hasattr(self, '_raw_selfie'):
                cv2.imwrite(path, self._raw_selfie)
            else:
                self.selfie.pixmap().save(path)
            log.info("Selfie saved to %s", path)
    # ─────────────────────────────────────────────────────────────────────────
    # Timer
    # ─────────────────────────────────────────────────────────────────────────
    def _tick_timer(self) -> None:
        self._elapsed = self._elapsed.addSecs(1)
        self.total_time.setText(self._elapsed.toString("hh:mm:ss"))

    # ─────────────────────────────────────────────────────────────────────────
    # Frame processing
    # ─────────────────────────────────────────────────────────────────────────
    @pyqtSlot(np.ndarray, np.ndarray, int, int)
    def _on_frame(
        self,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        fps: int,
        emotion_index: int,
    ) -> None:
        """Receive a processed frame from VideoThread and update the UI."""
        # Store for selfie capture
        self._latest_raw = raw_frame
        self._latest_index = emotion_index

        # FPS overlay
        cv2.putText(
            annotated_frame,
            f"FPS: {fps}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 120),
            2,
            cv2.LINE_AA,
        )

        self._display_frame(annotated_frame)
        self._update_dominant(emotion_index)

        # Attempt selfie capture if button is armed
        if self.selfie_button.isChecked():
            self._try_capture_selfie(raw_frame, emotion_index)

    def _display_frame(self, frame: np.ndarray) -> None:
        """Convert a BGR OpenCV frame to QPixmap and show it."""
        self.temp = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.img_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.img_label.width(),
                self.img_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _update_dominant(self, emotion_index: int) -> None:
        """Accumulate emotion samples and update the dominant-emotion label."""
        self.emotions.append(emotion_index)
        if len(self.emotions) >= config.DOMINANT_SAMPLE_FRAMES:
            unique, counts = np.unique(self.emotions, return_counts=True)
            dominant_pos = int(np.argmax(counts))
            self.domin = int(unique[dominant_pos])
            if self.domin in rest_emotions:
                self.status_label.setText(rest_emotions[self.domin].capitalize())
            self.emotions.clear()

    def _try_capture_selfie(self, raw_frame: np.ndarray, emotion_index: int) -> None:
        """Save a selfie frame when the detected emotion matches the target."""
        if emotion_index == -1:
            return
        detected = rest_emotions.get(emotion_index, "")
        if self.xx and detected.lower() == self.xx.lower():
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(
                rgb,
                (config.SELFIE_WIDTH, config.SELFIE_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self._raw_selfie=raw_frame.copy()
            self.selfie.setPixmap(QPixmap.fromImage(qimg))
            self.selfie_button.setChecked(False)
            self.save_button.setEnabled(True)
            self.xx = None
            log.info("Selfie captured — emotion: %s", detected)

    # ─────────────────────────────────────────────────────────────────────────
    # Window close
    # ─────────────────────────────────────────────────────────────────────────
    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Ensure the camera thread is stopped before the window closes."""
        self.video_thread.stop()
        self.video_thread.wait(2000)
        log.info("Application closed.")
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_gui_app() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FacialApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui_app()