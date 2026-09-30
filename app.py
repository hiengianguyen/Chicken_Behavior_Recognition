"""Flask API for real-time chicken inference.

The API owns one background inference worker. The worker reads the configured
video source, runs ChickenPipeline, and stores the newest encoded frame and
status snapshot for the UI endpoints to consume.
"""

import argparse
import json
import threading
import time

import cv2
import serial
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from src.firestore_api import firestore_api
from src.pipeline.pipeline import ChickenPipeline


class InferenceService:
    """Run one ChickenPipeline instance and expose its latest results safely."""

    def __init__(self, detector_model, behavior_model):
        self.detector_model = detector_model
        self.behavior_model = behavior_model
        self.pipeline = None
        self.capture = None
        self.worker = None
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.latest_frame = None
        self.latest_data = None
        self.frame_number = 0
        self.last_error = None
        self.source = None
        self.started_at = None

    def start(self, source=0):
        with self.condition:
            if self.worker and self.worker.is_alive():
                return False

            self.source = self._normalize_source(source)
            self.stop_event.clear()
            self.last_error = None
            self.frame_number = 0
            self.started_at = time.time()
            self.worker = threading.Thread(
                target=self._run,
                name="chicken-inference",
                daemon=True,
            )
            self.worker.start()
            return True

    def stop(self):
        self.stop_event.set()
        capture = self.capture
        if capture is not None:
            capture.release()

    def status(self):
        worker_running = bool(self.worker and self.worker.is_alive())
        with self.condition:
            data = self.latest_data
            return {
                "running": worker_running,
                "source": self.source,
                "frame_number": self.frame_number,
                "has_frame": self.latest_frame is not None,
                "last_error": self.last_error,
                "data_timestamp": data.get("timestamp") if data else None,
            }

    def latest(self):
        with self.condition:
            return self.latest_data

    def frame_stream(self):
        """Yield the newest JPEG as an MJPEG HTTP stream."""
        last_frame_number = -1
        while True:
            with self.condition:
                self.condition.wait_for(
                    lambda: self.frame_number != last_frame_number
                    or not (self.worker and self.worker.is_alive()),
                    timeout=2.0,
                )
                frame = self.latest_frame
                frame_number = self.frame_number
                running = bool(self.worker and self.worker.is_alive())

            if frame is not None and frame_number != last_frame_number:
                last_frame_number = frame_number
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
                )
            elif not running:
                break

    def _run(self):
        try:
            self.pipeline = ChickenPipeline(
                detector_model=self.detector_model,
                behavior_model=self.behavior_model,
            )
            self.capture = cv2.VideoCapture(self.source)
            if not self.capture.isOpened():
                raise RuntimeError(f"Cannot open video source: {self.source}")

            while not self.stop_event.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    break

                processed_frame = self.pipeline.process(frame)
                ok, encoded = cv2.imencode(".jpg", processed_frame)
                if not ok:
                    continue

                with self.condition:
                    self.latest_frame = encoded.tobytes()
                    self.latest_data = self.pipeline.status_manager.snapshot()
                    self.frame_number += 1
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.last_error = str(exc)
                self.condition.notify_all()
        finally:
            if self.capture is not None:
                self.capture.release()
            self.capture = None
            with self.condition:
                self.condition.notify_all()

    @staticmethod
    def _normalize_source(source):
        if isinstance(source, int):
            return source
        source = str(source)
        return int(source) if source.isdigit() else source


class ArduinoReader:
    """Read newline-delimited JSON from Arduino and keep the latest valid sample."""

    def __init__(self, port="COM3", baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.condition = threading.Condition()
        self.latest_data = None
        self.sample_id = 0
        self.last_error = None
        self.stop_event = threading.Event()
        self.serial = None
        self.worker = threading.Thread(
            target=self._run,
            name="arduino-reader",
            daemon=True,
        )
        self.worker.start()

    def latest(self):
        with self.condition:
            if (
                self.latest_data is not None
                and time.time() - self.latest_data["timestamp"] > 3
            ):
                return None, self.last_error or "No fresh sensor data received"
            return self.latest_data, self.last_error

    def _run(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            while not self.stop_event.is_set():
                raw_line = self.serial.readline()
                if not raw_line:
                    continue

                try:
                    payload = json.loads(raw_line.decode("utf-8").strip())
                    data = {
                        "temperature": float(payload["temperature"]),
                        "humidity": float(payload["humidity"]),
                        "gas": float(payload["gas"]),
                        "timestamp": time.time(),
                    }
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    continue

                with self.condition:
                    self.sample_id += 1
                    data["sample_id"] = self.sample_id
                    self.latest_data = data
                    self.last_error = None
                    self.condition.notify_all()
        except serial.SerialException as exc:
            with self.condition:
                self.last_error = str(exc)
        finally:
            if self.serial is not None and self.serial.is_open:
                self.serial.close()


def create_app():
    service = InferenceService(
        detector_model="models/best.pt",
        behavior_model="weights/best_model.pt",
    )

    app = Flask(__name__)
    CORS(app)
    app.register_blueprint(firestore_api)
    arduino = ArduinoReader(port="COM4", baudrate=9600)
    app.config["inference_service"] = service

    @app.get("/api/sensor")
    def get_sensor():
        data, error = arduino.latest()
        if data is None:
            return jsonify({"ready": False, "data": None, "error": error}), 202

        try:
            after = int(request.args.get("after", "-1"))
        except ValueError:
            return jsonify({"error": "after must be an integer"}), 400

        if after >= data["sample_id"]:
            return jsonify({
                "ready": True,
                "fresh": False,
                "sample_id": data["sample_id"],
            })
        return jsonify({
            "ready": True,
            "fresh": True,
            "sample_id": data["sample_id"],
            "data": data,
        })

    @app.get("/api/video")
    def video():
        source = request.args.get("source", "0")
        if not service.worker or not service.worker.is_alive():
            service.start(source)
        return Response(
            service.frame_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/latest-data")
    def latest_data():
        data = service.latest()
        if data is None:
            return jsonify({"ready": False, "data": None}), 202
        return jsonify({"ready": True, "data": data})

    @app.get("/api/status")
    def api_status():
        return jsonify(service.status())

    @app.post("/api/start")
    def start():
        payload = request.get_json(silent=True) or {}
        source = payload.get("source", request.args.get("source", "0"))
        started = service.start(source)
        return jsonify({"started": started, **service.status()}), 202

    @app.post("/api/stop")
    def stop():
        service.stop()
        return jsonify({"stopped": True, **service.status()})

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, **service.status()})

    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChickenCareAI Flask API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
