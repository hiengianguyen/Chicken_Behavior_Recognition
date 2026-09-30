"""Firestore endpoints for UI-managed farm records."""

import os
import threading
import time
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore
from flask import Blueprint, current_app, jsonify, request


firestore_api = Blueprint("firestore_api", __name__)
COLLECTIONS = {"notifications", "devices", "automationRules", "power", "historyPower", "settings"}
_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        app = None
        try:
            app = firebase_admin.get_app()
        except ValueError:
            credentials_path = Path(
                os.getenv("FIREBASE_CREDENTIALS", "Database/key.json")
            )
            if not credentials_path.is_absolute():
                credentials_path = Path(__file__).resolve().parents[1] / credentials_path
            app = firebase_admin.initialize_app(
                credentials.Certificate(str(credentials_path))
            )

        _client = firestore.client(app=app)
        return _client


def _error_response(error):
    current_app.logger.exception("Firestore request failed", exc_info=error)
    return jsonify({"error": "Firestore is unavailable or the request is invalid."}), 503


def _valid_document_id(document_id):
    return bool(document_id) and len(document_id) <= 128 and "/" not in document_id


def _record_payload(payload):
    if not isinstance(payload, dict):
        return None
    record = dict(payload)
    record.pop("id", None)
    return record


@firestore_api.get("/api/data/<collection_name>")
def list_records(collection_name):
    if collection_name not in COLLECTIONS:
        return jsonify({"error": "Unknown collection."}), 404

    try:
        records = []
        for document in _get_client().collection(collection_name).limit(500).stream():
            records.append({"id": document.id, **(document.to_dict() or {})})
        return jsonify({"items": records})
    except Exception as error:
        return _error_response(error)


@firestore_api.put("/api/data/<collection_name>/<document_id>")
def upsert_record(collection_name, document_id):
    if collection_name not in COLLECTIONS:
        return jsonify({"error": "Unknown collection."}), 404
    payload = _record_payload(request.get_json(silent=True))
    if not _valid_document_id(document_id) or payload is None:
        return jsonify({"error": "A valid document ID and JSON object are required."}), 400

    try:
        _get_client().collection(collection_name).document(document_id).set(payload)
        return jsonify({"id": document_id, "saved": True})
    except Exception as error:
        return _error_response(error)


@firestore_api.delete("/api/data/<collection_name>/<document_id>")
def delete_record(collection_name, document_id):
    if collection_name not in COLLECTIONS:
        return jsonify({"error": "Unknown collection."}), 404
    if not _valid_document_id(document_id):
        return jsonify({"error": "A valid document ID is required."}), 400

    try:
        _get_client().collection(collection_name).document(document_id).delete()
        return jsonify({"id": document_id, "deleted": True})
    except Exception as error:
        return _error_response(error)


@firestore_api.post("/api/notifications/classify")
def classify_notification():
    payload = request.get_json(silent=True) or {}
    sensor_type = payload.get("sensorType")
    operator = payload.get("operator")
    try:
        value = float(payload.get("value"))
        threshold = float(payload.get("threshold"))
    except (TypeError, ValueError):
        return jsonify({"error": "Numeric value and threshold are required."}), 400

    if sensor_type not in {"TEMP", "HUM", "NH3", "CO2"} or operator not in {">", ">=", "<", "<="}:
        return jsonify({"error": "Unsupported sensor type or comparison operator."}), 400

    units = {"TEMP": "°C", "HUM": "%", "NH3": "ppm", "CO2": "ppm"}
    labels = {"TEMP": "nhiệt độ", "HUM": "độ ẩm", "NH3": "nồng độ NH3", "CO2": "nồng độ CO2"}
    operator_labels = {">": "lớn hơn", ">=": "lớn hơn hoặc bằng", "<": "nhỏ hơn", "<=": "nhỏ hơn hoặc bằng"}
    triggered = {
        ">": value > threshold,
        ">=": value >= threshold,
        "<": value < threshold,
        "<=": value <= threshold,
    }[operator]
    if not triggered:
        return jsonify({"triggered": False, "notification": None})

    unit = units[sensor_type]
    label = labels[sensor_type]
    location = str(payload.get("location", "Chuồng nuôi"))[:120]
    severity = "critical" if sensor_type in {"NH3", "CO2"} or abs(value - threshold) >= 5 else "warning"
    notification = {
        "id": f"ENV-{int(time.time() * 1000)}",
        "category": "environment",
        "severity": severity,
        "sensorType": sensor_type,
        "operator": operator,
        "threshold": threshold,
        "sensorValue": f"{value:g} {unit}",
        "unit": unit,
        "sensorLabel": label,
        "title": f"Cảnh báo {label} {operator_labels[operator]} {threshold:g} {unit}",
        "subtitle": f"{label.capitalize()} hiện tại: {value:g} {unit} · {location}",
        "description": f"Giá trị {label} đo được là {value:g} {unit}, đã vượt điều kiện {operator} {threshold:g} {unit}.",
        "timestamp": time.strftime("%H:%M - %d/%m/%Y"),
        "read": False,
    }

    if payload.get("save", True):
        try:
            _get_client().collection("notifications").document(notification["id"]).set(
                _record_payload(notification)
            )
        except Exception as error:
            return _error_response(error)

    return jsonify({"triggered": True, "notification": notification})


