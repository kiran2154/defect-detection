from flask import Flask, render_template, request, jsonify
from datetime import datetime
import json
import os

app = Flask(__name__)

DATA_FILE = "detections_history.jsonl"  # line-by-line JSON
detections = []  # in‑memory list of all detections


def load_data():
    """Load previous detections from file at startup."""
    global detections
    detections = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    detections.append(json.loads(line.strip()))
                except Exception:
                    pass


def save_detection(record: dict):
    """Append one detection record to file + memory."""
    detections.append(record)
    with open(DATA_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detections", methods=["GET"])
def api_get_detections():
    return jsonify({"success": True, "data": detections})


@app.route("/api/add_detection", methods=["POST"])
def api_add_detection():
    """Called from your YOLO script for every processed image."""
    data = request.json or {}
    # make sure timestamp exists
    if "timestamp" not in data:
        data["timestamp"] = datetime.now().isoformat()
    save_detection(data)
    return jsonify({"success": True})


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Return summary for cards + charts."""
    total = len(detections)
    by_severity = {}
    by_class = {}

    for d in detections:
        for det in d.get("detections", []):
            sev = det.get("severity", "Unknown")
            cls = det.get("class_name", "Unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1
            by_class[cls] = by_class.get(cls, 0) + 1

    last_time = detections[-1]["timestamp"] if detections else None

    return jsonify(
        {
            "success": True,
            "data": {
                "total_images": total,
                "by_severity": by_severity,
                "by_class": by_class,
                "last_detection": last_time,
            },
        }
    )


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    load_data()
    app.run(debug=True)