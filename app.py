"""Local face-recognition attendance tracker.

Usage:
  python app.py enroll --name "Jane Doe"
  python app.py run
  python app.py export --output attendance.csv

Only enroll people who have explicitly agreed to biometric processing.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "attendance.db"
TOLERANCE = 0.48
FRAME_SCALE = 0.25


def connect_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            encoding BLOB NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL REFERENCES people(id),
            attended_at TEXT NOT NULL,
            attendance_date TEXT NOT NULL,
            UNIQUE(person_id, attendance_date)
        )"""
    )
    conn.commit()
    return conn


def get_known_faces(conn: sqlite3.Connection):
    rows = conn.execute("SELECT id, name, encoding FROM people ORDER BY name").fetchall()
    ids, names, encodings = [], [], []
    for person_id, name, blob in rows:
        vector = np.frombuffer(blob, dtype=np.float64)
        if vector.shape != (128,):
            print(f"Skipping invalid face data for {name!r}. Re-enroll this person.")
            continue
        ids.append(person_id)
        names.append(name)
        encodings.append(vector)
    return ids, names, encodings


def enroll(name: str, camera_index: int) -> int:
    name = " ".join(name.split()).strip()
    if not name:
        raise ValueError("Name cannot be empty.")
    conn = connect_db()
    try:
        if conn.execute("SELECT 1 FROM people WHERE name = ?", (name,)).fetchone():
            raise ValueError(f"{name!r} is already enrolled.")
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions or --camera.")
        print("Look at the camera. Press SPACE to capture a frame, or ESC to cancel.")
        encoding = None
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Could not read a frame from the webcam.")
                cv2.putText(frame, "SPACE: capture | ESC: cancel", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Enroll face", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("Enrollment cancelled.")
                    return 1
                if key == 32:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    locations = face_recognition.face_locations(rgb, model="hog")
                    if len(locations) != 1:
                        print(f"Please ensure exactly one face is visible (found {len(locations)}).")
                        continue
                    vectors = face_recognition.face_encodings(rgb, locations)
                    if not vectors:
                        print("Could not encode the face; try again with better lighting.")
                        continue
                    encoding = vectors[0]
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()
        if encoding is None:
            return 1
        conn.execute(
            "INSERT INTO people(name, encoding, created_at) VALUES (?, ?, ?)",
            (name, encoding.astype(np.float64).tobytes(), datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()
        print(f"Enrolled {name!r}. Face encoding stored locally in {DB_PATH}.")
        return 0
    finally:
        conn.close()


def run_attendance(camera_index: int) -> int:
    conn = connect_db()
    try:
        person_ids, names, known = get_known_faces(conn)
        if not known:
            print("No enrolled faces. Run: python app.py enroll --name \"Full Name\"")
            return 1
        camera = cv2.VideoCapture(camera_index)
        if not camera.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions or --camera.")
        print("Attendance is running. Press Q to quit.")
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Warning: failed to read a webcam frame.")
                    break
                small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(rgb, model="hog")
                vectors = face_recognition.face_encodings(rgb, locations)
                labels = []
                for vector in vectors:
                    distances = face_recognition.face_distance(known, vector)
                    best = int(np.argmin(distances))
                    if distances[best] <= TOLERANCE:
                        name = names[best]
                        person_id = person_ids[best]
                        now = datetime.now()
                        cur = conn.execute(
                            """INSERT OR IGNORE INTO attendance
                               (person_id, attended_at, attendance_date) VALUES (?, ?, ?)""",
                            (person_id, now.isoformat(timespec="seconds"), now.date().isoformat()),
                        )
                        if cur.rowcount:
                            conn.commit()
                            print(f"Attendance marked: {name} at {now:%Y-%m-%d %H:%M:%S}")
                        labels.append(name)
                    else:
                        labels.append("Unknown")
                for (top, right, bottom, left), label in zip(locations, labels):
                    top, right, bottom, left = [int(v / FRAME_SCALE) for v in (top, right, bottom, left)]
                    color = (0, 180, 0) if label != "Unknown" else (0, 0, 220)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, label, (left, max(20, top - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                cv2.imshow("Smart Attendance - press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()
        return 0
    finally:
        conn.close()


def export_attendance(output: str) -> int:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    conn = connect_db()
    try:
        rows = conn.execute(
            """SELECT p.name, a.attended_at, a.attendance_date
               FROM attendance a JOIN people p ON p.id = a.person_id
               ORDER BY a.attended_at"""
        ).fetchall()
        with destination.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["name", "attended_at", "attendance_date"])
            writer.writerows(rows)
        print(f"Exported {len(rows)} attendance record(s) to {destination}")
        return 0
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Local webcam face-recognition attendance tracker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    enroll_parser = subparsers.add_parser("enroll", help="Enroll one person using the webcam")
    enroll_parser.add_argument("--name", required=True, help="Person's full name")
    enroll_parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    run_parser = subparsers.add_parser("run", help="Recognize enrolled people and mark daily attendance")
    run_parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    export_parser = subparsers.add_parser("export", help="Export attendance records to CSV")
    export_parser.add_argument("--output", default="attendance.csv", help="Output CSV path")
    args = parser.parse_args()
    try:
        if args.command == "enroll":
            return enroll(args.name, args.camera)
        if args.command == "run":
            return run_attendance(args.camera)
        return export_attendance(args.output)
    except (ValueError, RuntimeError, sqlite3.Error, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
