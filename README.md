# Smart Attendance System using Face Recognition

A local Python application that uses a webcam and face recognition to enroll participants, mark attendance once per person per calendar day, store records in SQLite, and export attendance to CSV.

## Repository status

The original repository contained only a README, so there was no application source code, dependency manifest, or tests to repair. This commit adds a minimal working command-line implementation rather than claiming to have fixed source files that were not present.

## Requirements

- Python 3.10 or 3.11 recommended
- Webcam
- Windows, macOS, or Linux with a working OpenCV camera backend
- `face-recognition` depends on `dlib`; installation may require a compiler or a compatible prebuilt package, depending on your OS/Python version.

## Setup

Create and activate a virtual environment:

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `face-recognition` or `dlib` fails to install, use a Python version supported by the available dlib build tools for your operating system and consult the package installation instructions. The app cannot run until those dependencies install successfully.

## Run

1. Enroll a participant (with their informed consent):
   ```bash
   python app.py enroll --name "Jane Doe"
   ```
   Look at the webcam and press **Space** when exactly one face is visible. Press **Esc** to cancel.

2. Start attendance:
   ```bash
   python app.py run
   ```
   Press **Q** to stop. A person's attendance is recorded at most once per calendar day.

3. Export records:
   ```bash
   python app.py export --output attendance.csv
   ```

Use `--camera 1` if your webcam is not device 0. Run `python app.py --help` for command help.

## Data and privacy

- The SQLite database is created at `data/attendance.db` on first run.
- Face encodings and attendance records stay local by default; no email or cloud upload is implemented.
- The `data/` directory is ignored by Git because it contains biometric templates and attendance information.
- Obtain informed consent before enrollment, restrict access to the database, and provide a non-biometric attendance alternative. Face matching can produce false matches or missed matches; verify records before relying on them for consequential decisions.
- Delete the local `data/` directory to remove enrolled face encodings and attendance records.

## Current limitations

- One face sample is stored per person; lighting, camera quality, pose, and demographic differences can affect accuracy.
- Face recognition is not liveness detection; a photo or screen may fool a basic webcam setup.
- No authentication, admin UI, email notifications, or remote hosting is included.
- Test with your own webcam and environment before real use. This repository has not been run on your machine, so local installation and camera access still need verification.
