# Driver Drowsiness Detector

Real-time driver drowsiness detection using a laptop webcam, MediaPipe face landmarks, and computer-vision metrics that flag eye closure, yawning, and low blink rate before they become dangerous.

---

## Features

| Feature | Description |
|---------|-------------|
| **EAR** | Eye Aspect Ratio — detects eye closure in real time |
| **MAR** | Mouth Aspect Ratio — detects yawning |
| **PERCLOS** | % of eyelid closure over a rolling 3-second window |
| **Blink Rate** | Blinks per minute; abnormally low rate triggers an upgrade |
| **Multi-level Alerts** | Five severity tiers: NORMAL → LOW → MEDIUM → HIGH → CRITICAL |
| **Session Logging** | Every processed frame is written to a timestamped CSV in `data/logs/` |
| **Drive Summary** | A formatted console report + `.txt` file generated automatically at the end of each session |

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
pip install uv

# Install all project dependencies
uv sync
```

---

## Usage

```bash
uv run python main.py
```

Press **`q`** in the video window to stop. After quitting, the drive summary is printed automatically.

---

## Metrics Explained

### EAR — Eye Aspect Ratio
Measures how open your eyes are using six landmark points around each eye. A value around **0.3** is normal; values **below ~0.21** indicate the eye is closed. Sustained low EAR triggers drowsiness alerts.

### MAR — Mouth Aspect Ratio
Measures the vertical opening of the mouth relative to its width. A value **above 0.6** typically signals a yawn, which is an early sign of fatigue.

### PERCLOS — Percentage of Eyelid Closure
Tracks what fraction of the last 3 seconds (90 frames at 30 fps) your eyes were below the EAR threshold. Values **above 0.15** indicate drowsiness; above **0.30** is considered severe.

### Blink Rate
Normal blink rate is roughly 15–20 blinks per minute. A blink rate between **1 and 10 bpm** combined with elevated PERCLOS suggests drowsiness and upgrades the alert level by one step.

---

## Alert Levels

| Level | Trigger Condition |
|-------|-------------------|
| **NORMAL** | All metrics within safe range |
| **LOW** | Eyes closed for more than 10 consecutive frames |
| **MEDIUM** | Yawning detected (consecutive MAR frames exceeded) |
| **HIGH** | PERCLOS above threshold, or eyes closed for many consecutive frames, or blink-rate upgrade |
| **CRITICAL** | PERCLOS > 30 %, or eyes closed for 60+ consecutive frames |

> **Blink-rate upgrade rule:** if `1 ≤ blink_rate ≤ 10` *and* `PERCLOS > 10 %`, the computed level is stepped up by one tier.

---

## Session Logs

### CSV columns (`data/logs/session_YYYYMMDD_HHMMSS.csv`)

| Column | Description |
|--------|-------------|
| `timestamp` | Unix timestamp of the frame |
| `frame_num` | Sequential frame number within the session |
| `ear` | Average Eye Aspect Ratio for that frame |
| `mar` | Mouth Aspect Ratio for that frame |
| `perclos` | PERCLOS value (0–1) for that frame |
| `blink_rate` | Blinks per minute at that point in the session |
| `alert_level` | Drowsiness tier for that frame |
| `consecutive_drowsy_frames` | How many frames in a row EAR was below threshold |
| `microsleep_flag` | `1` if alert_level is CRITICAL, else `0` |

### Summary report (`data/logs/session_YYYYMMDD_HHMMSS.txt`)

Printed to the console and saved as a `.txt` file next to the CSV when the session ends:

```
====================================================
🚗  Drive Summary
====================================================
  Total Drive Time    : 12.4 min
  Drowsiness Score    : 87/100  🟢
  Peak Alert Level    : HIGH
  Microsleep Episodes : 0

  Time at Each Alert Level:
    NORMAL    :  78.3%
    LOW       :   9.1%
    MEDIUM    :   5.4%
    HIGH      :   7.2%
    CRITICAL  :   0.0%

  Time Spent Drowsy   : 7.2%
  Average EAR         : 0.284
  Average Blink Rate  : 17.3 blinks/min

  💡 Great drive! You stayed alert throughout.
====================================================
```

**Drowsiness score:** `100 × (1 − fraction_of_drowsy_frames)`. A score of 100 means no time spent in HIGH/CRITICAL; 0 means the entire session was severely drowsy.
