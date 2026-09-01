# HUD State Machine: Scanning Flow

The terminal's HUD is driven by a state machine (`app/scan_state_machine.py`)
that controls both the visual overlay (scan line, corner brackets, text
prompt) and when in the pipeline the liveness check and authorization
classification actually resolve into a decision.

## States

| State | Entry condition | Displayed | Exit condition |
|---|---|---|---|
| **IDLE** | Start, DENIED timeout, or manual reset | "Position your face within the frame", static guide frame | Face detected in frame → STABILIZING |
| **STABILIZING** | Face in frame | Frame "wakes up", no text yet | Stable for ~0.5s (`STABILIZE_DURATION`) → SCANNING. Face disappears → IDLE |
| **SCANNING** | Stability confirmed | "Hold still and blink", animated scan line, sharp brackets (no glow) | At least `MIN_SCAN_DURATION` (2s) elapsed AND blink detected AND classification ready → GRANTED or DENIED. Timeout (4s, `SCAN_TIMEOUT`) without a blink → DENIED. Face leaves the frame → straight to IDLE |
| **RESULT_GRANTED** | Authorized + liveness confirmed | "ACCESS GRANTED", full-screen message | After 1.5s (`GRANTED_DISPLAY_DURATION`) → WELCOME |
| **WELCOME** | From GRANTED | Stylized "logged in" screen (mocked, no real authentication/database) | Stays, or manual reset via the "Scan again" button |
| **RESULT_DENIED** | Unauthorized OR liveness timeout (shared screen for both) | "ACCESS DENIED", blinking red glow (~4 Hz) | After 3s (`DENIED_DISPLAY_DURATION`) → IDLE, or manual reset via the "Scan again" button |

## Timing parameters

Defined as constants in `app/scan_state_machine.py`:

- `STABILIZE_DURATION = 0.5` — stability requirement before scanning starts
- `MIN_SCAN_DURATION = 2.0` — minimum scan duration, matching one full scan-line cycle (`SCAN_PERIOD_SECONDS` in `app/hud_overlay.py`), so the result is never revealed before the animation has had time to read
- `SCAN_TIMEOUT = 4.0` — max wait time for a blink during SCANNING
- `GRANTED_DISPLAY_DURATION = 1.5` — display time for ACCESS GRANTED before Welcome
- `DENIED_DISPLAY_DURATION = 3.0` — display time for ACCESS DENIED before returning to IDLE

## Design decisions worth noting

- **Fixed guide frame, not a dynamic bounding box.** The HUD (scan line,
  corner brackets) is drawn at a fixed, centered position
  (`get_guide_frame()`) rather than following BlazeFace's real,
  dynamically detected bounding box. This eliminates a jitter problem:
  the real bounding box varies slightly frame to frame even when the
  subject is still, which made the corner brackets shake. The real
  bounding box is still used internally for the actual face crop fed to
  the embedding model, via `latest_frame_store`.
- **Liveness status is reused directly from `LivenessDetector`**, rather
  than the state machine building its own, parallel blink count.
  `LivenessDetector` already maintains a rolling window of blink
  timestamps internally (`blink_timestamps`, `blink_window_seconds`); a
  second, independent counter would risk drifting out of sync with it.
- **Single Writer Principle:** `video_frame_callback` is the sole writer
  to `ScanStateMachine` under normal operation (it reads the
  authorization result from `result_store`, which `embedding_worker`
  writes to separately). `force_reset()` (called from the "Scan again"
  button on the Streamlit main thread) is a deliberate, documented
  exception — a single button press is a negligible risk compared to
  continuous concurrent writes, so `ScanStateMachine` needs no thread
  safety of its own (no lock).
- **Deliberate minimum scan duration ("theater").** Without
  `MIN_SCAN_DURATION`, a fast classification result (~300-600ms) could
  make the scanning animation disappear before it had time to read as
  intentional. The computation itself is unaffected — only when the
  result is revealed is deliberately delayed.
- **"Unauthorized" and "liveness failed"** (a spoofing attempt, or no
  blink detected) share the same ACCESS DENIED screen, despite
  representing two different security layers (identity matching vs.
  liveness verification). A deliberate UX-simplification decision for
  this prototype.
- **The Welcome screen is a mockup** with no real account management or
  database — the idea of a full login flow is part of the design, but
  out of scope for the course/prototype.
- **A manual "Scan again" button** (`force_reset()`) was needed because
  a person who stays in front of the camera after a DENIED result would
  otherwise automatically cycle through the whole flow again (IDLE →
  STABILIZING → SCANNING → same result) roughly every 5.5 seconds, which
  can look like the system is "stuck" rather than actually resetting.

## Known technical pitfalls (resolved during development)

- **Blocking while-loops in Streamlit prevent button clicks.** A
  `while ctx.state.playing: ... time.sleep()` loop never returns control
  to Streamlit's script runner, so widget interactions (like button
  clicks) never get a chance to register. Resolved with
  `@st.fragment(run_every=...)`, which can rerun both on an interval and
  on user interaction without blocking the rest of the page.
- **Tight classification cycles can disrupt the WebRTC connection.**
  `embedding_worker` running classification back-to-back with no pause
  generates enough short-lived numpy temporaries per second to trigger
  frequent Python GC pauses, which can briefly hold the GIL and delay
  `aiortc`'s async event loop (WebRTC keepalive signals), which in turn
  can cause the connection to be considered dead and dropped. Resolved
  with a deliberate `time.sleep(0.3)` between classifications — harmless
  since `MIN_SCAN_DURATION` already floors scanning at 2 seconds
  regardless of how quickly the result is ready.