"""
Manuellt testskript för ScanStateMachine - simulerar en tidslinje av
update()-anrop för att verifiera alla tillståndsövergångar innan
integration i video_frame_callback.
"""

import time

from app.scan_state_machine import (
    DENIED_DISPLAY_DURATION,
    GRANTED_DISPLAY_DURATION,
    SCAN_TIMEOUT,
    STABILIZE_DURATION,
    ScanState,
    ScanStateMachine,
)


def check(machine, expected, label):
    actual = machine.state
    result = "OK" if actual == expected else "FEL"
    print(f"[{result}] {label}: förväntade {expected.name}, fick {actual.name}")


print("=== Scenario 1: full lyckad väg (IDLE -> ... -> WELCOME) ===")
m = ScanStateMachine()
check(m, ScanState.IDLE, "startläge")

m.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
check(m, ScanState.STABILIZING, "ansikte upptäckt")

m.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
check(m, ScanState.STABILIZING, "fortfarande stabiliserar (för tidigt)")

time.sleep(STABILIZE_DURATION + 0.1)
m.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
check(m, ScanState.SCANNING, "stabilitet uppnådd")

m.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
check(m, ScanState.SCANNING, "väntar fortfarande på liveness/auktorisering")

m.update(face_detected=True, liveness_status="levande", auth_result="authorized")
check(m, ScanState.RESULT_GRANTED, "levande + auktoriserad")

time.sleep(GRANTED_DISPLAY_DURATION + 0.1)
m.update(face_detected=True, liveness_status="levande", auth_result="authorized")
check(m, ScanState.WELCOME, "visningstid för GRANTED passerad")

m.update(face_detected=True, liveness_status="levande", auth_result="authorized")
check(m, ScanState.WELCOME, "stannar kvar i WELCOME")


print("\n=== Scenario 2: nekad väg (SCANNING -> RESULT_DENIED -> IDLE) ===")
m2 = ScanStateMachine()
m2.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
time.sleep(STABILIZE_DURATION + 0.1)
m2.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
check(m2, ScanState.SCANNING, "i SCANNING")

m2.update(face_detected=True, liveness_status="levande", auth_result="unauthorized")
check(m2, ScanState.RESULT_DENIED, "levande men ej auktoriserad")

time.sleep(DENIED_DISPLAY_DURATION + 0.1)
m2.update(face_detected=True, liveness_status="levande", auth_result="unauthorized")
check(m2, ScanState.IDLE, "visningstid för DENIED passerad")


print("\n=== Scenario 3: liveness-timeout (misstänkt foto) ===")
m3 = ScanStateMachine()
m3.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
time.sleep(STABILIZE_DURATION + 0.1)
m3.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
check(m3, ScanState.SCANNING, "i SCANNING")

time.sleep(SCAN_TIMEOUT + 0.1)
m3.update(face_detected=True, liveness_status="misstänkt_foto", auth_result=None)
check(m3, ScanState.RESULT_DENIED, "timeout utan levande-status")


print("\n=== Scenario 4: ansikte försvinner under STABILIZING ===")
m4 = ScanStateMachine()
m4.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
check(m4, ScanState.STABILIZING, "ansikte upptäckt")

m4.update(face_detected=False, liveness_status="osäker_ännu", auth_result=None)
check(m4, ScanState.IDLE, "ansikte försvann under stabilisering")


print("\n=== Scenario 5: ansikte försvinner under SCANNING ===")
m5 = ScanStateMachine()
m5.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
time.sleep(STABILIZE_DURATION + 0.1)
m5.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
check(m5, ScanState.SCANNING, "i SCANNING")

m5.update(face_detected=False, liveness_status="osäker_ännu", auth_result=None)
check(m5, ScanState.IDLE, "ansikte försvann under scanning - direkt avbrott")

print("\nKlart.")