"""
tests/test_scan_state_machine.py

Enhetstester för HUD-scanningsflödets tillståndsmaskin i
app/scan_state_machine.py.

Tidsberoende övergångar (STABILIZE_DURATION, SCAN_TIMEOUT,
GRANTED_DISPLAY_DURATION, DENIED_DISPLAY_DURATION) testas med en
mockad klocka (fake_clock-fixture) snarare än riktiga time.sleep()-
anrop, för snabb och deterministisk testkörning.
"""

import pytest

from app.scan_state_machine import (
    DENIED_DISPLAY_DURATION,
    GRANTED_DISPLAY_DURATION,
    MIN_SCAN_DURATION,
    SCAN_TIMEOUT,
    STABILIZE_DURATION,
    ScanState,
    ScanStateMachine,
)


class FakeClock:
    """
    Mutabel klocka för att styra time.time() deterministiskt i tester.

    Startar på ett godtyckligt fast värde (0.0) och flyttas framåt
    explicit via advance(), istället för att förlita sig på riktig
    förfluten tid mellan asserts.
    """

    def __init__(self):
        self.current = 0.0

    def time(self):
        return self.current

    def advance(self, seconds):
        self.current += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("app.scan_state_machine.time.time", clock.time)
    return clock


def _advance_to_stabilized(machine, fake_clock):
    """
    Hjälpfunktion: matar in ett upptäckt, stabilt ansikte tills
    STABILIZE_DURATION har passerat, dvs. tills maskinen når SCANNING.
    """
    machine.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
    fake_clock.advance(STABILIZE_DURATION + 0.1)
    machine.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)


class TestIdleAndStabilizing:
    def test_initial_state_is_idle(self, fake_clock):
        machine = ScanStateMachine()
        assert machine.state == ScanState.IDLE

    def test_face_detected_transitions_to_stabilizing(self, fake_clock):
        machine = ScanStateMachine()
        machine.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)
        assert machine.state == ScanState.STABILIZING

    def test_stays_in_stabilizing_before_duration_elapsed(self, fake_clock):
        machine = ScanStateMachine()
        machine.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)

        fake_clock.advance(STABILIZE_DURATION - 0.1)
        machine.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)

        assert machine.state == ScanState.STABILIZING

    def test_transitions_to_scanning_after_stabilize_duration(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)

        assert machine.state == ScanState.SCANNING

    def test_face_lost_during_stabilizing_returns_to_idle(self, fake_clock):
        machine = ScanStateMachine()
        machine.update(face_detected=True, liveness_status="osäker_ännu", auth_result=None)

        machine.update(face_detected=False, liveness_status="osäker_ännu", auth_result=None)

        assert machine.state == ScanState.IDLE


class TestScanning:
    def test_transitions_to_granted_when_alive_and_authorized(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)

        fake_clock.advance(MIN_SCAN_DURATION + 0.1)
        machine.update(face_detected=True, liveness_status="levande", auth_result="authorized")

        assert machine.state == ScanState.RESULT_GRANTED

    def test_does_not_grant_before_min_scan_duration_even_if_ready(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)

        # Liveness + auktorisering klara direkt, men MIN_SCAN_DURATION har
        # inte passerat än - scanning-animationen ska hinna synas.
        machine.update(face_detected=True, liveness_status="levande", auth_result="authorized")

        assert machine.state == ScanState.SCANNING

    def test_transitions_to_denied_when_alive_and_unauthorized(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)

        fake_clock.advance(MIN_SCAN_DURATION + 0.1)
        machine.update(face_detected=True, liveness_status="levande", auth_result="unauthorized")

        assert machine.state == ScanState.RESULT_DENIED

    def test_stays_in_scanning_while_waiting_for_classification(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)

        # liveness redan "levande", men auth_result saknas ännu
        # (embedding_worker inte hunnit klassificera) -> ingen övergång.
        machine.update(face_detected=True, liveness_status="levande", auth_result=None)

        assert machine.state == ScanState.SCANNING

    def test_times_out_to_denied_without_liveness_confirmation(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)

        fake_clock.advance(SCAN_TIMEOUT + 0.1)
        machine.update(face_detected=True, liveness_status="misstänkt_foto", auth_result=None)

        assert machine.state == ScanState.RESULT_DENIED

    def test_face_lost_during_scanning_returns_to_idle_immediately(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)

        machine.update(face_detected=False, liveness_status="osäker_ännu", auth_result=None)

        assert machine.state == ScanState.IDLE


class TestResultDisplayAndReset:
    def test_granted_transitions_to_welcome_after_display_duration(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)
        fake_clock.advance(MIN_SCAN_DURATION + 0.1)
        machine.update(face_detected=True, liveness_status="levande", auth_result="authorized")

        fake_clock.advance(GRANTED_DISPLAY_DURATION + 0.1)
        machine.update(face_detected=True, liveness_status="levande", auth_result="authorized")

        assert machine.state == ScanState.WELCOME

    def test_welcome_state_persists(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)
        fake_clock.advance(MIN_SCAN_DURATION + 0.1)
        machine.update(face_detected=True, liveness_status="levande", auth_result="authorized")
        fake_clock.advance(GRANTED_DISPLAY_DURATION + 0.1)
        machine.update(face_detected=True, liveness_status="levande", auth_result="authorized")

        fake_clock.advance(10.0)
        machine.update(face_detected=True, liveness_status="levande", auth_result="authorized")

        assert machine.state == ScanState.WELCOME

    def test_denied_transitions_to_idle_after_display_duration(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)
        fake_clock.advance(MIN_SCAN_DURATION + 0.1)
        machine.update(face_detected=True, liveness_status="levande", auth_result="unauthorized")

        fake_clock.advance(DENIED_DISPLAY_DURATION + 0.1)
        machine.update(face_detected=True, liveness_status="levande", auth_result="unauthorized")

        assert machine.state == ScanState.IDLE


class TestDisplayText:
    def test_idle_shows_positioning_prompt(self, fake_clock):
        machine = ScanStateMachine()
        assert machine.display_text() == "Position your face within the frame"

    def test_scanning_shows_hold_still_and_blink(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)
        assert machine.display_text() == "Hold still and blink"

    def test_granted_shows_access_granted(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)
        fake_clock.advance(MIN_SCAN_DURATION + 0.1)
        machine.update(face_detected=True, liveness_status="levande", auth_result="authorized")
        assert machine.display_text() == "ACCESS GRANTED"

    def test_denied_shows_access_denied(self, fake_clock):
        machine = ScanStateMachine()
        _advance_to_stabilized(machine, fake_clock)
        fake_clock.advance(MIN_SCAN_DURATION + 0.1)
        machine.update(face_detected=True, liveness_status="levande", auth_result="unauthorized")
        assert machine.display_text() == "ACCESS DENIED"