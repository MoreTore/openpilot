#!/usr/bin/env python3
import json
import time
from dataclasses import dataclass

import numpy as np

from cereal import messaging, car
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.drive_helpers import MIN_SPEED
from openpilot.tools.longitudinal_maneuvers.maneuversd import Action, Maneuver as _Maneuver

STATUS_PARAM = "LateralManeuverStatus"

# thresholds for starting maneuvers
MAX_SPEED_DEV = 0.7  # deviation in m/s
MAX_CURV = 0.004  # 250 m radius
MAX_ROLL = 0.12  # 6.8°
TIMER = 2.0  # sec stable conditions before starting maneuver


def _default_status():
  return {
    "state": "idle",
    "phase": "",
    "maneuver": "",
    "runIndex": 0,
    "runTotal": 0,
    "stepIndex": 0,
    "stepTotal": 0,
    "phaseStepIndex": 0,
    "phaseStepTotal": 0,
    "uiShow": False,
    "uiSize": "small",
    "uiText1": "Lateral Maneuvers",
    "uiText2": "",
    "updatedAtSec": 0.0,
    "history": [],
  }


def _load_status(params: Params):
  status = _default_status()
  raw = params.get(STATUS_PARAM, encoding="utf-8") or ""
  if raw:
    if isinstance(raw, dict):
      status.update(raw)
    else:
      try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
          status.update(payload)
      except Exception:
        pass

  history = status.get("history")
  if not isinstance(history, list):
    history = []
  status["history"] = [str(line) for line in history if str(line).strip()][-120:]

  try:
    status["updatedAtSec"] = float(status.get("updatedAtSec") or 0.0)
  except Exception:
    status["updatedAtSec"] = 0.0

  return status


def _save_status(params: Params, status):
  status_copy = dict(status)
  history = status_copy.get("history")
  if not isinstance(history, list):
    history = []
  status_copy["history"] = [str(line) for line in history if str(line).strip()][-120:]
  status_copy["updatedAtSec"] = float(status_copy.get("updatedAtSec") or time.monotonic())
  params.put_nonblocking(STATUS_PARAM, status_copy)
  return status_copy


def _append_history(status, line):
  if not line:
    return status
  history = list(status.get("history", []))
  history.append(str(line))
  status["history"] = history[-120:]
  return status


def _status_signature(status) -> str:
  status_copy = dict(status)
  status_copy.pop("updatedAtSec", None)
  return json.dumps(status_copy, sort_keys=True, separators=(",", ":"))


@dataclass
class Maneuver(_Maneuver):
  _baseline_curvature: float = 0.0

  def get_accel(self, v_ego: float, lat_active: bool, curvature: float, roll: float) -> float:
    self._run_completed = False
    ready = abs(v_ego - self.initial_speed) < MAX_SPEED_DEV and lat_active and abs(curvature) < MAX_CURV and abs(roll) < MAX_ROLL
    self._ready_cnt = (self._ready_cnt + 1) if ready else max(self._ready_cnt - 1, 0)

    if self._ready_cnt > (TIMER / DT_MDL):
      if not self._active:
        self._baseline_curvature = curvature
      self._active = True

    if not self._active:
      return 0.0

    return self._step()

  def reset(self):
    super().reset()
    self._ready_cnt = 0


def _sine_action(amplitude, period, duration):
  t = np.linspace(0, duration, int(duration / DT_MDL) + 1)
  accel = amplitude * np.sin(2 * np.pi * t / period)
  return Action(accel.tolist(), t.tolist())


MANEUVERS = [
  Maneuver(
    "step right 20mph",
    [Action([0.5], [1.0]), Action([-0.5], [1.5])],
    repeat=2,
    initial_speed=20. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step left 20mph",
    [Action([-0.5], [1.0]), Action([0.5], [1.5])],
    repeat=2,
    initial_speed=20. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "sine 0.5Hz 20mph",
    [_sine_action(1.0, 2.0, 2.0), Action([0.0], [0.5])],
    repeat=2,
    initial_speed=20. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "jitter 20mph",
    [Action([-0.5 if i % 2 == 0 else 0.5], [0.1]) for i in range(10)],
    repeat=2,
    initial_speed=20. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step right 30mph",
    [Action([0.5], [1.0]), Action([-0.5], [1.5])],
    repeat=2,
    initial_speed=30. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "step left 30mph",
    [Action([-0.5], [1.0]), Action([0.5], [1.5])],
    repeat=2,
    initial_speed=30. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "sine 0.5Hz 30mph",
    [_sine_action(1.0, 2.0, 2.0), Action([0.0], [0.5])],
    repeat=2,
    initial_speed=30. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "jitter 30mph",
    [Action([-0.5 if i % 2 == 0 else 0.5], [0.1]) for i in range(10)],
    repeat=2,
    initial_speed=30. * CV.MPH_TO_MS,
  ),
]


def main():
  config_realtime_process(5, Priority.CTRL_LOW)

  params = Params()
  cloudlog.info("lateral_maneuversd is waiting for CarParams")
  messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)

  sm = messaging.SubMaster(['carState', 'carControl', 'controlsState', 'selfdriveState', 'modelV2'], poll='modelV2')
  pm = messaging.PubMaster(['lateralManeuverPlan', 'alertDebug'])

  maneuvers = iter(MANEUVERS)
  maneuver = None
  complete_cnt = 0
  aborted_cnt = 0
  abort_reason = ''
  display_holdoff = 0
  prev_text = ""
  last_started_run = None
  last_completed_run = None
  all_finished_logged = False

  status = _load_status(params)
  if status.get("state") in ("idle", "stopped"):
    status.update({
      "state": "armed",
      "phase": "",
      "maneuver": "",
      "runIndex": 0,
      "runTotal": 0,
      "stepIndex": 0,
      "stepTotal": 0,
      "phaseStepIndex": 0,
      "phaseStepTotal": 0,
      "uiShow": True,
      "uiSize": "small",
      "uiText1": "Lateral Maneuvers Armed",
      "uiText2": "Set target speed and let lateral stabilize.",
      "updatedAtSec": time.monotonic(),
    })
    _append_history(status, "Armed from The Galaxy. Stabilize on a straight, flat road to start.")

  last_status_signature = ""
  last_status_write = 0.0

  while True:
    sm.update()

    if maneuver is None:
      maneuver = next(maneuvers, None)

    alert_msg = messaging.new_message('alertDebug')
    alert_msg.valid = True
    plan_send = messaging.new_message('lateralManeuverPlan')

    accel = 0.0
    v_ego = max(sm['carState'].vEgo, 0.0)
    curvature = sm['controlsState'].desiredCurvature

    state = "idle"
    phase = ""
    maneuver_desc = maneuver.description if maneuver is not None else ""
    run_total = maneuver.repeat + 1 if maneuver is not None else 0
    run_index = maneuver._repeated + 1 if maneuver is not None else 0
    step_total = len(maneuver.actions) if maneuver is not None else 0
    step_index = maneuver._action_index + 1 if maneuver is not None and maneuver.active else 0

    if complete_cnt > 0:
      complete_cnt -= 1
      alert_msg.alertDebug.alertText1 = "Completed"
      alert_msg.alertDebug.alertText2 = maneuver_desc
      state = "completed"
      phase = "holdoff"
    elif maneuver is not None:
      CS = sm['carState']
      if CS.steeringPressed or CS.gasPressed:
        aborted_cnt = int(1.0 / DT_MDL)
        abort_reason = ('steering pressed' if CS.steeringPressed else 'gas pressed').ljust(20)
      aborted = aborted_cnt > 0
      speed_out_of_range = maneuver.active and abs(v_ego - maneuver.initial_speed) > MAX_SPEED_DEV
      if aborted or speed_out_of_range:
        maneuver.reset()

      roll = sm['carControl'].orientationNED[0] if len(sm['carControl'].orientationNED) == 3 else 0.0
      speed_ready = abs(v_ego - maneuver.initial_speed) < MAX_SPEED_DEV
      lat_ready = sm['carControl'].latActive
      curv_ok = abs(curvature) < MAX_CURV
      roll_ok = abs(roll) < MAX_ROLL

      accel = maneuver.get_accel(v_ego, lat_ready, curvature, roll)
      run_index = maneuver._repeated + 1 if not maneuver.finished else maneuver.repeat + 1
      step_index = maneuver._action_index + 1 if maneuver.active else 0

      if maneuver._run_completed:
        completed_run = maneuver._repeated + 1 if maneuver.finished else max(maneuver._repeated, 1)
        completed_key = (maneuver.description, completed_run)
        if last_completed_run != completed_key:
          _append_history(status, f"Completed {maneuver.description} run {completed_run}/{run_total}.")
          last_completed_run = completed_key
        complete_cnt = int(1.0 / DT_MDL)
        alert_msg.alertDebug.alertText1 = "Complete"
        alert_msg.alertDebug.alertText2 = maneuver.description
        state = "completed"
        phase = "run_complete"
      elif maneuver.active:
        started_key = (maneuver.description, maneuver._repeated + 1)
        if last_started_run != started_key:
          _append_history(status, f"Started {maneuver.description} run {maneuver._repeated + 1}/{run_total}.")
          last_started_run = started_key

        action_remaining = maneuver.actions[maneuver._action_index].time_bp[-1] - maneuver._action_frames * DT_MDL
        if maneuver.description.startswith("sine"):
          freq = maneuver.description.split()[1]
          alert_msg.alertDebug.alertText1 = f"Active sine {freq} {max(action_remaining, 0):.1f}s"
        else:
          alert_msg.alertDebug.alertText1 = f"Active {accel:+.1f}m/s² {max(action_remaining, 0):.1f}s"
        alert_msg.alertDebug.alertText2 = maneuver.description
        state = "running"
        phase = "active"
      elif aborted_cnt > 0:
        aborted_cnt -= 1
        alert_msg.alertDebug.alertText1 = abort_reason
        alert_msg.alertDebug.alertText2 = maneuver.description
        state = "aborted"
        phase = "abort"
      elif not (speed_ready and lat_ready):
        alert_msg.alertDebug.alertText1 = f"Set speed to {maneuver.initial_speed * CV.MS_TO_MPH:0.0f} mph"
        alert_msg.alertDebug.alertText2 = maneuver.description
        state = "setup"
        phase = "set_speed"
      elif maneuver._ready_cnt > 0:
        ready_time = max(TIMER - maneuver._ready_cnt * DT_MDL, 0)
        alert_msg.alertDebug.alertText1 = f"Starting: {int(ready_time) + 1}"
        alert_msg.alertDebug.alertText2 = maneuver.description
        state = "countdown"
        phase = "stabilizing"
      else:
        reason = "road not straight" if not curv_ok else "road not flat"
        if curv_ok and roll_ok:
          reason = "lateral not engaged"
        alert_msg.alertDebug.alertText1 = f"Waiting: {reason}"
        alert_msg.alertDebug.alertText2 = maneuver.description
        state = "waiting"
        phase = "road_check"
    else:
      alert_msg.alertDebug.alertText1 = "Maneuvers Finished"
      state = "finished"
      phase = "done"
      if not all_finished_logged:
        _append_history(status, "Finished all lateral maneuvers.")
        all_finished_logged = True

    setup_prefixes = ("Set speed", "Starting", "Waiting")
    text = alert_msg.alertDebug.alertText1
    same_text = text == prev_text or (text.startswith("Starting") and prev_text.startswith("Starting"))
    if not same_text and text.startswith(setup_prefixes) and prev_text.startswith(setup_prefixes) and display_holdoff > 0:
      alert_msg.alertDebug.alertText1 = prev_text
      display_holdoff -= 1
    else:
      prev_text = text
      display_holdoff = int(0.5 / DT_MDL) if text.startswith(setup_prefixes) else 0

    pm.send('alertDebug', alert_msg)

    plan_send.valid = maneuver is not None and maneuver.active and complete_cnt == 0
    if plan_send.valid:
      plan_send.lateralManeuverPlan.desiredCurvature = maneuver._baseline_curvature + accel / max(v_ego, MIN_SPEED) ** 2
    pm.send('lateralManeuverPlan', plan_send)

    status.update({
      "state": state,
      "phase": phase,
      "maneuver": alert_msg.alertDebug.alertText2 or maneuver_desc,
      "runIndex": run_index if maneuver is not None else 0,
      "runTotal": run_total,
      "stepIndex": step_index,
      "stepTotal": step_total,
      "phaseStepIndex": step_index,
      "phaseStepTotal": step_total,
      "uiShow": True,
      "uiSize": "mid" if alert_msg.alertDebug.alertText2 else "small",
      "uiText1": alert_msg.alertDebug.alertText1,
      "uiText2": alert_msg.alertDebug.alertText2,
    })

    status_signature = _status_signature(status)
    now = time.monotonic()
    if status_signature != last_status_signature or (now - last_status_write) >= 1.0:
      status["updatedAtSec"] = time.monotonic()
      _save_status(params, status)
      last_status_signature = status_signature
      last_status_write = now

    if maneuver is not None and maneuver.finished and complete_cnt == 0:
      maneuver = None


if __name__ == "__main__":
  main()
