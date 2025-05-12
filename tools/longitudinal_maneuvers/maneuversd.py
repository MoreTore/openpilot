#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass

import builtins
from cereal import messaging, car
from opendbc.car.common.conversions import Conversions as CV
from openpilot.common.realtime import DT_MDL
from openpilot.common.params import Params
from openpilot.common.swaglog import cloudlog

# Override print to also send to cloudlog.debug
_builtin_print = builtins.print

def print(*args, **kwargs):
  _builtin_print(*args, **kwargs)
  try:
    cloudlog.debug(' '.join(str(a) for a in args))
  except Exception:
    pass

@dataclass
class Action:
  accel_bp: list[float]  # m/s^2
  time_bp: list[float]   # seconds

  def __post_init__(self):
    assert len(self.accel_bp) == len(self.time_bp)


@dataclass
class Maneuver:
  description: str
  actions: list[Action]
  repeat: int = 0
  initial_speed: float = 0.  # m/s

  _active: bool = False
  _finished: bool = False
  _action_index: int = 0
  _action_frames: int = 0
  _ready_cnt: int = 0
  _repeated: int = 0

  def get_accel(self, v_ego: float, long_active: bool, standstill: bool, cruise_standstill: bool) -> float:
    print(f"[DEBUG] get_accel: v_ego={v_ego:.2f}, long_active={long_active}, standstill={standstill}, cruise_standstill={cruise_standstill}, initial_speed={self.initial_speed:.2f}")
    ready = abs(v_ego - self.initial_speed) < 0.3 and long_active and not cruise_standstill
    if self.initial_speed < 0.01:
      ready = ready and standstill
    self._ready_cnt = (self._ready_cnt + 1) if ready else 0
    print(f"[DEBUG] ready={ready}, _ready_cnt={self._ready_cnt}, _active={self._active}, _finished={self._finished}, _action_index={self._action_index}, _action_frames={self._action_frames}, _repeated={self._repeated}")

    if self._ready_cnt > (3. / DT_MDL):
      self._active = True
      print(f"[DEBUG] Maneuver activated")

    if not self._active:
      print(f"[DEBUG] Not active, returning speed correction: {min(max(self.initial_speed - v_ego, -2.), 2.)}")
      return min(max(self.initial_speed - v_ego, -2.), 2.)

    action = self.actions[self._action_index]
    action_accel = np.interp(self._action_frames * DT_MDL, action.time_bp, action.accel_bp)
    print(f"[DEBUG] Action index: {self._action_index}, action_frames: {self._action_frames}, action_accel: {action_accel}")

    self._action_frames += 1

    # reached duration of action
    if self._action_frames > (action.time_bp[-1] / DT_MDL):
      print(f"[DEBUG] Action duration reached for index {self._action_index}")
      # next action
      if self._action_index < len(self.actions) - 1:
        self._action_index += 1
        self._action_frames = 0
        print(f"[DEBUG] Moving to next action: {self._action_index}")
      # repeat maneuver
      elif self._repeated < self.repeat:
        self._repeated += 1
        self._action_index = 0
        self._action_frames = 0
        self._active = False
        print(f"[DEBUG] Repeating maneuver: repeated={self._repeated}")
      # finish maneuver
      else:
        self._finished = True
        print(f"[DEBUG] Maneuver finished")

    return float(action_accel)

  @property
  def finished(self):
    return self._finished

  @property
  def active(self):
    return self._active


MANEUVERS = [
  Maneuver(
    "come to stop",
    [Action([-0.5], [12])],
    repeat=2,
    initial_speed=5.,
  ),
  Maneuver(
    "start from stop",
    [Action([1.5], [6])],
    repeat=2,
    initial_speed=0.,
  ),
  Maneuver(
    "creep: alternate between +1m/s^2 and -1m/s^2",
    [
      Action([1], [3]), Action([-1], [3]),
      Action([1], [3]), Action([-1], [3]),
      Action([1], [3]), Action([-1], [3]),
    ],
    repeat=2,
    initial_speed=0.,
  ),
  Maneuver(
    "brake step response: -1m/s^2 from 20mph",
    [Action([-1], [3])],
    repeat=2,
    initial_speed=20. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "brake step response: -4m/s^2 from 20mph",
    [Action([-4], [3])],
    repeat=2,
    initial_speed=20. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "gas step response: +1m/s^2 from 20mph",
    [Action([1], [3])],
    repeat=2,
    initial_speed=20. * CV.MPH_TO_MS,
  ),
  Maneuver(
    "gas step response: +4m/s^2 from 20mph",
    [Action([4], [3])],
    repeat=2,
    initial_speed=20. * CV.MPH_TO_MS,
  ),
]


def main():
  params = Params()
  cloudlog.info("joystickd is waiting for CarParams")
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)

  sm = messaging.SubMaster(['carState', 'carControl', 'controlsState', 'selfdriveState', 'modelV2'], poll='modelV2')
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance', 'alertDebug'])

  maneuvers = iter(MANEUVERS)
  maneuver = None

  while True:
    sm.update()

    if maneuver is None:
      maneuver = next(maneuvers, None)
      print(f"[DEBUG] Loaded new maneuver: {getattr(maneuver, 'description', None)}")

    alert_msg = messaging.new_message('alertDebug')
    alert_msg.valid = True

    plan_send = messaging.new_message('longitudinalPlan')
    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    accel = 0
    v_ego = max(sm['carState'].vEgo, 0)

    if maneuver is not None:
      print(f"[DEBUG] Running maneuver: {maneuver.description}, active={maneuver.active}, finished={maneuver.finished}")
      accel = maneuver.get_accel(v_ego, sm['carControl'].longActive, sm['carState'].standstill, sm['carState'].cruiseState.standstill)

      if maneuver.active:
        alert_msg.alertDebug.alertText1 = f'Maneuver Active: {accel:0.2f} m/s^2'
      else:
        alert_msg.alertDebug.alertText1 = f'Setting up to {maneuver.initial_speed * CV.MS_TO_MPH:0.2f} mph'
      alert_msg.alertDebug.alertText2 = f'{maneuver.description}'
    else:
      alert_msg.alertDebug.alertText1 = 'Maneuvers Finished'
      print(f"[DEBUG] All maneuvers finished")

    pm.send('alertDebug', alert_msg)

    longitudinalPlan.aTarget = accel
    longitudinalPlan.shouldStop = v_ego < CP.vEgoStopping and accel < 1e-2

    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = True
    longitudinalPlan.hasLead = True

    longitudinalPlan.speeds = [0.2]  # triggers carControl.cruiseControl.resume in controlsd

    pm.send('longitudinalPlan', plan_send)

    assistance_send = messaging.new_message('driverAssistance')
    assistance_send.valid = True
    pm.send('driverAssistance', assistance_send)
    
    if maneuver is not None and maneuver.finished:
      print(f"[DEBUG] Maneuver completed: {maneuver.description}")
      maneuver = None

if __name__ == "__main__":
  main()