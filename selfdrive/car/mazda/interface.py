#!/usr/bin/env python3
from cereal import car
from openpilot.common.conversions import Conversions as CV
from openpilot.selfdrive.car.mazda.values import CAR, LKAS_LIMITS, GEN2, GEN1, MazdaFlags
from openpilot.selfdrive.car import get_safety_config
from openpilot.selfdrive.car.interfaces import CarInterfaceBase, TorqueFromLateralAccelCallbackType, FRICTION_THRESHOLD
from openpilot.selfdrive.controls.lib.drive_helpers import get_friction
from openpilot.selfdrive.global_ti import TI
from panda import Panda
from openpilot.common.params import Params

ButtonType = car.CarState.ButtonEvent.Type
EventName = car.CarEvent.EventName

class CarInterface(CarInterfaceBase):

  @staticmethod
  def _get_params(ret, params, candidate, fingerprint, car_fw, experimental_long, docs):
    ret.carName = "mazda"
    ret.radarUnavailable = True
    if candidate in GEN1:
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.mazda)]
      ret.steerActuatorDelay = 0.1
      if params.get_bool("EnableTI"):
        ret.flags |= MazdaFlags.TORQUE_INTERCEPTOR.value
        ret.safetyConfigs[0].safetyParam |= Panda.FLAG_MAZDA_ENABLE_TI
      if params.get_bool('EnableRI'):
        ret.experimentalLongitudinalAvailable = True
        ret.radarUnavailable = False
        ret.startingState = True
        ret.longitudinalTuning.kpBP = [0., 5., 30.]
        ret.longitudinalTuning.kpV = [1.3, 1.0, 0.7]
        ret.longitudinalTuning.kiBP = [0., 5., 20., 30.]
        ret.longitudinalTuning.kiV = [0.36, 0.23, 0.17, 0.1]
        ret.longitudinalTuning.deadzoneBP = [0.0, 30.0]
        ret.longitudinalTuning.deadzoneV = [0.0, 0.03]
        ret.longitudinalActuatorDelayLowerBound = 0.3
        ret.longitudinalActuatorDelayUpperBound = 1.5
        ret.flags |= MazdaFlags.RADAR_INTERCEPTOR.value
        ret.safetyConfigs[0].safetyParam |= Panda.FLAG_MAZDA_ENABLE_RI
      
    if candidate in GEN2:
      ret.experimentalLongitudinalAvailable = True
      ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.mazda2019)]
      ret.openpilotLongitudinalControl = experimental_long
      ret.stopAccel = -.5
      ret.vEgoStarting = .2
      ret.longitudinalTuning.kpBP = [0., 5., 35.]
      ret.longitudinalTuning.kpV = [0.0, 0.0, 0.0]
      ret.longitudinalTuning.kiBP = [0., 35.]
      ret.longitudinalTuning.kiV = [0.1, 0.1]
      ret.startingState = True
      ret.steerActuatorDelay = 0.1

    ret.dashcamOnly = False
    ret.steerLimitTimer = 0.8
    ret.tireStiffnessFactor = 0.70   # not optimized yet

    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    if candidate in (CAR.CX5, CAR.CX5_2022):
      ret.mass = 3655 * CV.LB_TO_KG
      ret.wheelbase = 2.7
      ret.steerRatio = 15.5
    elif candidate in (CAR.CX9, CAR.CX9_2021):
      ret.mass = 4217 * CV.LB_TO_KG
      ret.wheelbase = 3.1
      ret.steerRatio = 17.6
    elif candidate == CAR.MAZDA3:
      ret.mass = 2875 * CV.LB_TO_KG
      ret.wheelbase = 2.7
      ret.steerRatio = 14.0
    elif candidate == CAR.MAZDA6:
      ret.mass = 3443 * CV.LB_TO_KG
      ret.wheelbase = 2.83
      ret.steerRatio = 15.5
    elif candidate in CAR.MAZDA3_2019:
      ret.mass = 3000 * CV.LB_TO_KG
      ret.wheelbase = 2.725
      ret.steerRatio = 16.5
    elif candidate in (CAR.CX_30, CAR.CX_50):
      ret.mass = 3375 * CV.LB_TO_KG
      ret.wheelbase = 2.814
      ret.steerRatio = 15.5
      ret.steerActuatorDelay = 0.35
    elif candidate in (CAR.CX_60, CAR.CX_80, CAR.CX_70, CAR.CX_90):
      ret.mass = 4217 * CV.LB_TO_KG
      ret.wheelbase = 3.1
      ret.steerRatio = 17.6

    if candidate not in (CAR.CX5_2022, CAR.MAZDA3_2019, CAR.CX_30, CAR.CX_50, CAR.CX_60, CAR.CX_70, CAR.CX_80, CAR.CX_90):
      ret.minSteerSpeed = LKAS_LIMITS.DISABLE_SPEED * CV.KPH_TO_MS

    ret.centerToFront = ret.wheelbase * 0.41

    return ret

  # returns a car.CarState
  def _update(self, c, frogpilot_variables):
    if self.CP.carFingerprint in GEN1:
      if self.CP.enableTorqueInterceptor and not TI.enabled:
        TI.enabled = True
        self.cp_body = self.CS.get_body_can_parser(self.CP)
        self.can_parsers = [self.cp, self.cp_cam, self.cp_adas, self.cp_body, self.cp_loopback]

    ret = self.CS.update(self.cp, self.cp_cam, self.cp_body, frogpilot_variables)

    # events
    events = self.create_common_events(ret, frogpilot_variables)
    if self.CP.carFingerprint in GEN1:
      if self.CS.lkas_disabled:
        events.add(EventName.lkasDisabled)
      elif self.CS.low_speed_alert:
        events.add(EventName.belowSteerSpeed)

      if not self.CS.acc_active_last and not self.CS.ti_lkas_allowed:
        events.add(EventName.steerTempUnavailable)
      if not self.CS.ti_lkas_allowed and TI.enabled:
        events.add(EventName.torqueInterceptorTemporaryWarning)

    ret.events = events.to_msg()

    return ret

  def apply(self, c, now_nanos, frogpilot_variables):
    return self.CC.update(c, self.CS, now_nanos, frogpilot_variables)
