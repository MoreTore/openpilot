#!/usr/bin/env python3
from math import exp, fabs
import numpy as np

from cereal import car, custom
from panda import Panda
from openpilot.common.conversions import Conversions as CV
from openpilot.selfdrive.car.mazda.values import CAR, LKAS_LIMITS, MazdaFlags, GEN1, GEN2
from openpilot.selfdrive.car import create_button_events, get_safety_config
from openpilot.selfdrive.car.interfaces import CarInterfaceBase, TorqueFromLateralAccelCallbackType, LateralAccelFromTorqueCallbackType
from openpilot.common.params import Params

ButtonType = car.CarState.ButtonEvent.Type
FrogPilotButtonType = custom.FrogPilotCarState.ButtonEvent.Type
EventName = car.CarEvent.EventName

NON_LINEAR_TORQUE_PARAMS = {
 #CAR.MAZDA_3_2019: (7.77957, 0.62027, 0.16634, 0.31350),
 #CAR.MAZDA_3_2019: (7.77900, 0.62029, 0.16633, 0.31350),
 #CAR.MAZDA_3_2019: (8.47125, 0.59462, 0.17099, 0.29543),
 #CAR.MAZDA_3_2019: (11.30242, 0.62977, 0.15728, 0.32293),
 #CAR.MAZDA_3_2019: (16.76508, 0.71466, 0.15496, 0.39772), # this is going to seriously suck
 #CAR.MAZDA_3_2019: (9, 0.6, 0.2, 0.32293),
 #CAR.MAZDA_3_2019: (14.13208, 0.71564, 0.15621, 0.39431), #i cant wait for this to fucking slam the wheel to the right and kill me
 #CAR.MAZDA_3_2019: (13.43336, 0.73874, 0.14520, 0.40059), #maybe a little less violent?
 #CAR.MAZDA_3_2019: (14.30033, 0.73571, 0.14462, 0.39894), #is this not going to be truly horrible? what is the algorithm even doing
 #CAR.MAZDA_3_2019: (15.11403, 0.72348, 0.15010, 0.37949),
  CAR.MAZDA_3_2019: (15.38616, 0.71899, 0.15015, 0.37999), #i think i have hit the point of being negligable
  CAR.MAZDA_CX_30: (4.68689, 0.79999, 0.18244, 0.38763),
  CAR.MAZDA_CX_50: (4.68689, 0.79999, 0.18244, 0.38763)
}

class CarInterface(CarInterfaceBase):

  def get_lataccel_torque_siglin(self) -> float:

    def torque_from_lateral_accel_siglin_func(lateral_acceleration: float) -> float:
      # The "lat_accel vs torque" relationship is assumed to be the sum of "sigmoid + linear" curves
      # An important thing to consider is that the slope at 0 should be > 0 (ideally >1)
      # This has big effect on the stability about 0 (noise when going straight)
      non_linear_torque_params = NON_LINEAR_TORQUE_PARAMS.get(self.CP.carFingerprint)
      assert non_linear_torque_params, "The params are not defined"
      a, b, c, _ = non_linear_torque_params
      sig_input = a * lateral_acceleration
      sig = np.sign(sig_input) * (1 / (1 + exp(-fabs(sig_input))) - 0.5)
      steer_torque = (sig * b) + (lateral_acceleration * c)
      return float(steer_torque)

    lataccel_values = np.arange(-8.0, 8.0, 0.01)
    torque_values = [torque_from_lateral_accel_siglin_func(x) for x in lataccel_values]
    assert min(torque_values) < -1 and max(torque_values) > 1, "The torque values should cover the range [-1, 1]"
    return torque_values, lataccel_values

  def torque_from_lateral_accel(self) -> TorqueFromLateralAccelCallbackType:
    if self.CP.carFingerprint in NON_LINEAR_TORQUE_PARAMS:
      torque_values, lataccel_values = self.get_lataccel_torque_siglin()

      def torque_from_lateral_accel_siglin(lateral_acceleration: float, torque_params: car.CarParams.LateralTorqueTuning):
        return np.interp(lateral_acceleration, lataccel_values, torque_values)
      return torque_from_lateral_accel_siglin
    else:
      return self.torque_from_lateral_accel_linear

  def lateral_accel_from_torque(self) -> LateralAccelFromTorqueCallbackType:
    if self.CP.carFingerprint in NON_LINEAR_TORQUE_PARAMS:
      torque_values, lataccel_values = self.get_lataccel_torque_siglin()

      def lateral_accel_from_torque_siglin(torque: float, torque_params: car.CarParams.LateralTorqueTuning):
        return np.interp(torque, torque_values, lataccel_values)
      return lateral_accel_from_torque_siglin
    else:
      return self.lateral_accel_from_torque_linear

  @staticmethod
  def _get_params(ret, candidate, fingerprint, car_fw, experimental_long, docs, frogpilot_toggles):
    ret.carName = "mazda"
    ret.safetyConfigs = [get_safety_config(car.CarParams.SafetyModel.mazda)]
    ret.radarUnavailable = True
    ret.dashcamOnly = False
    ret.openpilotLongitudinalControl = True
    p = Params()
    if p.get_bool("ManualTransmission"):
      ret.flags |= MazdaFlags.MANUAL_TRANSMISSION.value
      ret.transmissionType = car.CarParams.TransmissionType.manual
    else:
      ret.transmissionType = car.CarParams.TransmissionType.automatic

    if candidate in GEN1:
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_MAZDA_GEN1
      if p.get_bool("TorqueInterceptorEnabled"): # Torque Interceptor Installed
        print("Torque Interceptor Installed")
        ret.flags |= MazdaFlags.TORQUE_INTERCEPTOR.value
        ret.safetyConfigs[0].safetyParam |= Panda.FLAG_MAZDA_TORQUE_INTERCEPTOR
      if p.get_bool("RadarInterceptorEnabled"): # Radar Interceptor Installed
        ret.flags |= MazdaFlags.RADAR_INTERCEPTOR.value
        ret.experimentalLongitudinalAvailable = True
        ret.radarUnavailable = False
        ret.startingState = True
        ret.longitudinalTuning.kpBP = [0., 5., 30.]
        ret.longitudinalTuning.kpV = [1.3, 1.0, 0.7]
        ret.longitudinalTuning.kiBP = [0., 5., 20., 30.]
        ret.longitudinalTuning.kiV = [0.36, 0.23, 0.17, 0.1]
        ret.safetyConfigs[0].safetyParam |= Panda.FLAG_MAZDA_RADAR_INTERCEPTOR

      if p.get_bool("NoMRCC"): # No Mazda Radar Cruise Control; Missing CRZ_CTRL signal
        ret.flags |= MazdaFlags.NO_MRCC.value
        ret.safetyConfigs[0].safetyParam |= Panda.FLAG_MAZDA_NO_MRCC
      if p.get_bool("NoFSC"):  # No Front Sensing Camera
        ret.flags |= MazdaFlags.NO_FSC.value
        ret.safetyConfigs[0].safetyParam |= Panda.FLAG_MAZDA_NO_FSC

      ret.steerActuatorDelay = 0.1
      ret.enableBsm = True

    if candidate in GEN2:
      ret.safetyConfigs[0].safetyParam |= Panda.FLAG_MAZDA_GEN2
      ret.experimentalLongitudinalAvailable = True
      ret.stopAccel = -.5
      ret.vEgoStarting = .2
      ret.longitudinalActuatorDelay = 0.35 # gas is 0.25s and brake looks like 0.5
      ret.longitudinalTuning.kpBP = [0., 5., 35.]
      ret.longitudinalTuning.kpV = [0.0, 0.0, 0.0]
      ret.longitudinalTuning.kiBP = [0., 35.]
      ret.longitudinalTuning.kiV = [0.1, 0.1]
      ret.startingState = True
      ret.steerActuatorDelay = 0.335

    ret.steerLimitTimer = 0.8

    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    if candidate not in (CAR.MAZDA_CX5_2022, CAR.MAZDA_3_2019, CAR.MAZDA_CX_30, CAR.MAZDA_CX_50) and not ret.flags & MazdaFlags.TORQUE_INTERCEPTOR:
      ret.minSteerSpeed = LKAS_LIMITS.DISABLE_SPEED * CV.KPH_TO_MS

    ret.centerToFront = ret.wheelbase * 0.41

    return ret

  # returns a car.CarState
  def _update(self, c, frogpilot_toggles):
    ret, fp_ret = self.CS.update(self.cp, self.cp_cam, self.cp_body, frogpilot_toggles)
     # TODO: add button types for inc and dec
    ret.buttonEvents = [
      *create_button_events(self.CS.distance_button, self.CS.prev_distance_button, {1: ButtonType.gapAdjustCruise}),
      *create_button_events(self.CS.lkas_enabled, self.CS.lkas_previously_enabled, {1: FrogPilotButtonType.lkas}),
    ]

    # events
    events = self.create_common_events(ret)

    if self.CP.flags & MazdaFlags.GEN1:
      if self.CS.lkas_disabled:
        events.add(EventName.lkasDisabled)
      elif self.CS.low_speed_alert:
        events.add(EventName.belowSteerSpeed)

      if not self.CS.acc_active_last and not self.CS.ti_lkas_allowed:
        events.add(EventName.steerTempUnavailable)
      #if (not self.CS.ti_lkas_allowed) and (self.CP.flags & MazdaFlags.TORQUE_INTERCEPTOR):
      #  events.add(EventName.steerTempUnavailable) # torqueInterceptorTemporaryWarning

    ret.events = events.to_msg()

    return ret, fp_ret
