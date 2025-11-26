import numpy as np
from cereal import car
from opendbc.can.packer import CANPacker
from openpilot.selfdrive.car import apply_driver_steer_torque_limits, apply_ti_steer_torque_limits
from openpilot.selfdrive.car.interfaces import CarControllerBase
from openpilot.selfdrive.car.mazda import mazdacan
from openpilot.selfdrive.car.mazda.values import CarControllerParams, Buttons, MazdaFlags
from openpilot.common.realtime import ControlsTimer as Timer, DT_CTRL
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params

import cereal.messaging as messaging

VisualAlert = car.CarControl.HUDControl.VisualAlert
LongCtrlState = car.CarControl.Actuators.LongControlState


class CarController(CarControllerBase):
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.apply_steer_last = 0
    self.ti_apply_steer_last = 0
    self.packer = CANPacker(dbc_name)
    self.brake_counter = 0
    self.frame = 0
    self.ccp = CarControllerParams(CP)
    self.hold_timer = Timer(6.0)
    self.hold_delay = Timer(.5) # delay before we start holding as to not hit the brakes too hard
    self.resume_timer = Timer(0.5)
    self.cancel_delay = Timer(0.07) # 70ms delay to try to avoid a race condition with stock system
    self.acc_filter = FirstOrderFilter(0.0, .1, DT_CTRL, initialized=False)
    self.filtered_acc_last = 0
    self.params = Params()
    self.params_memory = Params("/dev/shm/params")
    self.blend_coeff = 0 #factor for blending OP and stock long. 0 is fully stock, 1 is fully OP
    self.transition_time = 2.5 #After this number of seconds, the smooth blending from stock to OP (or vice versa) is complete
    self.distance_last = None
    self.sm = messaging.SubMaster(['longitudinalPlan', 'radarState'])
    # self.accel_transition_thresh = 1.25 #m/s^2, if aEgo is less than this we want to use MRCC, if more than this we transition to OP long


  def update(self, CC, CS, now_nanos, frogpilot_toggles):
    self.sm.update(0)
    long_plan = self.sm['longitudinalPlan']
    allow_throttle = long_plan.allowThrottle

    lead_one = self.sm['radarState'].leadOne
    lead_status = lead_one.status  # whether lead is valid
    if lead_status:
      lead_distance = lead_one.dRel  # relative distance in meters
      lead_velocity = lead_one.vRel  # relative velocity in m/s
    else:
      lead_distance = None
      lead_velocity = None


    can_sends = []

    apply_steer = 0
    ti_apply_steer = 0

    if CC.latActive:
      # calculate steer and also set limits due to driver torque
      new_steer = int(round(CC.actuators.steer * self.ccp.STEER_MAX))
      apply_steer = apply_driver_steer_torque_limits(new_steer, self.apply_steer_last,
                                                     CS.out.steeringTorque, self.ccp)
      if self.CP.flags & MazdaFlags.TORQUE_INTERCEPTOR:
        if CS.ti_lkas_allowed:
          ti_new_steer = int(round(CC.actuators.steer * self.ccp.TI_STEER_MAX))
          ti_apply_steer = apply_ti_steer_torque_limits(ti_new_steer, self.ti_apply_steer_last,
                                                    CS.out.steeringTorque, self.ccp)
    self.apply_steer_last = apply_steer
    self.ti_apply_steer_last = ti_apply_steer

    if self.CP.flags & MazdaFlags.GEN1:
      if CC.cruiseControl.cancel:
        # If brake is pressed, let us wait >70ms before trying to disable crz to avoid
        # a race condition with the stock system, where the second cancel from openpilot
        # will disable the crz 'main on'. crz ctrl msg runs at 50hz. 70ms allows us to
        # read 3 messages and most likely sync state before we attempt cancel.
        self.brake_counter = self.brake_counter + 1
        if self.frame % 10 == 0 and not (CS.out.brakePressed and self.brake_counter < 7):
          # Cancel Stock ACC if it's enabled while OP is disengaged
          # Send at a rate of 10hz until we sync with stock ACC state
          can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP, CS.crz_btns_counter, Buttons.CANCEL))
      else:
        self.brake_counter = 0
        if CC.cruiseControl.resume and self.frame % 5 == 0:
          # Mazda Stop and Go requires a RES button (or gas) press if the car stops more than 3 seconds
          # Send Resume button when planner wants car to move
          can_sends.append(mazdacan.create_button_cmd(self.packer, self.CP, CS.crz_btns_counter, Buttons.RESUME))

      # send HUD alerts
      if self.frame % 50 == 0:
        ldw = CC.hudControl.visualAlert == VisualAlert.ldw
        steer_required = CC.hudControl.visualAlert == VisualAlert.steerRequired
        # TODO: find a way to silence audible warnings so we can add more hud alerts
        steer_required = steer_required and CS.lkas_allowed_speed
        can_sends.append(mazdacan.create_alert_command(self.packer, CS.cam_laneinfo, ldw, steer_required))

      if self.CP.flags & MazdaFlags.RADAR_INTERCEPTOR:
        hold = False
        if CS.out.standstill:
          hold = self.hold_timer.active()
        else:
          self.hold_timer.reset()

        if CC.longActive:
          raw_acc_output = CC.actuators.accel * 1150
          raw_acc_output = max(-1000, min(raw_acc_output, 1000))

          if self.params.get_bool("BlendedACC"):
            if self.params_memory.get_int("CEStatus"):
              self.acc_filter.update_alpha(abs(raw_acc_output-self.filtered_acc_last)/1000)
              filtered_acc_output = int(self.acc_filter.update(raw_acc_output))
            else:
              # we want to use the stock value in this case but we need a smooth transition.
              self.acc_filter.update_alpha(abs(CS.crz_info["ACCEL_CMD"]-self.filtered_acc_last)/1000)
              filtered_acc_output = int(self.acc_filter.update(CS.crz_info["ACCEL_CMD"]))

            CS.crz_info["ACCEL_CMD"] = int(filtered_acc_output)
            self.filtered_acc_last = filtered_acc_output
          else:
            acc_output = raw_acc_output

        if self.frame % 2 == 0:
          can_sends.extend(mazdacan.create_radar_command(self.packer, self.frame, CC.longActive, CS, hold))

    else:
      raw_acc_output = (CC.actuators.accel * 200) + 2000
      OPlong = (self.params.get_bool("ExperimentalLongitudinalEnabled") and CC.longActive)

      # if self.params.get_bool("BlendedACC"):
        # if self.params_memory.get_int("CEStatus"):
          # self.acc_filter.update_alpha(abs(raw_acc_output-self.filtered_acc_last)/1000)
          # filtered_acc_output = int(self.acc_filter.update(raw_acc_output))
          # if OPlong:
            # CS.acc["ACCEL_CMD"] = raw_acc_output
        # else:
          # we want to use the stock value in this case but we need a smooth transition.
          # self.acc_filter.update_alpha(abs(CS.acc["ACCEL_CMD"]-self.filtered_acc_last)/1000)
          # filtered_acc_output = CS.acc["ACCEL_CMD"]

        # self.filtered_acc_last = filtered_acc_output
      # elif OPlong:
        # CS.acc["ACCEL_CMD"] = raw_acc_output

      if OPlong:
        #Force CEM with distance setting
        if CS.distance_setting == 1:
          self.params_memory.put_int("CEStatus", 2)
        elif self.distance_last == 1:
          self.params_memory.put_int("CEStatus", 0)

        #Force CEM more aggressively when approaching leads
        if lead_status and (lead_distance < CS.out.vEgo or lead_velocity < -2):
          self.params_memory.put_int("CEStatus", 2)
        elif CS.distance_setting != 1:
          self.params_memory.put_int("CEStatus", 0)



        if self.params.get_bool("BlendedACC"):
          blended_acc_output = (self.blend_coeff * raw_acc_output) + ((1 - self.blend_coeff) * CS.acc["ACCEL_CMD"])
          CEStatus = self.params_memory.get_int("CEStatus")

          # gas_gate_thresh = np.interp(CS.out.vEgo, [0,1,5,15,25], [0,500,250,20,0])
          #If OP is gas gating, we'll allow it to take over control of long from MRCC. But only if MRCC commands are within this range.
          #This is mainly to prevent the car from drifting away from the lead at highway speeds.

          #blend in OP long
          if (CEStatus and self.blend_coeff < 1):# or (allow_throttle == False and (2000 - gas_gate_thresh) < CS.acc["ACCEL_CMD"] < (2000 + gas_gate_thresh)):
            self.blend_coeff += min((DT_CTRL / self.transition_time), (1 - self.blend_coeff))

          #blend out to MRCC
          elif CEStatus < 2 and self.blend_coeff > 0: #CEStatus == 1 is when CEM is forced off, but we still want to be decrementing in that scenario
            self.blend_coeff -= min((DT_CTRL / self.transition_time), self.blend_coeff)
            # self.accel_transition_thresh = 1.25

          if self.blend_coeff > 0:
            CS.acc["ACCEL_CMD"] = blended_acc_output


          self.transition_time = (0.045455 * CS.out.vEgo) + 0.5 #ramp transition time depending on vehicle speed. 0.5s at standstill, 3s at 55mph
          self.distance_last = CS.distance_setting

        else:
          # blended_acc_output = (self.blend_coeff * raw_acc_output) + ((1 - self.blend_coeff) * CS.acc["ACCEL_CMD"])

          # #Blend in MRCC when gas gating is active to disable it. Remove this section when gas gating gets better.
          # if allow_throttle or CC.actuators.accel < -3:
          #   self.blend_coeff += min((DT_CTRL / self.transition_time), (1 - self.blend_coeff))

          # else: #gas gating is active, so transition back to MRCC
          #   self.blend_coeff -= min((DT_CTRL / self.transition_time), self.blend_coeff)

          # if self.blend_coeff > 0:
          #   CS.acc["ACCEL_CMD"] = blended_acc_output

          # self.transition_time = (0.045455 * CS.out.vEgo) + 0.5 #ramp transition time depending on vehicle speed. 0.5s at standstill, 3s at 55mph

          CS.acc["ACCEL_CMD"] = raw_acc_output



      resume = False
      hold = False
      if Timer.interval(2): # send ACC command at 50hz
        """
        Without this hold/resum logic, the car will only stop momentarily.
        It will then start creeping forward again. This logic allows the car to
        apply the electric brake to hold the car. The hold delay also fixes a
        bug with the stock ACC where it sometimes will apply the brakes too early
        when coming to a stop.
        """
        if CS.out.standstill: # if we're stopped
          if not self.hold_delay.active(): # and we have been stopped for more than hold_delay duration. This prevents a hard brake if we aren't fully stopped.
            if ((CC.cruiseControl.resume and CC.actuators.longControlState != LongCtrlState.stopping) or
                CC.cruiseControl.override or CS.out.gasPressed or
                (CC.actuators.longControlState == LongCtrlState.starting) or CS.acc["RESUME"]): # if we are resuming or overriding, we want to release the brake
              self.resume_timer.reset() # reset the resume timer so its active
            else: # otherwise we're holding
              hold = self.hold_timer.active() # hold for 6s. This allows the electric brake to hold the car.

        else: # if we're moving
          self.hold_timer.reset() # reset the hold timer so its active when we stop
          self.hold_delay.reset() # reset the hold delay

        resume = self.resume_timer.active() # stay on for 0.5s to release the brake. This allows the car to move.
        can_sends.append(mazdacan.create_acc_cmd(self, self.packer, CS.acc, hold, resume))

    # send steering command
    can_sends.extend(mazdacan.create_steering_control(self.packer, self.CP,
                                                      self.frame, apply_steer, CS.cam_lkas))

    new_actuators = CC.actuators.as_builder()
    new_actuators.steer = apply_steer / self.ccp.STEER_MAX
    new_actuators.steerOutputCan = apply_steer

    self.frame += 1
    Timer.tick()
    return new_actuators, can_sends