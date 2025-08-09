#!/usr/bin/env python3
import random

from cereal import custom
from openpilot.common.conversions import Conversions as CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import TurnDirection
from openpilot.selfdrive.controls.lib.events import ET, EVENTS, EventName, Events, FROGPILOT_EVENTS
from openpilot.selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY

from openpilot.frogpilot.common.frogpilot_variables import CRUISING_SPEED, NON_DRIVING_GEARS, params_memory

FrogPilotEventName = custom.FrogPilotCarEvent.EventName

DEJA_VU_G_FORCE = 0.75
RANDOM_EVENTS_CHANCE = 0.01 * DT_MDL

class FrogPilotEvents:
  def __init__(self, FrogPilotPlanner, error_log, ThemeManager):
    self.frogpilot_planner = FrogPilotPlanner
    self.theme_manager = ThemeManager

    self.error_log = error_log

    self.events = Events(frogpilot=True)

    self.accel30_played = False
    self.accel35_played = False
    self.accel40_played = False
    self.always_on_lateral_enabled_previously = False
    self.dejaVuCurve_played = False
    self.fcw_played = False
    self.firefoxSteerSaturated_played = False
    self.goatSteerSaturated_played = False
    self.hal9000_played = False
    self.holidayActive_played = False
    self.previous_traffic_mode = False
    self.random_event_playing = False
    self.startup_seen = False
    self.stopped_for_light = False
    self.thisIsFineSteerSaturated_played = False
    self.toBeContinued_played = False
    self.vCruise69_played = False
    self.yourFrogTriedToKillMe_played = False
    self.youveGotMail_played = False

    self.max_acceleration = 0
    self.random_event_timer = 0
    self.tracking_lead_distance = 0

  def update(self, v_cruise, sm, frogpilot_toggles):
    alerts_empty = all(sm[state].alertText1 == "" and sm[state].alertText2 == "" for state in ["controlsState", "frogpilotControlsState"])

    self.events.clear()

    if self.random_event_playing:
      self.random_event_timer += DT_MDL

      if self.random_event_timer >= 5:
        self.theme_manager.update_wheel_image(frogpilot_toggles.wheel_image)

        params_memory.put_bool("UpdateWheelImage", True)

        self.random_event_playing = False

        self.random_event_timer = 0

    if self.error_log.is_file():
      if frogpilot_toggles.random_events:
        self.events.add(FrogPilotEventName.openpilotCrashedRandomEvent)
      else:
        self.events.add(FrogPilotEventName.openpilotCrashed)

    if self.frogpilot_planner.frogpilot_vcruise.forcing_stop:
      self.events.add(FrogPilotEventName.forcingStop)

    if not self.frogpilot_planner.tracking_lead and sm["carState"].standstill and sm["carState"].gearShifter not in NON_DRIVING_GEARS and frogpilot_toggles.green_light_alert:
      if not self.frogpilot_planner.model_stopped and self.stopped_for_light:
        self.events.add(FrogPilotEventName.greenLight)

      self.stopped_for_light = self.frogpilot_planner.cem.stop_light_detected
    else:
      self.stopped_for_light = False

    if not self.holidayActive_played and self.startup_seen and alerts_empty and frogpilot_toggles.current_holiday_theme != "stock" and len(self.events) == 0:
      self.events.add(FrogPilotEventName.holidayActive)

      self.holidayActive_played = True

    if self.frogpilot_planner.tracking_lead and sm["carState"].standstill and sm["carState"].gearShifter not in NON_DRIVING_GEARS and frogpilot_toggles.lead_departing_alert:
      if self.tracking_lead_distance == 0:
        self.tracking_lead_distance = self.frogpilot_planner.lead_one.dRel

      lead_departing = self.frogpilot_planner.lead_one.dRel - self.tracking_lead_distance > 1
      lead_departing &= self.frogpilot_planner.lead_one.vLead > 1

      if lead_departing:
        self.events.add(FrogPilotEventName.leadDeparting)
    else:
      self.tracking_lead_distance = 0

    if not self.random_event_playing and frogpilot_toggles.random_events:
      acceleration = sm["carState"].aEgo

      if not sm["carState"].gasPressed:
        self.max_acceleration = max(acceleration, self.max_acceleration)
      else:
        self.max_acceleration = 0

      if not self.accel30_played and 3.5 > self.max_acceleration >= 3.0 and acceleration < 1.5:
        self.events.add(FrogPilotEventName.accel30)

        self.theme_manager.update_wheel_image("weeb_wheel", random_event=True)

        params_memory.put_bool("UpdateWheelImage", True)

        self.accel30_played = True
        self.random_event_playing = True

        self.max_acceleration = 0

      elif not self.accel35_played and 4.0 > self.max_acceleration >= 3.5 and acceleration < 1.5:
        self.events.add(FrogPilotEventName.accel35)

        self.theme_manager.update_wheel_image("tree_fiddy", random_event=True)

        params_memory.put_bool("UpdateWheelImage", True)

        self.accel35_played = True
        self.random_event_playing = True

        self.max_acceleration = 0

      elif not self.accel40_played and self.max_acceleration >= 4.0 and acceleration < 1.5:
        self.events.add(FrogPilotEventName.accel40)

        self.theme_manager.update_wheel_image("great_scott", random_event=True)

        params_memory.put_bool("UpdateWheelImage", True)

        self.accel40_played = True
        self.random_event_playing = True

        self.max_acceleration = 0

      if not self.dejaVuCurve_played and sm["carState"].vEgo > CRUISING_SPEED:
        if self.frogpilot_planner.lateral_acceleration >= DEJA_VU_G_FORCE * ACCELERATION_DUE_TO_GRAVITY:
          self.events.add(FrogPilotEventName.dejaVuCurve)

          self.dejaVuCurve_played = True
          self.random_event_playing = True

      if not self.hal9000_played and sm["controlsState"].alertType == ET.NO_ENTRY:
        self.events.add(FrogPilotEventName.hal9000)

        self.hal9000_played = True
        self.random_event_playing = True

      saturated_events = [
        ("controlsState", [EventName.steerSaturated], EVENTS),
        ("frogpilotControlsState", [FrogPilotEventName.goatSteerSaturated], FROGPILOT_EVENTS)
      ]
      saturated_alert_match = any(sm[state].alertText2 == events[event][ET.WARNING].alert_text_2 for state, event_group, events in saturated_events for event in event_group)
      if saturated_alert_match:
        event_choices = []
        if not self.firefoxSteerSaturated_played:
          event_choices.append("firefoxSteerSaturated")
        if not self.goatSteerSaturated_played:
          event_choices.append("goatSteerSaturated")
        if not self.thisIsFineSteerSaturated_played:
          event_choices.append("thisIsFineSteerSaturated")

        if event_choices and random.random() < RANDOM_EVENTS_CHANCE:
          event_choice = random.choice(event_choices)

          if event_choice == "firefoxSteerSaturated":
            self.events.add(FrogPilotEventName.firefoxSteerSaturated)

            self.theme_manager.update_wheel_image("firefox", random_event=True)

            params_memory.put_bool("UpdateWheelImage", True)

            self.firefoxSteerSaturated_played = True

          elif event_choice == "goatSteerSaturated":
            self.events.add(FrogPilotEventName.goatSteerSaturated)

            self.theme_manager.update_wheel_image("goat", random_event=True)

            params_memory.put_bool("UpdateWheelImage", True)

            self.goatSteerSaturated_played = True

          elif event_choice == "thisIsFineSteerSaturated":
            self.events.add(FrogPilotEventName.thisIsFineSteerSaturated)

            self.theme_manager.update_wheel_image("this_is_fine", random_event=True)

            params_memory.put_bool("UpdateWheelImage", True)

            self.thisIsFineSteerSaturated_played = True

          self.random_event_playing = True

      if not self.vCruise69_played and 70 > max(sm["controlsState"].vCruise, sm["controlsState"].vCruiseCluster) * (1 if frogpilot_toggles.is_metric else CV.KPH_TO_MPH) >= 69:
        self.events.add(FrogPilotEventName.vCruise69)

        self.vCruise69_played = True
        self.random_event_playing = True

      fcw_alert_match = sm["controlsState"].alertText1 == EVENTS[EventName.fcw][ET.PERMANENT].alert_text_1 and sm["controlsState"].alertText2 == EVENTS[EventName.fcw][ET.PERMANENT].alert_text_2
      stock_aeb_alert_match = sm["controlsState"].alertText1 == EVENTS[EventName.stockAeb][ET.PERMANENT].alert_text_1 and sm["controlsState"].alertText2 == EVENTS[EventName.stockAeb][ET.PERMANENT].alert_text_2
      if fcw_alert_match or stock_aeb_alert_match:
        event_choices = []
        if not self.toBeContinued_played:
          event_choices.append("toBeContinued")
        if not self.yourFrogTriedToKillMe_played:
          event_choices.append("yourFrogTriedToKillMe")

        event_choice = random.choice(event_choices)
        if event_choice == "toBeContinued":
          self.events.add(FrogPilotEventName.toBeContinued)

          self.toBeContinued_played = True

        elif event_choice == "yourFrogTriedToKillMe":
          self.events.add(FrogPilotEventName.yourFrogTriedToKillMe)

          self.yourFrogTriedToKillMe_played = True

        self.random_event_playing = True

      if not self.youveGotMail_played and sm["frogpilotCarState"].alwaysOnLateralEnabled and not self.always_on_lateral_enabled_previously:
        if random.random() < RANDOM_EVENTS_CHANCE:
          self.events.add(FrogPilotEventName.youveGotMail)

          self.youveGotMail_played = True
          self.random_event_playing = True

      self.always_on_lateral_enabled_previously = sm["frogpilotCarState"].alwaysOnLateralEnabled

    if frogpilot_toggles.speed_limit_changed_alert and self.frogpilot_planner.frogpilot_vcruise.slc.speed_limit_changed_timer == DT_MDL:
      self.events.add(FrogPilotEventName.speedLimitChanged)

    self.startup_seen |= sm["frogpilotControlsState"].alertText1 == frogpilot_toggles.startup_alert_top and sm["frogpilotControlsState"].alertText2 == frogpilot_toggles.startup_alert_bottom

    if sm["frogpilotCarState"].trafficModeEnabled != self.previous_traffic_mode:
      if self.previous_traffic_mode:
        self.events.add(FrogPilotEventName.trafficModeInactive)
      else:
        self.events.add(FrogPilotEventName.trafficModeActive)
      self.previous_traffic_mode = sm["frogpilotCarState"].trafficModeEnabled

    if sm["frogpilotModelV2"].turnDirection == TurnDirection.turnLeft:
      self.events.add(FrogPilotEventName.turningLeft)
    elif sm["frogpilotModelV2"].turnDirection == TurnDirection.turnRight:
      self.events.add(FrogPilotEventName.turningRight)
