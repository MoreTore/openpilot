#!/usr/bin/env python3
import math
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL

from openpilot.frogpilot.common.frogpilot_variables import CITY_SPEED_LIMIT, CRUISING_SPEED, THRESHOLD, params_memory

class ConditionalExperimentalMode:
  def __init__(self, FrogPilotPlanner):
    self.frogpilot_planner = FrogPilotPlanner

    self.curvature_filter = FirstOrderFilter(0, 1, DT_MDL)
    self.slow_lead_filter = FirstOrderFilter(0, 1, DT_MDL)
    self.stop_light_filter = FirstOrderFilter(0, 0.5, DT_MDL)

    self.curve_detected = False
    self.experimental_mode = False
    self.stop_light_detected = False

  def update(self, v_ego, sm, frogpilot_toggles):
    if frogpilot_toggles.experimental_mode_via_press:
      self.status_value = params_memory.get_int("CEStatus")
    else:
      self.status_value = 0

    if self.status_value not in (1, 2):
      self.update_conditions(v_ego, sm, frogpilot_toggles)

      self.experimental_mode = self.check_conditions(v_ego, sm, frogpilot_toggles)

      params_memory.put_int("CEStatus", self.status_value if self.experimental_mode else 0)
    else:
      self.experimental_mode = self.status_value == 2 or sm["carState"].standstill and self.experimental_mode and self.frogpilot_planner.model_stopped
      self.stop_light_detected &= self.status_value not in (1, 2)
      self.stop_light_filter.x = 0

  def check_conditions(self, v_ego, sm, frogpilot_toggles):
    below_speed = not self.frogpilot_planner.frogpilot_following.following_lead and v_ego < frogpilot_toggles.conditional_limit
    below_speed_with_lead = self.frogpilot_planner.frogpilot_following.following_lead and v_ego < frogpilot_toggles.conditional_limit_lead
    if below_speed or below_speed_with_lead:
      self.status_value = 3 if self.frogpilot_planner.frogpilot_following.following_lead else 4
      return True

    desired_lane = self.frogpilot_planner.lane_width_left if sm["carState"].leftBlinker else self.frogpilot_planner.lane_width_right
    lane_available = desired_lane >= frogpilot_toggles.lane_detection_width or not frogpilot_toggles.conditional_signal_lane_detection
    if v_ego < frogpilot_toggles.conditional_signal and (sm["carState"].leftBlinker or sm["carState"].rightBlinker) and not lane_available:
      self.status_value = 5
      return True

    approaching_maneuver = sm["frogpilotNavigation"].approachingIntersection or sm["frogpilotNavigation"].approachingTurn
    if approaching_maneuver and (not self.frogpilot_planner.frogpilot_following.following_lead or frogpilot_toggles.conditional_navigation_lead) and frogpilot_toggles.conditional_navigation:
      self.status_value = 6 if sm["frogpilotNavigation"].approachingIntersection else 7
      return True

    if self.curve_detected and (not self.frogpilot_planner.frogpilot_following.following_lead or frogpilot_toggles.conditional_curves_lead) and frogpilot_toggles.conditional_curves:
      self.status_value = 8
      return True

    if self.slow_lead_detected and frogpilot_toggles.conditional_lead:
      self.status_value = 9 if self.frogpilot_planner.lead_one.vLead < 1 else 10
      return True

    if self.stop_light_detected and frogpilot_toggles.conditional_model_stop_time != 0:
      self.status_value = 11 if not self.frogpilot_planner.frogpilot_vcruise.forcing_stop else 12
      return True

    if self.frogpilot_planner.frogpilot_vcruise.slc.experimental_mode:
      self.status_value = 13
      return True

    return False

  def update_conditions(self, v_ego, sm, frogpilot_toggles):
    self.curve_detection(v_ego, frogpilot_toggles)
    self.slow_lead(v_ego, frogpilot_toggles)
    self.stop_sign_and_light(v_ego, sm, frogpilot_toggles.conditional_model_stop_time)

  def curve_detection(self, v_ego, frogpilot_toggles):
    self.curvature_filter.update(self.frogpilot_planner.road_curvature_detected or self.frogpilot_planner.driving_in_curve)
    self.curve_detected = self.curvature_filter.x >= THRESHOLD and v_ego > CRUISING_SPEED

  def slow_lead(self, v_ego, frogpilot_toggles):
    if self.frogpilot_planner.tracking_lead:
      lead = self.frogpilot_planner.lead_one
      lead_distance = lead.dRel
      relative_speed = v_ego - lead.vLead

      # How long until we crash into them?
      if relative_speed > 0:
        time_to_impact = lead_distance / relative_speed
      else:
        time_to_impact = 1000.0 # We dont care lets set a value we dont care about later

      # Stopped leads get the GUI stop sign toggle, or 6 sec default. Moving leads get the hardcoded 4 second rate.
      stop_time_threshold = self.get_safe_stop_time(frogpilot_toggles.conditional_model_stop_time)
      drive_time_threshold = 4.0

      # Check if they are stopped (or < 4.5 mph)
      is_stopped = lead.vLead < 2.0

      if is_stopped:
        # Will we hit them within N seconds (GUI stop sign toggle, or 6 sec default)
        lead_detected = (time_to_impact < stop_time_threshold) and frogpilot_toggles.conditional_stopped_lead
      else:
        # We will hit them within 4 seconds, and we are going at least 11mph faster than them
        closing_fast = relative_speed > CRUISING_SPEED
        lead_detected = closing_fast and (time_to_impact < drive_time_threshold) and frogpilot_toggles.conditional_slower_lead

      self.slow_lead_filter.update(lead_detected)
      self.slow_lead_detected = self.slow_lead_filter.x >= THRESHOLD

    else:
      self.slow_lead_filter.x = 0
      self.slow_lead_detected = False

  def stop_sign_and_light(self, v_ego, sm, model_time):
    if not sm["frogpilotCarState"].trafficModeEnabled:
      model_stopping = self.frogpilot_planner.model_length < v_ego * model_time

      self.stop_light_filter.update(self.frogpilot_planner.model_stopped or model_stopping)

      light_detected = self.stop_light_filter.x >= THRESHOLD

      lead_ignored = False
      if self.frogpilot_planner.tracking_lead:
          lead = self.frogpilot_planner.lead_one

          relative_speed = v_ego - lead.vLead
          # Find how long until we hit the lead
          time_to_impact_lead = lead.dRel / max(relative_speed, 0.1)

          # Find how long until we hit the stop line
          time_to_reach_light = self.frogpilot_planner.model_length / max(v_ego, 0.1)

          # If we hit the line before the lead, ignore the lead
          # so we dont follow a red light runner thru the intersection, or a right turn from the road to our right
          if time_to_reach_light < time_to_impact_lead:
              lead_ignored = True

          # if they are going faster than us leading up to the intersection, let frog deal with it
          if lead.vLead > v_ego:
              lead_ignored = True

          # stopped lead check (just in case the toggle is disabled)
          if lead.vLead < 2.0:
              lead_ignored = True

      # Stop for the stop sign if we don't like our lead car (or if we dont have one)
      self.stop_light_detected = light_detected and (not self.frogpilot_planner.tracking_lead or lead_ignored)

    else:
      self.stop_light_filter.x = 0
      self.stop_light_detected = False

  def get_safe_stop_time(self, raw_value):
    """
    Stop time for stopped leads and red light/stop sign. Return 6 seconds if we have a shit config. (So we dont rear-end someone)
    """
    fallback_value = 6

    try:
        # Null check
        if raw_value is None:
            return fallback_value

        # Float it
        val = float(raw_value)

        # NaN inf check
        if math.isnan(val) or math.isinf(val):
            return fallback_value

        # Guard against template val
        if val <= 1.5:
            return fallback_value

        # Return the valid number
        return val

    except (ValueError, TypeError):
        # We broke something, use fallback
        return fallback_value