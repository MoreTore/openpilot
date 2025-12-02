#!/usr/bin/env python3
import math
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.interfaces import ACCEL_MIN
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

      # 4 seconds (or however hard we can brake to get to 4 seconds)
      safe_approach_dist = self.get_safe_distance(relative_speed, 4.0)

      closing_quickly = relative_speed > CRUISING_SPEED # are we 11mph faster than them?
      close_proximity = lead_distance < safe_approach_dist

      slower_lead = closing_quickly and close_proximity and frogpilot_toggles.conditional_slower_lead


      # try to stop within N seconds to make the lead car. if we cant brake that hard, start braking sooner
      safe_stopped_dist = self.get_safe_distance(relative_speed, self.get_safe_stop_time(frogpilot_toggles.conditional_model_stop_time))

      # Is the lead stopped? Will we make it within 9 sec? Do we need to brake early?
      lead_is_stopped = lead.vLead < 1
      lead_is_in_range = lead_distance < safe_stopped_dist

      stopped_lead = lead_is_stopped and lead_is_in_range and frogpilot_toggles.conditional_stopped_lead


      self.slow_lead_filter.update(slower_lead or stopped_lead)
      self.slow_lead_detected = self.slow_lead_filter.x >= THRESHOLD
    else:
      self.slow_lead_filter.x = 0
      self.slow_lead_detected = False

  def stop_sign_and_light(self, v_ego, sm, model_time):
    if not sm["frogpilotCarState"].trafficModeEnabled:
      model_length = self.frogpilot_planner.model_length

      # can we make it? stop earlier if needed
      safe_stop_dist = self.get_safe_distance(v_ego, self.get_safe_stop_time(model_time))

      model_stopping = model_length < safe_stop_dist

      self.stop_light_filter.update(self.frogpilot_planner.model_stopped or model_stopping)
      light_detected = self.stop_light_filter.x >= THRESHOLD

      should_stop_for_light = light_detected

      if self.frogpilot_planner.tracking_lead:
        lead = self.frogpilot_planner.lead_one

        # stopped lead
        lead_is_stopped = lead.vLead < 2.0

        # lead is beyond the stop point beyond 5.5m
        stop_is_distinct = model_length < (lead.dRel - 5.5)

        # stop if we have a lead that we dont care about
        should_stop_for_light = light_detected and (lead_is_stopped or stop_is_distinct)

      self.stop_light_detected = should_stop_for_light

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

  def get_safe_distance(self, velocity, time_threshold):
      """
      Return the distance to the target.
      Let's say we want to stop for a red light 9 seconds out.
      Mazda can only brake at 3 m/s^2. It's possible our 9-second goal will overshoot it.
      If we won't make the destination in the target time, we can brake earlier
      """
      # Mazda limits to 3 m/ss
      # Pull the braking limit from the car controller base (2.95 m/ss)
      safe_decel = abs(ACCEL_MIN) * 0.925 # Take the car's max braking, and give a bit of wiggle room just in case

      # 1. Time threshold ()
      d_time = velocity * time_threshold

      # 2. Physics Limit (v^2 / 2a)
      d_physics = (velocity ** 2) / (2 * safe_decel)

      # Return the larger distance (forcing us to engage EARLIER if we are going too fast)
      return max(d_time, d_physics)