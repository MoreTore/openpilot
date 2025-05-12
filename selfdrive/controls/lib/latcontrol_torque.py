import math
import numpy as np

from cereal import log
from opendbc.car.interfaces import LatControlInputs
from opendbc.car.vehicle_model import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.common.pid import PIDController
from openpilot.common.realtime import DT_CTRL
from openpilot.common.filter_simple import FirstOrderFilter

# At higher speeds, lateral accel correlates to steering torque.
# We apply a sig+lin feed-forward, friction compensation,
# and inertial compensation on steering rate acceleration.

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [15, 13, 10, 5]

class LatControlTorque(LatControl):
    def __init__(self, CP, CI):
        super().__init__(CP, CI)
        self.torque_params = CP.lateralTuning.torque.as_builder()
        self.pid = PIDController(
            self.torque_params.kp,
            self.torque_params.ki,
            k_f=self.torque_params.kf,
            pos_limit=self.steer_max,
            neg_limit=-self.steer_max
        )
        self.torque_from_lateral_accel = CI.torque_from_lateral_accel()

        # Inertia feed-forward setup using steering rate
        self.I_sw = 0.1  # kg·m²
        self.tau_inertia = 0.05  # seconds
        self.inertia_filter = FirstOrderFilter(0.0, self.tau_inertia, DT_CTRL)
        self.prev_rate_rad = 0.0

    def update_live_torque_params(self, latAccelFactor, latAccelOffset, friction):
        self.torque_params.latAccelFactor = latAccelFactor
        self.torque_params.latAccelOffset = latAccelOffset
        self.torque_params.friction = friction

    def update(self, active, CS, VM, params, steer_limited_by_controls,
               desired_curvature, calibrated_pose, curvature_limited):
        pid_log = log.ControlsState.LateralTorqueState.new_message()
        if not active:
            pid_log.active = False
            return 0.0, 0.0, pid_log

        # Compute actual lateral acceleration
        actual_curvature_vm = -VM.calc_curvature(
            math.radians(CS.steeringAngleDeg - params.angleOffsetDeg),
            CS.vEgo, params.roll
        )
        roll_comp = params.roll * ACCELERATION_DUE_TO_GRAVITY
        if self.torque_params.useSteeringAngle:
            actual_curvature = actual_curvature_vm
            curvature_deadzone = abs(
                VM.calc_curvature(math.radians(self.steering_angle_deadzone_deg), CS.vEgo, 0.0)
            )
        else:
            assert calibrated_pose is not None
            pose_curv = calibrated_pose.angular_velocity.yaw / CS.vEgo
            actual_curvature = np.interp(CS.vEgo, [2.0, 5.0], [actual_curvature_vm, pose_curv])
            curvature_deadzone = 0.0
        desired_lat_accel = desired_curvature * CS.vEgo**2
        actual_lat_accel = actual_curvature * CS.vEgo**2
        lateral_deadzone = curvature_deadzone * CS.vEgo**2

        # PID error with low-speed curvature blend
        low_k = np.interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y)**2
        set_lat = desired_lat_accel + low_k * desired_curvature
        meas_lat = actual_lat_accel + low_k * actual_curvature
        torque_sp = self.torque_from_lateral_accel(
            LatControlInputs(set_lat, roll_comp, CS.vEgo, CS.aEgo),
            self.torque_params, set_lat, lateral_deadzone,
            friction_compensation=False, gravity_adjusted=False
        )
        torque_me = self.torque_from_lateral_accel(
            LatControlInputs(meas_lat, roll_comp, CS.vEgo, CS.aEgo),
            self.torque_params, meas_lat, lateral_deadzone,
            friction_compensation=False, gravity_adjusted=False
        )
        pid_log.error = float(torque_sp - torque_me)

        # Nonlinear + linear feed-forward (gravity adjusted)
        grav_adj = desired_lat_accel - roll_comp
        ff = self.torque_from_lateral_accel(
            LatControlInputs(grav_adj, roll_comp, CS.vEgo, CS.aEgo),
            self.torque_params,
            desired_lat_accel - actual_lat_accel,
            lateral_deadzone,
            friction_compensation=True,
            gravity_adjusted=True
        )

        # Inertia feed-forward: compute acceleration from steeringRateDeg
        rate_rad = math.radians(CS.steeringRateDeg)
        raw_alpha = (rate_rad - self.prev_rate_rad) / DT_CTRL
        self.prev_rate_rad = rate_rad
        filt_alpha = self.inertia_filter.update(raw_alpha)
        inertia_ff = self.I_sw * filt_alpha

        # Total feed-forward
        total_ff = ff + inertia_ff

        # PID update
        freeze = steer_limited_by_controls or CS.steeringPressed or CS.vEgo < 5
        output_torque = self.pid.update(
            pid_log.error,
            feedforward=total_ff,
            speed=CS.vEgo,
            freeze_integrator=freeze
        )

        # Logging
        pid_log.active = True
        pid_log.p = float(self.pid.p)
        pid_log.i = float(self.pid.i)
        pid_log.d = float(self.pid.d)
        pid_log.f = float(self.pid.f)
        pid_log.output = float(-output_torque)
        pid_log.actualLateralAccel = float(actual_lat_accel)
        pid_log.desiredLateralAccel = float(desired_lat_accel)
        pid_log.saturated = bool(
            self._check_saturation(abs(output_torque) >= self.steer_max - 1e-3,
                                   CS, steer_limited_by_controls, curvature_limited)
        )

        return -output_torque, 0.0, pid_log
