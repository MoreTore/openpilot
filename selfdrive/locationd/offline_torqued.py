#!/usr/bin/env python3
import numpy as np
from collections import deque, defaultdict
import signal
import threading

import cereal.messaging as messaging
from cereal import car, log
from openpilot.selfdrive.controls.lib.vehicle_model import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.locationd.helpers import PointBuckets, ParameterEstimator, PoseCalibrator, Pose
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.api import CommaApi
from openpilot.tools.lib.auth_config import get_token


HISTORY = 5  # secs
POINTS_PER_BUCKET = 5000
MIN_POINTS_TOTAL = 4000
MIN_POINTS_TOTAL_QLOG = 600
FIT_POINTS_TOTAL = 2000
FIT_POINTS_TOTAL_QLOG = 600
MIN_VEL = 15  # m/s
FRICTION_FACTOR = 1.5  # ~85% of data coverage
FACTOR_SANITY = 0.3
FACTOR_SANITY_QLOG = 0.5
FRICTION_SANITY = 0.5
FRICTION_SANITY_QLOG = 0.8
STEER_MIN_THRESHOLD = 0.02  # torque command threshold
MIN_FILTER_DECAY = 50
MAX_FILTER_DECAY = 250
LAT_ACC_THRESHOLD = 4 # m/s^2 maximum lateral acceleration allowed
LOOKBACK = 0.5  # secs for sensor standard deviation
SYNTHETIC_POINTS = 100  # number of synthetic data points to generate when starting from scratch
MIN_SIGMOID_SHARPNESS = 0.0
MAX_SIGMOID_SHARPNESS = 10.0
MIN_SIGMOID_TORQUE_GAIN = 0.0
MAX_SIGMOID_TORQUE_GAIN = 2.0
MIN_LAT_ACCEL_FACTOR = 0.0
MAX_LAT_ACCEL_FACTOR = 2.0
MAX_LAT_ACCEL_OFFSET = 0.3

STEER_BUCKET_BOUNDS = [
  (-1.0, -0.9), (-0.9, -0.8), (-0.8, -0.7), (-0.7, -0.6), (-0.6, -0.5),
  (-0.5, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0), (0, 0.1),
  (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7),
  (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)
]

MIN_BUCKET_POINTS = np.array([
  0, 0, 0, 0, 0,
  100, 300, 500, 500, 500,
  500, 300, 100, 0, 0,
  0, 0, 0,
])
MIN_ENGAGE_BUFFER = 2  # secs

VERSION = 2  # bump this to invalidate old parameter caches
ALLOWED_BRANDS = ['toyota', 'hyundai', 'rivian', 'honda']
ALLOWED_CARS = ['MAZDA_3_2019']


def sig_centered(z):
  pos = 1.0 / (1.0 + np.exp(-z)) - 0.5
  neg = np.exp(z) / (1.0 + np.exp(z)) - 0.5
  return np.where(z >= 0.0, pos, neg)

def model(x, a, b, c, d):
  xs = x - d
  return sig_centered(a * xs) * b + c * xs

def jacobian(x, a, b, c, d):
  xs = x - d
  # plain σ for derivative (cheaper than calling centred helper again)
  s  = 1.0 / (1.0 + np.exp(-np.clip(a * xs, -50.0, 50.0)))
  ds = s * (1.0 - s)          # σ′(z)
  sc = s - 0.5                # (σ − 0.5) value

  # Cols: ∂f/∂a,  ∂f/∂b,  ∂f/∂c,  ∂f/∂d    (N × 4)
  return np.column_stack([
    b * ds * xs,              # a-derivative
    sc,                       # b-derivative
    xs,                       # c-derivative
    -b * a * ds - c           # d-derivative
  ])

def slope2rot(slope):
  sin = np.sqrt(slope ** 2 / (slope ** 2 + 1))
  cos = np.sqrt(1 / (slope ** 2 + 1))
  return np.array([[cos, -sin], [sin, cos]])


class TorqueBuckets(PointBuckets):
  def add_point(self, x, y):
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, 1.0, y])
        break


class ThreadSafeTorqueBuckets(TorqueBuckets):
  """A threadsafe variant to aggregate points from multiple routes concurrently."""
  def __init__(self, *args, **kwargs):
    # Use an RLock to avoid deadlocks from re-entrant acquisitions when
    # methods like is_valid() call into super() which may invoke __len__(),
    # and our __len__ also acquires the same lock.
    from threading import RLock
    super().__init__(*args, **kwargs)
    self._lock = RLock()

  def __len__(self):
    with self._lock:
      return super().__len__()

  def is_valid(self) -> bool:
    with self._lock:
      return super().is_valid()

  def is_calculable(self) -> bool:
    with self._lock:
      return super().is_calculable()

  def add_point(self, x, y):
    with self._lock:
      return super().add_point(x, y)

  def get_points(self, num_points: int = None):
    with self._lock:
      return super().get_points(num_points)

  def load_points(self, points: list[list[float]]):
    with self._lock:
      return super().load_points(points)


class TorqueEstimator(ParameterEstimator):
  def __init__(self, CP, decimated=False, track_all_points=False, *, quiet: bool = False, shared_buckets: PointBuckets | None = None):
    self.hist_len = int(HISTORY / DT_MDL)
    self.lag = 0.0
    self.track_all_points = track_all_points  # for offline analysis, without max lateral accel or max steer torque filters
    self.quiet = quiet
    self._shared_buckets = shared_buckets
    if decimated:
      self.min_bucket_points = MIN_BUCKET_POINTS / 10
      self.min_points_total = MIN_POINTS_TOTAL_QLOG
      self.fit_points = FIT_POINTS_TOTAL_QLOG
      self.factor_sanity = FACTOR_SANITY_QLOG
      self.friction_sanity = FRICTION_SANITY_QLOG

    else:
      self.min_bucket_points = MIN_BUCKET_POINTS
      self.min_points_total = MIN_POINTS_TOTAL
      self.fit_points = FIT_POINTS_TOTAL
      self.factor_sanity = FACTOR_SANITY
      self.friction_sanity = FRICTION_SANITY

    self.offline_friction = 0.0
    self.offline_latAccelFactor = 0.0
    self.offline_sigmoidSharpness = 0.0
    self.offline_sigmoidTorqueGain = 0.0

    self.resets = 0.0

    self.use_params = (CP.carName in ALLOWED_BRANDS or CP.carFingerprint in ALLOWED_CARS) and \
                      CP.lateralTuning.which() == 'torque'

    if CP.lateralTuning.which() == 'torque':
      self.offline_friction = 0.36
      self.offline_latAccelFactor = .1
      self.offline_sigmoidSharpness = 3.8818
      self.offline_sigmoidTorqueGain = 1.5

    self.calibrator = PoseCalibrator()

    self.reset()

    initial_params = {
      'latAccelFactor': self.offline_latAccelFactor,
      'latAccelOffset': 0.0,
      'frictionCoefficient': self.offline_friction,
      'sigmoidSharpness': self.offline_sigmoidSharpness,
      'sigmoidTorqueGain': self.offline_sigmoidTorqueGain,
      'points': []
    }

    # if any of the initial params are NaN, set them to 0.0 but skip "points"
    for k, v in initial_params.items():
      if isinstance(v, float) and np.isnan(v):
        initial_params[k] = 0.0

    self.linear_tune = initial_params["sigmoidSharpness"] == 0.0 and initial_params["sigmoidTorqueGain"] == 0.0
    self.decay = MIN_FILTER_DECAY
    self.min_lataccel_factor = (1.0 - self.factor_sanity) * self.offline_latAccelFactor
    self.max_lataccel_factor = (1.0 + self.factor_sanity) * self.offline_latAccelFactor
    self.min_sigmoid_sharpness = (1.0 - self.factor_sanity) * self.offline_sigmoidSharpness
    self.max_sigmoid_sharpness = (1.0 + self.factor_sanity) * self.offline_sigmoidSharpness
    self.min_sigmoid_torque_gain = (1.0 - self.factor_sanity) * self.offline_sigmoidTorqueGain
    self.max_sigmoid_torque_gain = (1.0 + self.factor_sanity) * self.offline_sigmoidTorqueGain
    self.min_friction = (1.0 - self.friction_sanity) * self.offline_friction
    self.max_friction = (1.0 + self.friction_sanity) * self.offline_friction

    # try to restore cached params
    params = Params()
    params_cache = params.get("CarParamsPrevRoute")
    torque_cache = params.get("LiveTorqueParameters")
    if params_cache is not None and torque_cache is not None:
      try:
        with log.Event.from_bytes(torque_cache) as log_evt:
          cache_ltp = log_evt.liveTorqueParameters
        with car.CarParams.from_bytes(params_cache) as msg:
          cache_CP = msg
        if self.get_restore_key(cache_CP, cache_ltp.version) == self.get_restore_key(CP, VERSION):
          if cache_ltp.liveValid:
            initial_params = {
              'latAccelFactor': cache_ltp.latAccelFactorFiltered,
              'latAccelOffset': cache_ltp.latAccelOffsetFiltered,
              'frictionCoefficient': cache_ltp.frictionCoefficientFiltered,
              'sigmoidSharpness': cache_ltp.sigmoidSharpnessFiltered,
              'sigmoidTorqueGain': cache_ltp.sigmoidTorqueGainFiltered,
            }
          initial_params['points'] = cache_ltp.points
          self.decay = cache_ltp.decay
          self.filtered_points.load_points(initial_params['points'])
          cloudlog.info("restored torque params from cache")
      except Exception:
        cloudlog.exception("failed to restore cached torque params")
        params.remove("LiveTorqueParameters")

    # if not initial_params.get('points'):
    #   self.generate_points(initial_params)

    self.filtered_params = {}
    for param in initial_params:
      self.filtered_params[param] = FirstOrderFilter(initial_params[param], self.decay, DT_MDL)

    self.nonlinear_params = np.array([self.offline_sigmoidSharpness, self.offline_sigmoidTorqueGain, self.offline_latAccelFactor, 0.0])

  @staticmethod
  def get_restore_key(CP, version):
    a, b , c, d = None, None , None, None
    if CP.lateralTuning.which() == 'torque':
      a = 1.0 # CP.lateralTuning.torque.sigmoidSharpness
      b = 0.1 # CP.lateralTuning.torque.sigmoidTorqueGain
      c = 0.36 # CP.lateralTuning.torque.friction
      d = 3.8818 # CP.lateralTuning.torque.latAccelFactor
    print(CP.lateralTuning.which())
    return (CP.carFingerprint, CP.lateralTuning.which(), a, b, c, d, version)

  def reset(self):
    self.resets += 1.0
    self.decay = MIN_FILTER_DECAY
    self.raw_points = defaultdict(lambda: deque(maxlen=self.hist_len))
    # Allow sharing buckets across estimators for multithreaded processing
    if getattr(self, "_shared_buckets", None) is not None:
      self.filtered_points = self._shared_buckets
    else:
      self.filtered_points = TorqueBuckets(x_bounds=STEER_BUCKET_BOUNDS,
                                           min_points=self.min_bucket_points,
                                           min_points_total=self.min_points_total,
                                           points_per_bucket=POINTS_PER_BUCKET,
                                           rowsize=3)
    self.all_torque_points = []

  def estimate_params(self) -> tuple:
    pts = self.filtered_points.get_points(self.fit_points)
    if pts.size == 0:
      cloudlog.error("No points to fit.")
      return (np.nan,)*5

    # ── linear fit and friction estimate ───────────────────
    # total least square solution as both x and y are noisy observations
    # this is empirically the slope of the hysteresis parallelogram as opposed to the line through the diagonals
    try:
      _, _, v = np.linalg.svd(pts, full_matrices=False)
      slope, offset = -v.T[0:2, 2] / v.T[2, 2]
      _, spread = np.matmul(pts[:, [0, 2]], slope2rot(slope)).T
      friction_coeff = np.std(spread) * FRICTION_FACTOR
      # robust fallback if std is degenerate
      if not np.isfinite(friction_coeff) or friction_coeff <= 0:
        med = float(np.median(spread))
        mad = float(np.median(np.abs(spread - med)))
        friction_coeff = 1.4826 * mad * FRICTION_FACTOR
    except np.linalg.LinAlgError as e:
      cloudlog.exception(f"Error computing live torque params: {e}")
      slope = offset = friction_coeff = np.nan

    if self.linear_tune:
      return (0.0, 0.0, slope, offset, friction_coeff)

    # ── Regularized Gauss–Newton with prior (simple and stable) ──
    # Center prior on filtered params if available to prevent jitter.
    try:
      p_prior = np.array([
        float(self.filtered_params['sigmoidSharpness'].x),
        float(self.filtered_params['sigmoidTorqueGain'].x),
        float(self.filtered_params['latAccelFactor'].x),
        float(self.filtered_params['latAccelOffset'].x),
      ], dtype=float)
    except Exception:
      d0 = float(-offset / slope) if (np.isfinite(slope) and abs(slope) > 1e-6) else 0.0
      p_prior = np.array([
        self.offline_sigmoidSharpness,
        self.offline_sigmoidTorqueGain,
        self.offline_latAccelFactor,
        d0
      ], dtype=float)

    params = p_prior.copy()
    x = pts[:, 2].astype(float)
    y = pts[:, 0].astype(float)

    N = float(len(x))
    it_max = 6
    tol = 2e-4
    # Heavier regularization on d and c to reduce drift
    reg_diag = N * np.array([5e-3, 5e-3, 5e-2, 1e-1], dtype=float)

    for _ in range(it_max):
      a, b, c, d = params
      r = model(x, a, b, c, d) - y
      J = jacobian(x, a, b, c, d)

      # light Huber using friction as scale to tame outliers
      if np.isfinite(friction_coeff) and friction_coeff > 0:
        delta_h = 1.5 * float(friction_coeff)
        abs_r = np.abs(r)
        w = np.where(abs_r <= delta_h, 1.0, (delta_h / (abs_r + 1e-12)))
        Jw = J * w[:, None]
        rw = r * w
      else:
        Jw = J
        rw = r

      H = Jw.T @ Jw + np.diag(reg_diag)
      g = Jw.T @ rw + reg_diag * (params - p_prior)
      try:
        delta = -np.linalg.solve(H, g)
      except np.linalg.LinAlgError:
        return (0.0, 0.0, slope, offset, friction_coeff)

      # trust-region like clamp
      max_step = np.array([0.2, 0.08, 0.05, 0.01])
      delta = np.clip(delta, -max_step, max_step)
      params_new = params + delta

      # bounds
      params_new[0] = np.clip(params_new[0], MIN_SIGMOID_SHARPNESS, MAX_SIGMOID_SHARPNESS)
      params_new[1] = np.clip(params_new[1], MIN_SIGMOID_TORQUE_GAIN, MAX_SIGMOID_TORQUE_GAIN)
      params_new[2] = np.clip(params_new[2], MIN_LAT_ACCEL_FACTOR,  MAX_LAT_ACCEL_FACTOR)
      params_new[3] = np.clip(params_new[3], -MAX_LAT_ACCEL_OFFSET, MAX_LAT_ACCEL_OFFSET)

      if np.max(np.abs(delta)) < tol:
        params = params_new
        break
      params = params_new

    if not np.all(np.isfinite(params)):
      return (0.0, 0.0, slope, offset, friction_coeff)

    a, b, c, d = params
    self.nonlinear_params = np.array([a, b, c, d])
    self.friction_coeff = friction_coeff
    if not self.quiet:
      print(f"Estimated params: a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}, friction={friction_coeff:.3f}")
    return a, b, c, d, friction_coeff

  def update_params(self, params):
    self.decay = min(self.decay + DT_MDL, MAX_FILTER_DECAY)
    for param, value in params.items():
      self.filtered_params[param].update(value)
      self.filtered_params[param].update_alpha(self.decay)

  def handle_log(self, t, which, msg):
    if which == "carControl":
      self.raw_points["carControl_t"].append(t + self.lag)
      self.raw_points["lat_active"].append(msg.carControl.latActive)
    elif which == "carOutput":
      self.raw_points["carOutput_t"].append(t + self.lag)
      self.raw_points["steer_torque"].append(-msg.carOutput.actuatorsOutput.steer)
    elif which == "carState":
      self.raw_points["carState_t"].append(t + self.lag)
      # TODO: check if high aEgo affects resulting lateral accel
      self.raw_points["vego"].append(msg.carState.vEgo)
      self.raw_points["steer_override"].append(msg.carState.steeringPressed)
      self.raw_points["steer_angle"].append(msg.carState.steeringAngleDeg)
    elif which == "liveCalibration":
      self.calibrator.feed_live_calib(msg.liveCalibration)
    elif which == "liveDelay":
      self.lag = msg.liveDelay.lateralDelay
    # calculate lateral accel from past steering torque
    elif which == "livePose":
      if len(self.raw_points['steer_torque']) == self.hist_len:
        device_pose = Pose.from_live_pose(msg.livePose)
        calibrated_pose = self.calibrator.build_calibrated_pose(device_pose)
        angular_velocity_calibrated = calibrated_pose.angular_velocity

        yaw_rate = angular_velocity_calibrated.yaw
        roll = device_pose.orientation.roll
        # check lat active up to now (without lag compensation)
        lat_active = np.interp(np.arange(t - MIN_ENGAGE_BUFFER, t + self.lag, DT_MDL),
                               self.raw_points['carControl_t'], self.raw_points['lat_active']).astype(bool)
        steer_override = np.interp(np.arange(t - MIN_ENGAGE_BUFFER, t + self.lag, DT_MDL),
                                   self.raw_points['carState_t'], self.raw_points['steer_override']).astype(bool)
        vego = np.interp(t, self.raw_points['carState_t'], self.raw_points['vego'])
        steer = np.interp(t, self.raw_points['carOutput_t'], self.raw_points['steer_torque']).item()
        lateral_acc = (vego * yaw_rate) - (np.sin(roll) * ACCELERATION_DUE_TO_GRAVITY).item()
        # Steering wheel stability check
        steering_angle_std = np.std(np.interp(np.arange(t - LOOKBACK, t + self.lag, DT_MDL),
                                        self.raw_points['carState_t'], self.raw_points['steer_angle']))
        if all(lat_active) and not any(steer_override) and (vego > MIN_VEL) and (abs(steer) > STEER_MIN_THRESHOLD) and (steering_angle_std < 1.0):
          if abs(lateral_acc) <= LAT_ACC_THRESHOLD:
            self.filtered_points.add_point(steer, lateral_acc)
            if not self.quiet:
              print(f"t={t:.1f}s: added point (steer={steer:.3f}, lat_acc={lateral_acc:.3f})")

          if self.track_all_points:
            self.all_torque_points.append([steer, lateral_acc])


  def get_msg(self, valid=True, with_points=False):
    msg = messaging.new_message('liveTorqueParameters')
    msg.valid = valid
    liveTorqueParameters = msg.liveTorqueParameters
    liveTorqueParameters.version = VERSION
    liveTorqueParameters.useParams = self.use_params

    # Calculate raw estimates when possible, only update filters when enough points are gathered
    if self.filtered_points.is_calculable():
      sigmoidSharpness, sigmoidTorqueGain, latAccelFactor, latAccelOffset, frictionCoeff = self.estimate_params()
      # liveTorqueParameters.latAccelFactorRaw = float(latAccelFactor)
      # liveTorqueParameters.latAccelOffsetRaw = float(latAccelOffset)
      # liveTorqueParameters.frictionCoefficientRaw = float(frictionCoeff)
      # liveTorqueParameters.sigmoidSharpnessRaw = float(sigmoidSharpness)
      # liveTorqueParameters.sigmoidTorqueGainRaw = float(sigmoidTorqueGain)

      if self.filtered_points.is_valid():
        if any(val is np.isnan(val) for val in [latAccelFactor, latAccelOffset, frictionCoeff, sigmoidSharpness, sigmoidTorqueGain]):
          cloudlog.exception("Live torque parameters are invalid.")
          liveTorqueParameters.liveValid = False
          self.reset()
        else:
          liveTorqueParameters.liveValid = True
          latAccelFactor = np.clip(latAccelFactor, self.min_lataccel_factor, self.max_lataccel_factor)
          frictionCoeff = np.clip(frictionCoeff, self.min_friction, self.max_friction)
          self.update_params({'latAccelFactor': latAccelFactor,
                              'latAccelOffset': latAccelOffset,
                              'frictionCoefficient': frictionCoeff,
                              'sigmoidSharpness': sigmoidSharpness,
                              'sigmoidTorqueGain': sigmoidTorqueGain,
                              })

    if with_points:
      liveTorqueParameters.points = self.filtered_points.get_points()[:, [0, 2]].tolist()

    liveTorqueParameters.latAccelFactorFiltered = float(self.filtered_params['latAccelFactor'].x)
    liveTorqueParameters.latAccelOffsetFiltered = float(self.filtered_params['latAccelOffset'].x)
    liveTorqueParameters.frictionCoefficientFiltered = float(self.filtered_params['frictionCoefficient'].x)
    # liveTorqueParameters.sigmoidSharpnessFiltered = float(self.filtered_params['sigmoidSharpness'].x)
    # liveTorqueParameters.sigmoidTorqueGainFiltered = float(self.filtered_params['sigmoidTorqueGain'].x)
    liveTorqueParameters.totalBucketPoints = len(self.filtered_points)

    liveTorqueParameters.decay = self.decay
    liveTorqueParameters.maxResets = self.resets
    return msg

  def generate_points(self, initial_params) -> None:
    print("Pre-loading points with synthetic data: ", initial_params)
    cloudlog.info(f"Pre-loading points with synthetic data: {initial_params}")

    a = initial_params['sigmoidSharpness']
    b = initial_params['sigmoidTorqueGain']
    c = initial_params['latAccelFactor']
    d = initial_params['latAccelOffset']
    friction = initial_params['frictionCoefficient']
    self.nonlinear_params = np.array([a, b, c, d])
    self.friction_coeff = friction

    rng = np.random.default_rng(42)
    x_sample = rng.uniform(-4, 4, SYNTHETIC_POINTS)
    sigma_base = 0.10
    lat_accel_jitter = x_sample + rng.normal(0, sigma_base, size=x_sample.shape)
    envelope = np.exp(-(lat_accel_jitter / 1.0) ** 2)
    steer_jitter = (
      model(lat_accel_jitter, a, b, c, d)
      + rng.normal(0, sigma_base, size=x_sample.shape)
      + rng.normal(0, friction * envelope, size=x_sample.shape)
    )
    for τ, a_lat in zip(steer_jitter, lat_accel_jitter, strict=False):
      self.filtered_points.add_point(τ, a_lat)

  def plot(self, base_filename="bucket_plot", file_ext=".png"):
    import matplotlib.pyplot as plt
    plt.ion()

    # Thread-safe snapshot of all points
    combined = self.filtered_points.get_points()
    if combined.size == 0:
      return

    steer_all   = combined[:, 0]
    lateral_all = combined[:, 2]

    # ── figure ───────────────────────────────────────────────
    if not hasattr(self, 'fig'):
      self.fig = plt.figure(figsize=(16, 4))
      self.ax = self.fig.add_subplot(111)
    else:
      self.ax.clear()

    # fitted curve + friction band
    a, b, c, d = self.nonlinear_params          # 4-tuple
    sigma_f    = getattr(self, "friction_coeff", 0.0)

    x_line = np.linspace(-4, 4, 400)
    y_fit  = model(x_line, a, b, c, d)

    self.ax.plot(x_line, y_fit,          color="red",  lw=2, label="Fitted curve")
    if sigma_f > 0:
        self.ax.plot(x_line, y_fit + sigma_f, color="blue", ls="--", lw=1.5, label="friction band")
        self.ax.plot(x_line, y_fit - sigma_f, color="blue", ls="--", lw=1.5, label="")
        # fill in the area between the two curves
        self.ax.fill_between(x_line, y_fit - sigma_f, y_fit + sigma_f, color="grey", alpha=0.3)
    self.ax.scatter(lateral_all, steer_all, s=8, alpha=0.4, label="Filtered samples")

    # ── cosmetics ────────────────────────────────────────────
    self.ax.set_xlim(-4, 4)
    self.ax.set_ylim(-1, 1)
    self.ax.set_xlabel("Lateral acceleration (m/s²)")
    self.ax.set_ylabel("Steering torque (Nm equiv)")
    self.ax.set_title("Torque vs lateral acceleration (all buckets)")
    # print the current parameters
    if hasattr(self, 'friction_coeff'):
      self.ax.text(0.05, 0.9, f"Friction: {self.friction_coeff:.3f}", transform=self.ax.transAxes)
      self.ax.text(0.05, 0.85, f"LatAccelFactor: {self.filtered_params['latAccelFactor'].x:.3f}", transform=self.ax.transAxes)
      self.ax.text(0.05, 0.8, f"SigmoidSharpness: {self.filtered_params['sigmoidSharpness'].x:.3f}", transform=self.ax.transAxes)
      self.ax.text(0.05, 0.75, f"SigmoidTorqueGain: {self.filtered_params['sigmoidTorqueGain'].x:.3f}", transform=self.ax.transAxes)
      self.ax.text(0.05, 0.7, f"LatAccelOffset: {self.filtered_params['latAccelOffset'].x:.3f}", transform=self.ax.transAxes)
      self.ax.text(0.05, 0.65, f"Decay: {self.decay:.3f}", transform=self.ax.transAxes)
      self.ax.text(0.05, 0.6, f"Valid: {self.filtered_points.is_valid()}", transform=self.ax.transAxes)

    self.ax.grid(True)
    self.ax.legend()
    self.fig.tight_layout()

    self.fig.canvas.draw()
    plt.pause(0.05)
    # print is noisy; keep it quiet during frequent refreshes


def _process_route(lr, shared_buckets, demo, cp_holder=None):
  """Process a single route file, adding points to shared_buckets if provided.
  Returns (CarParams, frames_processed)."""
  estimator = None
  frames = 0
  cp_seen = None
  # optional cooperative shutdown
  stop_event = None
  try:
    # if caller passed via cp_holder key, fetch it (backward compat if not provided separately)
    stop_event = cp_holder.get('stop_event') if isinstance(cp_holder, dict) else None
  except Exception:
    stop_event = None
  for msg in lr:
    if stop_event is not None and stop_event.is_set():
      break
    frames += 1
    if msg.which() == 'carParams':
      cp_seen = msg.carParams
      # publish first seen CP to plotter
      if cp_holder is not None and cp_holder.get('cp') is None:
        cp_holder['cp'] = cp_seen
      if estimator is None:
        estimator = TorqueEstimator(cp_seen, decimated=demo, track_all_points=False, quiet=True,
                                    shared_buckets=shared_buckets)
    if estimator is not None:
      t = msg.logMonoTime * 1e-9
      estimator.handle_log(t, msg.which(), msg)

    if frames % 40000 == 0:
      print(f"Processed {frames} frames")
  return cp_seen, frames


def _plot_worker(shared_buckets, demo, stop_event, cp_holder):
  """Background thread that periodically renders the live plot from shared buckets.
  It waits for the first CarParams to appear, then reuses a TorqueEstimator with the shared buckets.
  """
  import time
  estimator = None
  while not stop_event.is_set():
    cp = cp_holder.get('cp')
    if cp is not None and estimator is None:
      estimator = TorqueEstimator(cp, decimated=demo, track_all_points=False, quiet=True,
                                  shared_buckets=shared_buckets)
    if estimator is not None:
      # compute current params and draw
      _ = estimator.get_msg(valid=True, with_points=False)
      estimator.plot()
    # modest refresh rate
    time.sleep(0.25)
  # final draw on exit
  if estimator is not None:
    _ = estimator.get_msg(valid=True, with_points=True)
    estimator.plot()


def main(demo=False, route=None, routes=None, workers: int = 0, use_pygame: bool = False, stop_event: threading.Event | None = None):
  config_realtime_process([0, 1, 2, 3], 5)

  # shared stop flag
  if stop_event is None:
    stop_event = threading.Event()

  # Single-route legacy path
  if route and not routes:
    lr = LogReader(route, sort_by_time=True)
    estimator = None
    frame = 0
    for msg in lr:
      frame += 1
      if msg.which() == 'carParams':
        print(f"Processing CarParams from route: {route}")
      match msg.which():
        case 'carParams':
          if estimator is not None:
            continue
          estimator = TorqueEstimator(msg.carParams, decimated=demo, track_all_points=True)
          print(f"Loaded CarParams from route: {route}")

      if estimator is not None:
        t = msg.logMonoTime * 1e-9
        estimator.handle_log(t, msg.which(), msg)
        msg = estimator.get_msg(valid=True, with_points=True)
        if frame % 10000 == 0:
          print(f"Processed {frame} frames")
          estimator.plot()
    # final plot
    if estimator is not None:
      estimator.plot()
    return

  # Multiroute concurrent processing
  if routes:
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    routes_dict = routes

    # Shared thread-safe buckets configured with generous totals (min total across estimators)
    shared = ThreadSafeTorqueBuckets(x_bounds=STEER_BUCKET_BOUNDS,
                                     min_points=MIN_BUCKET_POINTS,
                                     min_points_total=MIN_POINTS_TOTAL,
                                     points_per_bucket=POINTS_PER_BUCKET,
                                     rowsize=3)
    print(f"Starting concurrent processing with {workers} workers for {len(routes_dict)} routes")
    # Holder to receive first CarParams from worker(s)
    cp_holder = {'cp': None, 'stop_event': stop_event}

    results = []
    estimator = None
    pygame_plotter = None
    if use_pygame:
      try:
        from openpilot.selfdrive.locationd.torque_viewer_pygame import PygamePlotter
        pygame_plotter = PygamePlotter()
      except Exception as e:
        print(f"Pygame viewer unavailable: {e}")
        pygame_plotter = None

    def _cleanup_viewers():
      # Try to close matplotlib and pygame viewers gracefully
      try:
        import matplotlib.pyplot as plt
        plt.close('all')
      except Exception:
        pass
      if pygame_plotter is not None:
        try:
          pygame_plotter.close()
        except Exception:
          pass

    try:
      with ThreadPoolExecutor(max_workers=workers) as ex:
        pending = {ex.submit(_process_route, r, shared, demo, cp_holder): r for r in routes_dict.values()}
        while pending and not stop_event.is_set():
          # Plot/update in main thread at ~4 Hz, while also consuming finished futures
          done, not_done = wait(set(pending.keys()), timeout=0.25, return_when=FIRST_COMPLETED)
          # Initialize estimator as soon as we have CarParams
          if estimator is None and cp_holder.get('cp') is not None:
            estimator = TorqueEstimator(cp_holder['cp'], decimated=demo, track_all_points=False, quiet=False,
                                        shared_buckets=shared)
          # Main-thread plot refresh
          if estimator is not None:
            _ = estimator.get_msg(valid=True, with_points=False)
            if pygame_plotter is not None:
              # draw via pygame
              combined = shared.get_points()
              a, b, c, d = tuple(estimator.nonlinear_params) if hasattr(estimator, 'nonlinear_params') else (0.0, 0.0, 0.0, 0.0)
              sigma_f = float(getattr(estimator, 'friction_coeff', 0.0))
              # record history
              try:
                pygame_plotter.record_params(a, b, c, d, sigma_f)
              except Exception:
                pass
              info = [
                f"points: {combined.shape[0] if combined is not None else 0}",
                f"friction: {sigma_f:.3f}",
                f"a={a:.3f} b={b:.3f} c={c:.3f} d={d:.3f}",
              ]
              keep = pygame_plotter.draw_frame(combined, model, (a, b, c, d), sigma_f, info)
              if not keep:
                stop_event.set()
                break
            else:
              estimator.plot()
          # Handle finished workers
          for fut in done:
            r = pending.pop(fut, None)
            try:
              cp_seen, frames = fut.result()
              print(f"Finished route {r} (frames={frames})")
              if cp_seen is not None:
                results.append(cp_seen)
            except Exception as e:
              print(f"Error processing route {r}: {e}")
          # Rebuild pending dict from not_done
          pending = {fut: pending[fut] for fut in not_done if fut in pending}
    except KeyboardInterrupt:
      # Single Ctrl+C: signal all threads to stop and cancel futures
      stop_event.set()
      try:
        ex.shutdown(wait=False, cancel_futures=True)
      except Exception:
        pass
      _cleanup_viewers()
      return
    finally:
      if stop_event.is_set():
        _cleanup_viewers()

    # Use the first CP to instantiate an estimator for parameter computation and plotting
    if not results:
      print("No CarParams found across routes; nothing to do.")
      return

    # If we never initialized during the loop (unlikely), do it now
    if estimator is None:
      estimator = TorqueEstimator(results[0], decimated=demo, track_all_points=False, quiet=False,
                                  shared_buckets=shared)
      # compute final message/params and plot
      _ = estimator.get_msg(valid=True, with_points=True)
      if pygame_plotter is not None:
        # Final few frames to render end state, then exit when window closed
        import time
        for _ in range(60):
          if stop_event.is_set():
            break
          combined = shared.get_points()
          a, b, c, d = tuple(estimator.nonlinear_params)
          sigma_f = float(getattr(estimator, 'friction_coeff', 0.0))
          info = [
            f"points: {combined.shape[0] if combined is not None else 0}",
            f"friction: {sigma_f:.3f}",
            f"a={a:.3f} b={b:.3f} c={c:.3f} d={d:.3f}",
          ]
          if not pygame_plotter.draw_frame(combined, model, (a, b, c, d), sigma_f, info):
            stop_event.set()
            break
          time.sleep(0.016)
      else:
        estimator.plot(base_filename="bucket_plot", file_ext=".png")
    return

if __name__ == "__main__":
  import argparse
  from concurrent.futures import ThreadPoolExecutor, as_completed
  # get current cpu count
  import multiprocessing
  import sys
  cpu_count = multiprocessing.cpu_count()
  print(f"Detected {cpu_count} CPU cores")
  parser = argparse.ArgumentParser(description='Offline torque estimator (single or multi-route).')
  parser.add_argument('--dongle_id', default="3b58edf884ab4eaf", help='Dongle ID to fetch routes from (default=3b58edf884ab4eaf).')
  parser.add_argument('--route', help='Route ID to fetch.')
  parser.add_argument('--limit', default=100, help='Number of routes to fetch from device. Default=100.')
  parser.add_argument('--demo', action='store_true', help='Demo mode: decimated thresholds.')
  parser.add_argument('--workers', type=int, default=cpu_count, help='Number of worker threads (default=min(cpu_count, len(routes))).')
  parser.add_argument('--pygame', action='store_true', help='Use pygame viewer for fast interactive plotting.')
  args = parser.parse_args()

  # install signal handlers to set a global stop flag
  stop_event = threading.Event()

  def _sig_handler(signum, frame):
    # set once; subsequent signals fall back to default
    if not stop_event.is_set():
      print("Received interrupt, shutting down...")
      stop_event.set()
  try:
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)
  except Exception:
    pass

  api = CommaApi(get_token())
  if not args.route:
    routes = api.request("GET", f"/v1/devices/{args.dongle_id}/routes_segments", params={"limit": args.limit})
  else:
    routes = [{"fullname": args.route}]
  print(f"Fetched {len(routes)} routes from device {args.dongle_id}")
  lr_dict = {}

  # If a single route was provided on the command line, call main in single-threaded
  # mode and skip pre-loading multiple LogReader objects via ThreadPoolExecutor.
  if args.route:
    # main will open and process the single route directly
    main(demo=args.demo, route=args.route, routes=None, workers=0, use_pygame=args.pygame, stop_event=stop_event)
    sys.exit(0)

  def create_lr(route):
    try:
        return route["fullname"], LogReader(route["fullname"], sort_by_time=True)
    except Exception as e:
        print(f"Error loading {route['fullname']}: {e}")
        return route["fullname"], None

  try:
    with ThreadPoolExecutor(max_workers=min(args.workers, len(routes))) as executor:
      futures = [executor.submit(create_lr, r) for r in routes]
      for future in as_completed(futures):
        if stop_event.is_set():
          break
        key, lr = future.result()
        if lr is not None:
          lr_dict[key] = lr
  except KeyboardInterrupt:
    stop_event.set()
    try:
      executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
      pass

  main(demo=args.demo, route=None, routes=lr_dict, workers=min(args.workers, len(lr_dict)), use_pygame=args.pygame, stop_event=stop_event)
