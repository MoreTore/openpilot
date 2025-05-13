#!/usr/bin/env python3
import numpy as np
from collections import deque, defaultdict

import cereal.messaging as messaging
from cereal import car, log
from opendbc.car.vehicle_model import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_MDL, DT_CTRL
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.locationd.helpers import PointBuckets, ParameterEstimator, PoseCalibrator, Pose
import matplotlib.pyplot as plt
import pickle
plt.ioff()

HISTORY = 5  # secs
POINTS_PER_BUCKET = 1500
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
STEER_MIN_THRESHOLD = 0.02
MIN_FILTER_DECAY = 50
MAX_FILTER_DECAY = 250
LAT_ACC_THRESHOLD = 4
# STEER_BUCKET_BOUNDS = [(-0.5, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0), (0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5)]
# MIN_BUCKET_POINTS = np.array([100, 300, 500, 500, 500, 500, 300, 100])

STEER_BUCKET_BOUNDS = [
    (-1.0, -0.9), (-0.9, -0.8), (-0.8, -0.7), (-0.7, -0.6), (-0.6, -0.5),
    (-0.5, -0.3), (-0.3, -0.2), (-0.2, -0.1), (-0.1, 0), (0, 0.1),
    (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7),
    (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)
]

MIN_BUCKET_POINTS = np.array([
    100, 100, 100, 100, 100,
    100, 300, 500, 500, 500,
    500, 300, 100, 100, 100,
    100, 100, 100,
])

MIN_ENGAGE_BUFFER = 2  # secs

VERSION = 2  # bump this to invalidate old parameter caches
ALLOWED_BRANDS = ['toyota', 'hyundai', 'rivian']
ALLOWED_CARS = ['MAZDA_3_2019']

NON_LINEAR_TORQUE_PARAMS = {
  "MAZDA_3_2019": (3.8818, 0.6873, 0.0999, 0.3605),
}


def sigmoid(z):
  z = np.clip(z, -50.0, 50.0)          # avoid overflow
  return 1.0 / (1.0 + np.exp(-z))

def model(x, a, b, c, d):
  xs = x - d
  return sigmoid(a * xs) * b + c * xs - 0.5 * b

def jacobian(x, a, b, c, d):
  xs = x - d
  s  = sigmoid(a * xs)
  ds = s * (1.0 - s)                  # σ′
  # ∂f/∂a, ∂f/∂b, ∂f/∂c, ∂f/∂d (N×4)
  return np.column_stack([
      b * ds * xs,                    # a-derivative
      s - 0.5,                        # b-derivative
      xs,                             # c-derivative
      -b * a * ds - c                  # d-derivative
  ])

def slope2rot(slope):
  sin = np.sqrt(slope ** 2 / (slope ** 2 + 1))
  cos = np.sqrt(1 / (slope ** 2 + 1))
  return np.array([[cos, -sin], [sin, cos]])

def op_friction(points):
  try:
    _, _, v = np.linalg.svd(points, full_matrices=False)
    slope, _ = -v.T[0:2, 2] / v.T[2, 2]
    _, spread = np.matmul(points[:, [0, 2]], slope2rot(slope)).T
    friction_coeff = np.std(spread) * FRICTION_FACTOR
  except np.linalg.LinAlgError as e:
    return 0
  return friction_coeff

class TorqueBuckets(PointBuckets):
  def add_point(self, x, y):
    for bound_min, bound_max in self.x_bounds:
      if (x >= bound_min) and (x < bound_max):
        self.buckets[(bound_min, bound_max)].append([x, 1.0, y])
        break


class TorqueEstimator(ParameterEstimator):
  def __init__(self, CP, decimated=False, track_all_points=False):
    self.hist_len = int(HISTORY / DT_CTRL)
    self.lag = 0.0
    self.track_all_points = track_all_points  # for offline analysis, without max lateral accel or max steer torque filters
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
    self.resets = 0.0
    self.use_params = CP.brand in ALLOWED_BRANDS and CP.lateralTuning.which() == 'torque'
    self.use_params |= CP.carFingerprint in ALLOWED_CARS

    if CP.lateralTuning.which() == 'torque':
      self.offline_friction = CP.lateralTuning.torque.friction
      self.offline_latAccelFactor = CP.lateralTuning.torque.latAccelFactor

    self.calibrator = PoseCalibrator()

    self.reset()

    initial_params = {
      'latAccelFactor': self.offline_latAccelFactor,
      'latAccelOffset': 0.0,
      'frictionCoefficient': self.offline_friction,
      'points': []
    }
    self.decay = MIN_FILTER_DECAY
    self.min_lataccel_factor = (1.0 - self.factor_sanity) * self.offline_latAccelFactor
    self.max_lataccel_factor = (1.0 + self.factor_sanity) * self.offline_latAccelFactor
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
              'frictionCoefficient': cache_ltp.frictionCoefficientFiltered
            }
          initial_params['points'] = cache_ltp.points
          self.decay = cache_ltp.decay
          self.filtered_points.load_points(initial_params['points'])
          cloudlog.info("restored torque params from cache")
      except Exception:
        cloudlog.exception("failed to restore cached torque params")
        params.remove("LiveTorqueParameters")

    self.filtered_params = {}
    for param in initial_params:
      self.filtered_params[param] = FirstOrderFilter(initial_params[param], self.decay, DT_CTRL)

  @staticmethod
  def get_restore_key(CP, version):
    a, b = None, None
    if CP.lateralTuning.which() == 'torque':
      a = CP.lateralTuning.torque.friction
      b = CP.lateralTuning.torque.latAccelFactor
    return (CP.carFingerprint, CP.lateralTuning.which(), a, b, version)

  def reset(self):
    self.resets += 1.0
    self.decay = MIN_FILTER_DECAY
    self.raw_points = defaultdict(lambda: deque(maxlen=self.hist_len))
    self.filtered_points = TorqueBuckets(x_bounds=STEER_BUCKET_BOUNDS,
                                         min_points=self.min_bucket_points,
                                         min_points_total=self.min_points_total,
                                         points_per_bucket=POINTS_PER_BUCKET,
                                         rowsize=3)
    self.all_torque_points = []

  def estimate_params_linear(self):
    points = self.filtered_points.get_points(self.fit_points)
    # total least square solution as both x and y are noisy observations
    # this is empirically the slope of the hysteresis parallelogram as opposed to the line through the diagonals
    try:
      _, _, v = np.linalg.svd(points, full_matrices=False)
      slope, offset = -v.T[0:2, 2] / v.T[2, 2]
      _, spread = np.matmul(points[:, [0, 2]], slope2rot(slope)).T
      friction_coeff = np.std(spread) * FRICTION_FACTOR
    except np.linalg.LinAlgError as e:
      cloudlog.exception(f"Error computing live torque params: {e}")
      slope = offset = friction_coeff = np.nan
    return slope, offset, friction_coeff


  def estimate_params(self):
    """
    Fit the 4-parameter steering-torque curve and extract
    a single static-friction amplitude (sigma_f).

    Returns (a, b, c, d, sigma_f) or (None, …) on failure.
    """
    # ── 1. gather data ──────────────────────────────────────────
    pts = self.filtered_points.get_points(self.fit_points)
    if pts.size == 0:
        cloudlog.info("No points to fit.")
        return (np.nan,)*5

    try:
      _, _, v = np.linalg.svd(pts, full_matrices=False)
      slope, _ = -v.T[0:2, 2] / v.T[2, 2]
      _, spread = np.matmul(pts[:, [0, 2]], slope2rot(slope)).T
      friction_coeff = np.std(spread) * FRICTION_FACTOR
    except np.linalg.LinAlgError as e:
      cloudlog.exception(f"Error computing live torque params: {e}")
      friction_coeff = np.nan

    x = pts[:, 2].astype(float)   # lateral acceleration
    y = pts[:, 0].astype(float)   # steering torque

    # ── 2. Gauss-Newton / LM fit for (a,b,c,d) ─────────────────
    # paste this into Desmos to visualize the fit:
    """
      S_{sigmoid}\left(x\right)=\frac{1}{\left(1+e^{-x}\right)}-.5
      f\left(x\right)\ =\ S_{sigmoid}\left(x\cdot a_{sigmoidSharpness}\right)\cdot b_{sigmoidTorqueGain}+\left(x\cdot c_{latAccelFactor}\right)
      a_{sigmoidSharpness}=3
      b_{sigmoidTorqueGain}=1
      c_{latAccelFactor}=.1
    """

    b0     = np.clip(np.ptp(y), 0.1, 2.0)
    params = np.array([3.0, b0, 0.0, 0.0])           # [a,b,c,d]
    lam, tol, it_max = 1e-3, 1e-5, 50

    for it in range(it_max):
        a, b, c, d = params
        r  = model(x, a, b, c, d) - y
        J  = jacobian(x, a, b, c, d)
        H  = J.T @ J
        g  = J.T @ r
        delta = np.linalg.solve(H + lam*np.eye(4), -g)

        if not np.all(np.isfinite(delta)):
            cloudlog.warning("Non-finite GN step – aborting")
            return (None,)*5

        params_new = params + delta
        # bounds
        params_new[0] = np.clip(params_new[0], 0.0, 10.0)   # a
        params_new[1] = np.clip(params_new[1], 0.0, 2.0)    # b
        params_new[2] = np.clip(params_new[2], 0.0, 5.0)    # c
        params_new[3] = np.clip(params_new[3], -.3, 0.3)    # d

        if np.max(np.abs(delta)) < tol:
            params = params_new
            break
        params = params_new

    a, b, c, d = params
    if not np.all(np.isfinite(params)):
        cloudlog.warning("Invalid parameters after GN fit")
        return (None,)*5

    # ── 3.  friction estimate from residual envelope ───────────
    # resid = y - model(x, a, b, c, d)

    # # bin residuals vs x to get local σ(x)
    # bins = 40
    # idx  = np.argsort(x)
    # x_sorted, r_sorted = x[idx], resid[idx]
    # edges   = np.linspace(x.min(), x.max(), bins + 1)
    # centers = 0.5 * (edges[1:] + edges[:-1])
    # sigmas  = np.array([
    #     np.std(r_sorted[(x_sorted >= lo) & (x_sorted < hi)])
    #     for lo, hi in zip(edges[:-1], edges[1:])
    # ])

    # tail_sigma  = sigmas[[0, -1]].mean()                # baseline noise
    # peak_sigma  = sigmas[np.argmin(np.abs(centers))]    # widest point
    # sigma_f     = max(peak_sigma - tail_sigma, 0.0) * FRICTION_FACTOR

    cloudlog.info(
        f"GN fit {it+1:02d} iters: "
        f"a={a:.4f}  b={b:.4f}  c={c:.4f}  d={d:.4f}  σ_f={friction_coeff:.4f}"
    )
    print(f"GN fit {it+1:02d} iters: a={a:.4f}  b={b:.4f}  c={c:.4f}  d={d:.4f}  σ_f={friction_coeff:.4f}")

    self.nonlinear_params = np.array([a, b, c, d])
    self.friction_coeff   = friction_coeff
    return a, b, c, d, friction_coeff

  def update_params(self, params):
    self.decay = min(self.decay + DT_CTRL, MAX_FILTER_DECAY)
    for param, value in params.items():
      self.filtered_params[param].update(value)
      self.filtered_params[param].update_alpha(self.decay)

  def handle_log(self, t, which, msg):
    if which == "carControl":
      self.raw_points["carControl_t"].append(t + self.lag)
      self.raw_points["lat_active"].append(msg.latActive)
    elif which == "carOutput":
      self.raw_points["carOutput_t"].append(t + self.lag)
      self.raw_points["steer_torque"].append(-msg.actuatorsOutput.torque)
    elif which == "carState":
      self.raw_points["carState_t"].append(t + self.lag)
      # TODO: check if high aEgo affects resulting lateral accel
      self.raw_points["vego"].append(msg.vEgo)
      self.raw_points["steer_override"].append(msg.steeringPressed)
      self.raw_points["steer_angle"].append(msg.steeringAngleDeg)
    elif which == "liveCalibration":
      self.calibrator.feed_live_calib(msg)
    elif which == "liveDelay":
      self.lag = msg.lateralDelay
    # calculate lateral accel from past steering torque
    elif which == "livePose":
      if len(self.raw_points['steer_torque']) == self.hist_len:
        device_pose = Pose.from_live_pose(msg)
        calibrated_pose = self.calibrator.build_calibrated_pose(device_pose)
        angular_velocity_calibrated = calibrated_pose.angular_velocity


        yaw_rate = angular_velocity_calibrated.yaw
        roll = device_pose.orientation.roll
        # check lat active up to now (without lag compensation)
        lat_active = np.interp(np.arange(t - MIN_ENGAGE_BUFFER, t + self.lag, DT_CTRL),
                               self.raw_points['carControl_t'], self.raw_points['lat_active']).astype(bool)
        steer_override = np.interp(np.arange(t - MIN_ENGAGE_BUFFER, t + self.lag, DT_CTRL),
                                   self.raw_points['carState_t'], self.raw_points['steer_override']).astype(bool)
        vego = np.interp(t, self.raw_points['carState_t'], self.raw_points['vego'])
        steer = np.interp(t, self.raw_points['carOutput_t'], self.raw_points['steer_torque']).item()
        lateral_acc = (vego * yaw_rate) - (np.sin(roll) * ACCELERATION_DUE_TO_GRAVITY).item()
        # average the steering torque over the last .25 seconds and add it to the filtered points instead of the raw point. the buffer has 500 samples at 100hz
        LOOKBACK = 0.5
        lookback_t = np.arange(t - LOOKBACK, t + self.lag, DT_CTRL)

        # avg_steer = np.mean(np.interp(lookback_t,
        #                               self.raw_points['carOutput_t'], self.raw_points['steer_torque']))
        steering_angle_std = np.std(np.interp(lookback_t,
                                               self.raw_points['carState_t'], self.raw_points['steer_angle']))
        # steering_angle = np.interp(t, self.raw_points['carState_t'], self.raw_points['steer_angle']).item()


        #print(avg_steer, steer)
        #print(self.raw_points['carControl_t'])
        if all(lat_active) and not any(steer_override) and (vego > MIN_VEL) and (abs(steer) > STEER_MIN_THRESHOLD) and steering_angle_std < 1.0:
          if abs(lateral_acc) <= LAT_ACC_THRESHOLD:
            #print(f"t: {t:.4f}, lateral_acc: {lateral_acc:.4f}, steer: {avg_steer:.4f}, steer_angle: {steering_angle:.4f}, steer_angle_std: {steering_angle_std:.4f}")
            #self.filtered_points.add_point(avg_steer, lateral_acc)
            self.filtered_points.add_point(steer, lateral_acc)

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
      latAccelFactor, latAccelOffset, frictionCoeff = self.estimate_params()
      liveTorqueParameters.latAccelFactorRaw = float(latAccelFactor)
      liveTorqueParameters.latAccelOffsetRaw = float(latAccelOffset)
      liveTorqueParameters.frictionCoefficientRaw = float(frictionCoeff)

      if self.filtered_points.is_valid():
        if any(val is None or np.isnan(val) for val in [latAccelFactor, latAccelOffset, frictionCoeff]):
          cloudlog.exception("Live torque parameters are invalid.")
          liveTorqueParameters.liveValid = False
          self.reset()
        else:
          liveTorqueParameters.liveValid = True
          latAccelFactor = np.clip(latAccelFactor, self.min_lataccel_factor, self.max_lataccel_factor)
          frictionCoeff = np.clip(frictionCoeff, self.min_friction, self.max_friction)
          self.update_params({'latAccelFactor': latAccelFactor, 'latAccelOffset': latAccelOffset, 'frictionCoefficient': frictionCoeff})

    if with_points:
      liveTorqueParameters.points = self.filtered_points.get_points()[:, [0, 2]].tolist()

    liveTorqueParameters.latAccelFactorFiltered = float(self.filtered_params['latAccelFactor'].x)
    liveTorqueParameters.latAccelOffsetFiltered = float(self.filtered_params['latAccelOffset'].x)
    liveTorqueParameters.frictionCoefficientFiltered = float(self.filtered_params['frictionCoefficient'].x)
    liveTorqueParameters.totalBucketPoints = len(self.filtered_points)
    liveTorqueParameters.decay = self.decay
    liveTorqueParameters.maxResets = self.resets
    return msg

  def pre_load_points(self, initial_tune) -> None:
    """
    Seed the buckets with synthetic points built from the initial tune.

    Parameters
    ----------
    initial_tune : tuple[float, float, float, float]
      (a, b, c, d) sigmoid parameters,
      a: 'steepness' of the curve\n
      b: 'max' torque\n
      c: linear slope offset\n
      d: horizonital offset in the torque curve\n
    """

    a, b, c, d, friction = initial_tune
    d = 0.0  # d is the offset in the torque curve

    def sig_centered(z):
      """ σ(z) − 0.5 in a numerically stable form. """
      pos = 1.0 / (1.0 + np.exp(-z)) - 0.5
      neg = np.exp(z) / (1.0 + np.exp(z)) - 0.5
      return np.where(z >= 0.0, pos, neg)

    def torque(lat_acc):
      xs = lat_acc - d
      return sig_centered(a * xs) * b + c * xs   # note: NO “-0.5·b”

    # -------------------------------------------------------------
    rng = np.random.default_rng(42)
    x_sample = rng.uniform(-4, 4, 20_000)

    # Sensor noise (horizontal jitter)
    lat_accel_jitter = x_sample + rng.normal(0, 0.10, size=x_sample.shape)

    # Torque noise: baseline + friction-shaped bump at centre
    sigma_base     = 0.10
    sigma_friction = friction
    envelope       = np.exp(-(lat_accel_jitter / 1.0) ** 2)

    steer_jitter = (
        torque(lat_accel_jitter)
        + rng.normal(0, sigma_base, size=x_sample.shape)
        + rng.normal(0, sigma_friction * envelope, size=x_sample.shape)
    )

    # Add synthetic points to the buckets
    for τ, a_lat in zip(steer_jitter, lat_accel_jitter):
        self.filtered_points.add_point(τ, a_lat)


  def save_filtered_points(self, base_filename="bucket_plot", file_ext=".png"):
    all_points = []  # Collect all bucket points for the combined plot

    # Iterate over each bucket in the filtered_points object
    for bounds in self.filtered_points.x_bounds:
      # Get the data for the current bucket. Each bucket is expected to be a list of points.
      bucket_data = self.filtered_points.buckets.get(bounds, [])

      # Check if the bucket has any data
      if not bucket_data:
        print(f"No data points in bucket {bounds}")
        continue

      # Convert bucket data to a numpy array for processing
      bucket_points = np.array(bucket_data.arr)
      if bucket_points.size == 0:
        print(f"No data points in bucket {bounds}")
        continue

      # Append these points to all_points for the combined plot
      all_points.append(bucket_points)


    # Create one combined plot if there are any points
    if all_points:
        combined = np.concatenate(all_points, axis=0)
        steer_all   = combined[:, 0]
        lateral_all = combined[:, 2]

        # ── figure ───────────────────────────────────────────────
        plt.figure(figsize=(10, 6))
        plt.scatter(lateral_all, steer_all, s=8, alpha=0.4, label="Filtered samples")

        # fitted curve + friction band
        a, b, c, d = self.nonlinear_params          # 4-tuple
        sigma_f    = getattr(self, "friction_coeff", 0.0)
        sigma_f_op = op_friction(self.filtered_points.get_points(self.fit_points))
        print(f"OP friction: {sigma_f_op:.4f}")

        x_line = np.linspace(-4, 4, 400)
        y_fit  = model(x_line, a, b, c, d)

        plt.plot(x_line, y_fit,          color="red",  lw=2, label="Fitted curve")
        if sigma_f > 0:
            # plt.plot(x_line, y_fit + sigma_f, color="red", ls="--", lw=1.5, label="+σ_f")
            # plt.plot(x_line, y_fit - sigma_f, color="red", ls="--", lw=1.5, label="−σ_f")
            plt.plot(x_line, y_fit + sigma_f_op, color="blue", ls="--", lw=1.5, label="+OP_σ_f")
            plt.plot(x_line, y_fit - sigma_f_op, color="blue", ls="--", lw=1.5, label="−OP_σ_f")

        # ── cosmetics ────────────────────────────────────────────
        plt.xlim(-4, 4)
        plt.ylim(-1, 1)
        plt.xlabel("Lateral acceleration (m/s²)")
        plt.ylabel("Steering torque (Nm equiv)")
        plt.title("Torque vs lateral acceleration (all buckets)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename_all = f"{base_filename}_all{file_ext}"
        plt.savefig(filename_all)
        plt.close()
        print(f"Combined plot saved as {filename_all}")

  def save_points(self, filename="torque_estimator.pkl"):
    # Save both raw_points and filtered_points (buckets) as plain data
    data = {
      'raw_points': {k: list(v) for k, v in dict(self.raw_points).items()},
      'filtered_points': {k: v for k, v in self.filtered_points.buckets.items()}
    }
    with open(filename, "wb") as f:
      pickle.dump(data, f)
    print(f"Estimator saved to {filename}")

  def load_points(self, filename="torque_estimator.pkl"):
    with open(filename, "rb") as f:
      data = pickle.load(f)
    # Reconstruct raw_points as a defaultdict with deques
    self.raw_points = defaultdict(lambda: deque(maxlen=self.hist_len))
    for key, value in data.get('raw_points', {}).items():
      self.raw_points[key] = deque(value, maxlen=self.hist_len)
    # Reconstruct filtered_points buckets directly
    if 'filtered_points' in data:
      self.filtered_points.buckets = data['filtered_points']
    print(f"Points loaded from {filename}")



def main(demo=False):
  config_realtime_process([0, 1, 2, 3], 5)

  pm = messaging.PubMaster(['liveTorqueParameters'])
  sm = messaging.SubMaster(['carControl', 'carOutput', 'carState', 'liveCalibration', 'livePose', 'liveDelay'], poll='livePose')

  params = Params()

  if demo:
    import time
    # benchmark
    start_time = time.time()
    estimator = TorqueEstimator(messaging.log_from_bytes(params.get("CarParamsPrevRoute", block=True), car.CarParams))
    print("Time taken to create TorqueEstimator:", time.time() - start_time)
    step_time = time.time()
    estimator.pre_load_points(NON_LINEAR_TORQUE_PARAMS['MAZDA_3_2019'])
    print("Time taken to pre-load points:", time.time() - step_time)
    step_time = time.time()
    estimator.estimate_params()
    print("Time taken to estimate params:", time.time() - step_time)
    exit(0)

  msg_filter = ['carControl', 'carOutput', 'carState', 'liveCalibration', 'livePose', 'liveDelay', 'carParams']

  #estimator = TorqueEstimator(messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams))
  from openpilot.tools.lib.logreader import LogReader
  lr = LogReader("3b58edf884ab4eaf|00000012--a994169ecb/0:", sort_by_time=True)
  estimator = None
  for msg in lr:
    t = msg.logMonoTime * 1e-9
    which = msg.which()
    if which not in msg_filter:
      continue
    if estimator is None and msg.which() == "carParams":
      estimator = TorqueEstimator(msg.carParams)
      estimator.pre_load_points(NON_LINEAR_TORQUE_PARAMS['MAZDA_3_2019'])
      estimator.estimate_params()
      estimator.save_filtered_points()

    if estimator:
      estimator.handle_log(t, which, msg._get(which))

  estimator.estimate_params()
  estimator.save_filtered_points()
  #estimator.save_points()

  exit(0)
  while True:
    sm.update()
    if sm.all_checks():
      for which in sm.updated.keys():
        if sm.updated[which]:
          t = sm.logMonoTime[which] * 1e-9
          estimator.handle_log(t, which, sm[which])

    # 4Hz driven by livePose
    if sm.frame % 5 == 0:
      pm.send('liveTorqueParameters', estimator.get_msg(valid=sm.all_checks()))

    # Cache points every 60 seconds while onroad
    if sm.frame % 240 == 0:
      msg = estimator.get_msg(valid=sm.all_checks(), with_points=True)
      params.put_nonblocking("LiveTorqueParameters", msg.to_bytes())


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description='Process the --demo argument.')
  parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
  args = parser.parse_args()
  main(demo=args.demo)
