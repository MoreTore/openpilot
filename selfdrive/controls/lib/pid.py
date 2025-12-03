import numpy as np
from numbers import Number

class PIDController:
  def __init__(self, k_p, k_i, k_f=0., k_d=0., pos_limit=1e308, neg_limit=-1e308, rate=100):
    self._k_p = k_p
    self._k_i = k_i
    self._k_d = k_d
    self.k_f = k_f   # feedforward gain
    if isinstance(self._k_p, Number):
      self._k_p = [[0], [self._k_p]]
    if isinstance(self._k_i, Number):
      self._k_i = [[0], [self._k_i]]
    if isinstance(self._k_d, Number):
      self._k_d = [[0], [self._k_d]]

    self.set_limits(pos_limit, neg_limit)

    self.i_rate = 1.0 / rate
    self.speed = 0.0

    self.reset()

  @property
  def k_p(self):
    return np.interp(self.speed, self._k_p[0], self._k_p[1])

  @property
  def k_i(self):
    return np.interp(self.speed, self._k_i[0], self._k_i[1])

  @property
  def k_d(self):
    return np.interp(self.speed, self._k_d[0], self._k_d[1])

  def reset(self):
    self.p = 0.0
    self.i = 0.0
    self.d = 0.0
    self.f = 0.0
    self.control = 0

  def set_limits(self, pos_limit, neg_limit):
    self.pos_limit = pos_limit
    self.neg_limit = neg_limit

  def update(self, error, error_rate=0.0, speed=0.0, feedforward=0., freeze_integrator=False):
    self.speed = speed
    self.p = float(error) * self.k_p
    self.f = feedforward * self.k_f
    self.d = error_rate * self.k_d

    if not freeze_integrator:
      i_candidate = self.i + error * self.k_i * self.i_rate
    else:
      i_candidate = self.i

    # Unclipped control with candidate integral
    u = self.p + i_candidate + self.d + self.f

    # Saturated output
    u_sat = np.clip(u, self.neg_limit, self.pos_limit)

    # Anti-windup: only integrate if
    # - we're not saturating, OR
    # - the integral change moves us OUT of saturation.
    if u == u_sat:
      # not saturated -> accept integral update
      self.i = i_candidate
    else:
      # saturated high, only allow integral if error < 0 (pull down)
      if u > self.pos_limit and error < 0:
        self.i = i_candidate
      # saturated low, only allow integral if error > 0 (push up)
      elif u < self.neg_limit and error > 0:
        self.i = i_candidate
      # else: reject this integral step (hold self.i)

    # Final control with updated integral
    control = self.p + self.i + self.d + self.f
    self.control = np.clip(control, self.neg_limit, self.pos_limit)
    return self.control
