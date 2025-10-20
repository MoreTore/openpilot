#!/usr/bin/env python3
# Lightweight, high-performance live viewer using pygame.
# Draws torque vs lateral acceleration points, fitted curve, and friction band.

from __future__ import annotations

import math
from dataclasses import dataclass
from collections import deque


@dataclass
class PlotLimits:
  x_min: float = -4.0
  x_max: float = 4.0
  y_min: float = -1.0
  y_max: float = 1.0


class PygamePlotter:
  hist: dict[str, deque[float]]
  def __init__(self, size: tuple[int, int] = (1280, 420), fps: int = 60, grid: bool = True):
    try:
      import pygame
    except Exception as e:
      raise RuntimeError("pygame is required for --pygame viewer. Install with: pip install pygame") from e

    import pygame
    pygame.init()
    pygame.display.set_caption("Torque vs Lateral Acceleration (pygame)")
    self.pygame = pygame
    self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    self.clock = pygame.time.Clock()
    self.font = pygame.font.SysFont("monospace", 16)

    self.grid_enabled = grid
    self.fps = fps
    self.limits = PlotLimits()
    # history buffers (last 100)
    self.hist_len = 100
    self.hist = {
      'a': deque(maxlen=self.hist_len),
      'b': deque(maxlen=self.hist_len),
      'c': deque(maxlen=self.hist_len),
      'd': deque(maxlen=self.hist_len),
      'friction': deque(maxlen=self.hist_len),
    }
    self.show_history = True

    # cached colors
    self.colors = {
      'bg': (18, 18, 18),
      'axes': (200, 200, 200),
      'grid': (48, 48, 48),
      'points': (100, 200, 255),
      'fit': (255, 80, 80),
      'band': (120, 160, 255),
      'manual_fit': (80, 255, 120),
      'textbox_bg': (30, 30, 30),
      'textbox_border': (100, 100, 100),
      'textbox_focus': (200, 200, 80),
      'text': (220, 220, 220),
    }

    # manual editable params (a,b,c,d) and UI state
    self.manual_params = {'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0}
    # the input strings shown in textboxes
    self.manual_inputs = {'a': '0.0', 'b': '0.0', 'c': '0.0', 'd': '0.0'}
    # focused textbox: one of 'a','b','c','d' or None
    self.focused_input: str | None = None
    # last parse status
    self.manual_valid = False

  def close(self):
    try:
      # try to destroy the window and quit pygame
      self.pygame.display.quit()
    except Exception:
      pass
    try:
      self.pygame.quit()
    except Exception:
      pass

  # Coordinate transforms
  def _to_px(self, x: float, y: float) -> tuple[int, int]:
    w, h = self.screen.get_size()
    # padding in pixels
    pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 40
    plot_w = max(1, w - pad_l - pad_r)
    plot_h = max(1, h - pad_t - pad_b)

    # normalize
    xn = (x - self.limits.x_min) / (self.limits.x_max - self.limits.x_min)
    yn = (y - self.limits.y_min) / (self.limits.y_max - self.limits.y_min)
    # transform (invert y for screen coords)
    px = int(pad_l + xn * plot_w)
    py = int(pad_t + (1.0 - yn) * plot_h)
    return px, py

  def _draw_grid(self):
    if not self.grid_enabled:
      return
    pg = self.pygame
    w, h = self.screen.get_size()
    # grid every 0.5 units on x, 0.2 on y
    x_step = 0.5
    y_step = 0.2
    # verticals
    x = math.ceil(self.limits.x_min / x_step) * x_step
    while x <= self.limits.x_max:
      x0, y0 = self._to_px(x, self.limits.y_min)
      x1, y1 = self._to_px(x, self.limits.y_max)
      pg.draw.line(self.screen, self.colors['grid'], (x0, y0), (x1, y1), 1)
      x += x_step
    # horizontals
    y = math.ceil(self.limits.y_min / y_step) * y_step
    while y <= self.limits.y_max:
      x0, y0 = self._to_px(self.limits.x_min, y)
      x1, y1 = self._to_px(self.limits.x_max, y)
      pg.draw.line(self.screen, self.colors['grid'], (x0, y0), (x1, y1), 1)
      y += y_step

  def _draw_axes(self):
    pg = self.pygame
    # x-axis at y=0
    x0, y0 = self._to_px(self.limits.x_min, 0.0)
    x1, y1 = self._to_px(self.limits.x_max, 0.0)
    pg.draw.line(self.screen, self.colors['axes'], (x0, y0), (x1, y1), 1)
    # y-axis at x=0
    x0, y0 = self._to_px(0.0, self.limits.y_min)
    x1, y1 = self._to_px(0.0, self.limits.y_max)
    pg.draw.line(self.screen, self.colors['axes'], (x0, y0), (x1, y1), 1)

  def _draw_points(self, combined_points, max_draw: int = 20000):
    # combined_points: Nx3 -> [steer, 1.0, lateral_acc]
    if combined_points is None or combined_points.size == 0:
      return
    pg = self.pygame
    n = combined_points.shape[0]
    step = max(1, n // max_draw)
    for i in range(0, n, step):
      τ = float(combined_points[i, 0])
      a_lat = float(combined_points[i, 2])
      px, py = self._to_px(a_lat, τ)
      pg.draw.circle(self.screen, self.colors['points'], (px, py), 1)

  def _draw_fit(self, model_fn, params: tuple[float, float, float, float] | None, sigma_f: float):
    if params is None:
      return
    pg = self.pygame
    a, b, c, d = params
    # build polyline for fit
    xs = [self.limits.x_min + i * (self.limits.x_max - self.limits.x_min) / 400 for i in range(401)]
    pts_fit = []
    for x in xs:
      y = float(model_fn(x, a, b, c, d))
      pts_fit.append(self._to_px(x, y))
    if len(pts_fit) >= 2:
      pg.draw.lines(self.screen, self.colors['fit'], False, pts_fit, 2)
    # friction band
    if sigma_f > 0:
      pts_hi = []
      pts_lo = []
      for x in xs:
        y = float(model_fn(x, a, b, c, d))
        pts_hi.append(self._to_px(x, y + sigma_f))
        pts_lo.append(self._to_px(x, y - sigma_f))
      if len(pts_hi) >= 2:
        pg.draw.lines(self.screen, self.colors['band'], False, pts_hi, 1)
      if len(pts_lo) >= 2:
        pg.draw.lines(self.screen, self.colors['band'], False, pts_lo, 1)

  def _draw_text(self, lines):
    x, y = 10, 10
    for line in lines:
      surf = self.font.render(line, True, self.colors['text'])
      self.screen.blit(surf, (x, y))
      y += surf.get_height() + 2
  def handle_input(self) -> bool:
    """Process events. Returns False if window should close."""
    pg = self.pygame
    for event in pg.event.get():
      if event.type == pg.QUIT:
        return False
      if event.type == pg.KEYDOWN:
        if event.key == pg.K_ESCAPE:
          return False
        if event.key == pg.K_g:
          self.grid_enabled = not self.grid_enabled
        if event.key in (pg.K_PLUS, pg.K_EQUALS):
          # zoom in
          self._zoom(0.9)
        if event.key == pg.K_MINUS:
          # zoom out
          self._zoom(1.1)
        # manual param text input handling
        if self.focused_input is not None:
          # Enter applies the value
          if event.key in (pg.K_RETURN, pg.K_KP_ENTER):
            self._apply_focused_input()
            # unfocus after apply
            self.focused_input = None
            continue
          # Tab moves focus to next
          if event.key == pg.K_TAB:
            self._focus_next()
            continue
          # Backspace
          if event.key == pg.K_BACKSPACE:
            s = self.manual_inputs.get(self.focused_input, '')
            self.manual_inputs[self.focused_input] = s[:-1]
            continue
          # Allow printable chars for numbers
          if event.unicode and event.unicode in '0123456789.+-eE':
            self.manual_inputs[self.focused_input] = self.manual_inputs.get(self.focused_input, '') + event.unicode
            continue
        # if no focused input, allow keyboard shortcuts as before
      if event.type == pg.MOUSEBUTTONDOWN:
        # check clicks on textboxes
        if event.button == 1:  # left click
          mx, my = event.pos
          clicked = self._check_textbox_click(mx, my)
          if clicked is not None:
            self.focused_input = clicked
            continue
    return True

  # Textbox helpers for manual params
  def _get_textbox_rects(self) -> dict[str, tuple[int, int, int, int]]:
    # place textboxes at top of the right panel (inside history area)
    rx, ry, rw, rh = self._layout_rect_right()
    # leave a small margin
    pad = 8
    box_h = 28
    gap = 6
    rects: dict[str, tuple[int, int, int, int]] = {}
    keys = ['a', 'b', 'c', 'd']
    for i, k in enumerate(keys):
      x = rx + pad
      y = ry + pad + i * (box_h + gap)
      rects[k] = (x, y, rw - pad * 2, box_h)
    return rects

  def _check_textbox_click(self, mx: int, my: int) -> str | None:
    rects = self._get_textbox_rects()
    for k, (x, y, w, h) in rects.items():
      if x <= mx <= x + w and y <= my <= y + h:
        return k
    return None

  def _apply_focused_input(self) -> None:
    if self.focused_input is None:
      return
    s = self.manual_inputs.get(self.focused_input, '')
    try:
      v = float(s)
      self.manual_params[self.focused_input] = v
      self.manual_valid = True
    except Exception:
      self.manual_valid = False

  def _focus_next(self) -> None:
    order = ['a', 'b', 'c', 'd']
    if self.focused_input is None:
      self.focused_input = order[0]
      return
    try:
      idx = order.index(self.focused_input)
    except ValueError:
      self.focused_input = order[0]
      return
    idx = (idx + 1) % (len(order) + 1)
    if idx >= len(order):
      self.focused_input = None
    else:
      self.focused_input = order[idx]

  def _draw_manual_textboxes(self) -> None:
    pg = self.pygame
    rects = self._get_textbox_rects()
    for k, rect in rects.items():
      x, y, w, h = rect
      # background
      pg.draw.rect(self.screen, self.colors['textbox_bg'], rect, 0)
      # border
      border_color = self.colors['textbox_focus'] if self.focused_input == k else self.colors['textbox_border']
      pg.draw.rect(self.screen, border_color, rect, 1)
      # label and value
      label_surf = self.font.render(f"{k}=", True, self.colors['text'])
      self.screen.blit(label_surf, (x + 4, y + (h - label_surf.get_height()) // 2))
      val = self.manual_inputs.get(k, '')
      val_surf = self.font.render(val, True, self.colors['text'])
      self.screen.blit(val_surf, (x + 28, y + (h - val_surf.get_height()) // 2))
    # validation indicator
    status = "OK" if self.manual_valid else "INVALID"
    status_color = (80, 255, 120) if self.manual_valid else (255, 80, 80)
    st = self.font.render(status, True, status_color)
    # draw at bottom of textbox area
    rx, ry, rw, rh = self._layout_rect_right()
    self.screen.blit(st, (rx + 6, ry + rh - st.get_height() - 6))

  def _zoom(self, factor: float):
    # zoom around origin
    self.limits.x_min *= factor
    self.limits.x_max *= factor
    self.limits.y_min *= factor
    self.limits.y_max *= factor

  def draw_frame(self, combined_points, model_fn, params: tuple[float, float, float, float] | None,
                 sigma_f: float, info_lines: list[str]) -> bool:
    """Draw one frame. Returns False if the window should close."""
    if not self.handle_input():
      return False
    self.screen.fill(self.colors['bg'])
    self._draw_grid()
    self._draw_axes()
    # Main plot on the left area (leave space for history panel if enabled)
    main_rect = self._layout_rect_left()
    self._draw_points_in_rect(combined_points, main_rect)
    # draw fitted model (existing)
    self._draw_fit_in_rect(model_fn, params, sigma_f, main_rect)
    # draw manual model if valid
    if self.manual_valid:
      manual_params_tuple = (self.manual_params['a'], self.manual_params['b'], self.manual_params['c'], self.manual_params['d'])
      # reuse _draw_fit_in_rect but draw with manual color by temporarily swapping colors
      old_fit = self.colors['fit']
      self.colors['fit'] = self.colors['manual_fit']
      self._draw_fit_in_rect(model_fn, manual_params_tuple, sigma_f, main_rect)
      self.colors['fit'] = old_fit
    self._draw_text(info_lines)
    if self.show_history:
      self._draw_history_panel()
      # draw textboxes for manual params in history panel area
      self._draw_manual_textboxes()
    self.pygame.display.flip()
    self.clock.tick(self.fps)
    return True

  # ── API ─────────────────────────────────────────────────────────
  def record_params(self, a: float, b: float, c: float, d: float, friction: float):
    self.hist['a'].append(float(a))
    self.hist['b'].append(float(b))
    self.hist['c'].append(float(c))
    self.hist['d'].append(float(d))
    self.hist['friction'].append(float(friction))

  # ── Layout helpers ─────────────────────────────────────────────
  def _layout_rect_left(self) -> tuple[int, int, int, int]:
    w, h = self.screen.get_size()
    right_w = int(w * 0.33) if self.show_history else 0
    return (0, 0, w - right_w, h)

  def _layout_rect_right(self) -> tuple[int, int, int, int]:
    w, h = self.screen.get_size()
    right_w = int(w * 0.33)
    return (w - right_w, 0, right_w, h)

  # ── Drawing in rects ───────────────────────────────────────────
  def _draw_points_in_rect(self, combined_points, rect: tuple[int, int, int, int]):
    if combined_points is None or combined_points.size == 0:
      return
    pg = self.pygame
    x, y, w, h = rect
    # draw border
    pg.draw.rect(self.screen, (60, 60, 60), rect, 1)
    # transform using current limits but map to rect
    n = combined_points.shape[0]
    step = max(1, n // 20000)
    for i in range(0, n, step):
      τ = float(combined_points[i, 0])
      a_lat = float(combined_points[i, 2])
      px, py = self._to_px(a_lat, τ)
      # remap to rect (since _to_px uses full surface, adjust offset)
      px = int(px * (w / self.screen.get_width()) + x)
      py = int(py * (h / self.screen.get_height()) + y)
      pg.draw.circle(self.screen, self.colors['points'], (px, py), 1)

  def _draw_fit_in_rect(self, model_fn, params, sigma_f: float, rect: tuple[int, int, int, int]):
    if params is None:
      return
    pg = self.pygame
    x, y, w, h = rect
    # border drawn in points function
    a, b, c, d = params
    xs = [self.limits.x_min + i * (self.limits.x_max - self.limits.x_min) / 400 for i in range(401)]
    pts_fit = []
    pts_hi = []
    pts_lo = []
    for xv in xs:
      yv = float(model_fn(xv, a, b, c, d))
      px, py = self._to_px(xv, yv)
      px = int(px * (w / self.screen.get_width()) + x)
      py = int(py * (h / self.screen.get_height()) + y)
      pts_fit.append((px, py))
      if sigma_f > 0:
        pxh, pyh = self._to_px(xv, yv + sigma_f)
        pxl, pyl = self._to_px(xv, yv - sigma_f)
        pxh = int(pxh * (w / self.screen.get_width()) + x)
        pyh = int(pyh * (h / self.screen.get_height()) + y)
        pxl = int(pxl * (w / self.screen.get_width()) + x)
        pyl = int(pyl * (h / self.screen.get_height()) + y)
        pts_hi.append((pxh, pyh))
        pts_lo.append((pxl, pyl))
    if len(pts_fit) >= 2:
      pg.draw.lines(self.screen, self.colors['fit'], False, pts_fit, 2)
    if sigma_f > 0 and len(pts_hi) >= 2 and len(pts_lo) >= 2:
      pg.draw.lines(self.screen, self.colors['band'], False, pts_hi, 1)
      pg.draw.lines(self.screen, self.colors['band'], False, pts_lo, 1)

  # ── History panel ─────────────────────────────────────────────
  def _draw_history_panel(self):
    pg = self.pygame
    rect = self._layout_rect_right()
    pg.draw.rect(self.screen, (60, 60, 60), rect, 1)
    x, y, w, h = rect
    # 5 rows
    rows = ['a', 'b', 'c', 'd', 'friction']
    row_h = max(1, h // len(rows))
    for i, key in enumerate(rows):
      r_rect = (x + 6, y + i * row_h + 6, w - 12, row_h - 12)
      self._draw_history_graph(r_rect, key)

  def _draw_history_graph(self, rect: tuple[int, int, int, int], key: str):
    pg = self.pygame
    x, y, w, h = rect
    pg.draw.rect(self.screen, (40, 40, 40), rect, 0)
    pg.draw.rect(self.screen, (80, 80, 80), rect, 1)
    series = list(self.hist[key])
    if len(series) < 2:
      # label
      label = self.font.render(key, True, self.colors['text'])
      self.screen.blit(label, (x + 4, y + 4))
      return
    vmin = min(series)
    vmax = max(series)
    if math.isclose(vmax, vmin):
      pad = max(1e-6, abs(vmax) * 0.05 + 1e-6)
      vmin -= pad
      vmax += pad
    # build points
    pts = []
    for i, v in enumerate(series[-self.hist_len:]):
      xn = i / (self.hist_len - 1 if self.hist_len > 1 else 1)
      yn = (v - vmin) / (vmax - vmin)
      px = int(x + xn * (w - 1))
      py = int(y + (1.0 - yn) * (h - 1))
      pts.append((px, py))
    if len(pts) >= 2:
      pg.draw.lines(self.screen, self.colors['points'], False, pts, 2)
    # label + current value
    label = self.font.render(f"{key}: {series[-1]:.3f}", True, self.colors['text'])
    self.screen.blit(label, (x + 6, y + 6))
