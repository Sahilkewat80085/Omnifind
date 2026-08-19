#!/usr/bin/env python3
"""
LKA Perception — Standalone Video-File Version (Windows, no ROS)
==================================================================
Ported from the Phase 2 CarMaker/ROS2 sim node for real-vehicle bring-up.
Same pipeline, same math — only the I/O layer changed:

  ROS2 sim node                      ->  This version
  --------------------------------------------------------------
  /RGBImage + /InertialData subs     ->  cv2.VideoCapture(video file)
  Live IMU pitch (rclpy Pose msg)    ->  static --pitch flag, or optional
                                          --pitch-csv (frame_idx,pitch_deg)
  /LKA/visualization publisher       ->  annotated .mp4 (--save-video)
  /LKA/bev publisher                 ->  annotated .mp4 (--save-bev-video)
  /LKA/lane_output publisher         ->  CSV row per frame (always written)

Pipeline (unchanged from sim):
  frame -> Dynamic BEV (homography cached, recomputed only if pitch
           changes > 0.5 deg) -> white-pixel segmentation on BEV ->
           RANSAC 2nd-degree poly fit (per lane) -> Kalman filter on
           poly coefficients -> EMA on curvature/lateral_offset ->
           lateral_offset, curvature, relative_yaw, confidence

REQUIRES RECALIBRATION FOR YOUR REAL CAMERA:
  --cam-height, --fov, --pitch, and the white-mask thresholds
  (--white-v-min / --white-s-max) were tuned for CarMaker's synthetic
  lane textures. Real footage will very likely need different white
  thresholds — run with --tune to get a live trackbar UI for that.

TUNED VARIANT — retuned against lane_2.mp4 (1920x1080 real footage,
painted white lines on mottled paver/brick ground, fisheye-ish wide lens):
  - white_v_min/white_s_max raised from 165/35 to 190/24. The default
    thresholds passed ~8% of BEV pixels as "white" on this footage (paver
    highlights, not lane paint), which fed RANSAC/the histogram-peak
    picker noise and caused intermittent lost-detection streaks (up to
    17 frames) and lane-line jumps as large as 1.4m frame-to-frame.
    190/24 isolates just the painted lines (<2% of BEV pixels) with zero
    detection dropouts across the genuine-lane portion of the clip.
  - lateral_offset_alpha 0.30->0.15, curvature_alpha 0.10->0.08: heavier
    EMA smoothing. Combined with the cleaner mask this took the p95
    frame-to-frame lateral-offset jump from ~0.10m to ~0.04m and
    eliminated jumps >0.1m entirely (was ~5-13% of frames).
  - RANSACLaneFitter now uses its own seeded np.random.RandomState
    instead of the global numpy RNG, so identical input no longer
    produces slightly different fits on different runs (verified: two
    back-to-back runs on the same clip previously disagreed on
    max-frame-jump by ~10%; a fixed per-fitter seed removes that
    non-determinism, which was an extra source of visual flicker on
    top of genuine motion).
  - Added MAX_WHITE_FRAC guard in detect_lane_pixels: if the mask ever
    covers more than 6% of the BEV (never happens on real lane pixels
    in this clip, peak ~1.8%), the frame is treated as "no detection"
    instead of fitting garbage. lane_2.mp4's last ~180 frames drive past
    the painted lines into a white-walled corridor, at which point the
    whole BEV floods white and the old code would fit RANSAC lines to
    building walls (offset jumps as large as 2.5m). The guard makes the
    engine gracefully report "no lane" there instead of hallucinating one.
  See tune_sweep.py in the repo root to reproduce/redo this parameter
  sweep against a different clip.

PATCHED — investigating "lanes merging" reported starting ~t=26s on
lane_1.mp4. Direct instrumentation against that clip found two distinct,
confirmed causes (not the originally-suspected BEV_Y_MAX geometry, which
checks out fine — diagnose_coverage confirms the full 2-8m range is
genuinely visible):

  1. t=25.2s-30.9s: a genuine ~5.7s total detection dropout. white_mask
     coverage jumps from ~3% to 7-14% right at t=25.23s, tripping the
     existing MAX_WHITE_FRAC=0.06 guard (correctly — this is the same
     "flooded BEV" scenario the guard was built for on lane_2.mp4, just
     happening on this clip too). Scene brightness stays flat (~85 mean,
     no glare/exposure spike) across this window, so it's a genuinely
     white/bright SURFACE entering frame, not a lighting artifact — worth
     checking what's physically there in the raw footage around t=25s.
  2. t=22-25s (before the dropout): lateral_offset already swings wildly
     (tens of meters) frame-to-frame, including in frames with healthy
     point counts and reasonable inlier spread. A minimum-inlier-span gate
     (MIN_INLIER_SPAN_M, added below) helps but does not fully explain
     it — some "healthy-looking" fits are still wrong, most likely because
     the histogram-peak left/right picker is intermittently locking onto
     a paver joint/seam instead of the painted line in this section. Use
     --debug-clusters to see the raw left/right point selection and
     pre-Kalman fit directly and confirm on your own footage.

Install:
    pip install opencv-python numpy

Usage:
    python lka_perception_video.py --video clip.mp4
    python lka_perception_video.py --video clip.mp4 --headless --save-video
    python lka_perception_video.py --video clip.mp4 --tune
    python lka_perception_video.py --video clip.mp4 --debug-clusters
    python lka_perception_video.py --video 0                    # webcam
    python lka_perception_video.py --video clip.mp4 --cam-height 1.2 --fov 90

Every run prints a coverage diagnostic: how close to the vehicle the current
cam-height/fov/pitch actually lets the camera see the ground. If it reports
a gap (visible ground starts farther out than --bev-y-min), the near rows
of the BEV are black/unmapped -> no lane pixels there -> lateral_offset
(evaluated at the vehicle row) gets extrapolated past real data and can be
badly wrong. Re-run with --bev-y-min set to the reported value, or fix
--cam-height/--fov/--pitch if that visible-from distance looks wrong for
what you can actually see in the raw footage.
"""

import argparse
import csv
import math
import os
import time
import warnings
from datetime import datetime

import cv2
import numpy as np

# np.RankWarning moved under np.exceptions in numpy>=2.0 and was removed
# from the top-level namespace; handle both so this runs on either version.
try:
    _rank_warning = np.RankWarning
except AttributeError:
    _rank_warning = np.exceptions.RankWarning
warnings.filterwarnings('ignore', category=_rank_warning)

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION DEFAULTS — override via CLI, retune for the real camera
# ═══════════════════════════════════════════════════════════════════════
CAM_HEIGHT_DEFAULT    = 1.4      # meters, camera height above ground
CAM_FOV_DEG_DEFAULT   = 120.0     # horizontal FOV, degrees
CAM_PITCH_DEG_DEFAULT = 0.0      # static mounting pitch offset, degrees
LANE_WIDTH_M_DEFAULT  = 3.0

# BEV output raster — fixed regardless of input camera resolution
BEV_W = 600
BEV_H = 800
BEV_X_RANGE = 8.0        # meters lateral, total (+-4m from center)
BEV_Y_MIN   = 2.0        # meters ahead (near edge of BEV)
BEV_Y_MAX   = 8.0       # meters ahead (far edge of BEV)
PREVIEW_DISTANCE_M = 2.0

WHITE_V_MIN_DEFAULT = 206
WHITE_S_MAX_DEFAULT = 30
MAX_WHITE_FRAC = 0.06   # reject mask as noise/scene-mismatch above this BEV coverage
MIN_INLIER_SPAN_M = 3.0  # require inliers to span at least this much real depth
                          # before trusting a 2nd-order fit — a narrow cluster
                          # can still clear RANSAC_MIN_INLIERS on point-count
                          # alone but its curvature term is essentially
                          # unconstrained outside that narrow band, and
                          # extrapolating it across the full BEV depth is
                          # what swings the drawn line wildly / makes it
                          # visually cross the other lane. See PATCHED note
                          # above — this helps but is not the full story for
                          # the t=22-25s instability on lane_1.mp4.

# How many consecutive frames to coast on the Kalman filter's own predicted
# state (zero-order-hold on the last known a/b/c) before conceding total
# loss. Short real-world dropouts (glare, a paver joint, a brief occlusion)
# are extremely common and a 1+ second blackout of "no lane at all" is a
# worse outcome for a real controller than coasting the last known
# trajectory for that same second — as long as the confidence signal
# reflects that staleness so the controller can react (slow down, reduce
# steering aggressiveness) rather than trusting a coasted estimate blindly.
MAX_PREDICT_ONLY_FRAMES = 45   # 1.5s @ 30fps
COAST_CONFIDENCE_FRAMES = MAX_PREDICT_ONLY_FRAMES  # confidence reaches 0 at the same point coasting gives up

# Physical curvature bound, derived from dynamic_tracks.pdf (the actual
# double-lane-change competition track): corridor alternates 6m-wide and
# 3m-wide sections -> lateral centerline shift = 6-3 = 3m, over transition
# lengths of 15-22m depending on segment (tightest = 15m). Modeling each
# transition as a half-cosine ramp, max curvature = (shift/2)*(pi/L)^2:
#   L=15m (tightest on the real track) -> kappa=0.0658 1/m, R_min=15.2m
#   L=22m (widest transition)          -> kappa=0.0306 1/m, R_min=32.7m
# No legitimate track feature ever demands a turn radius under ~15m. A bad
# RANSAC fit (wrong-feature lock-on, narrow-cluster extrapolation) can
# easily imply curvature corresponding to R<1m — physically impossible for
# this vehicle on this track. 2x margin above the tightest real requirement
# comfortably separates "genuinely sharp real corner" from "sensor garbage"
# without ever clipping an actual track feature.
MAX_PHYSICAL_CURVATURE = 2.0 * 0.0658  # ~0.132 1/m -> R_min ~7.6m clamp floor

# A fit can pass MIN_INLIER_SPAN_M (span >= 3m) while sitting entirely in
# the far field — e.g. inliers only from y=5m to y=8m — with zero real
# support anywhere near the vehicle. compute_lane_output still evaluates
# the polynomial at y_vehicle (the near edge) regardless, so this produces
# an inflated lateral_offset even with curvature now bounded: the offset
# is a position extrapolated across a multi-meter gap from the nearest
# real data, not a measurement. Require the nearest inlier point to be
# within this distance of BEV_Y_MIN before trusting the near-field
# evaluation on that side.
MAX_EXTRAPOLATION_GAP_M = 2.0

# How far beyond the BEV frame edges an extrapolated vehicle-row position
# is still allowed to fall, as a fraction of BEV_W. 0.5 = allow up to half
# a frame-width of overshoot either side (generous, for genuinely
# off-center detections) but reject anything landing further off the
# visible ground plane than that — see _reject_if_implausible_curvature.
MAX_OFF_FRAME_MARGIN_FRAC = 0.5

RANSAC_ITERS       = 40
RANSAC_THRESH      = 5.0
RANSAC_MIN_INLIERS = 15


# ═══════════════════════════════════════════════════════════════════════
# DynamicBEV — unchanged from the sim node
# ═══════════════════════════════════════════════════════════════════════
class DynamicBEV:
    """
    Computes Bird's Eye View using camera intrinsics and live pitch.
    No hardcoded 4-point transform — pure geometry.
    Homography is cached and only recomputed when pitch changes > 0.5 deg.
    """

    def __init__(self, cam_height, cam_fov_deg, img_w, img_h,
                 bev_w, bev_h, bev_x_range, bev_y_min, bev_y_max,
                 static_pitch_deg=0.0):
        self.cam_height = cam_height
        self.img_w = img_w
        self.img_h = img_h
        self.bev_w = bev_w
        self.bev_h = bev_h
        self.bev_x_range = bev_x_range
        self.bev_y_min = bev_y_min
        self.bev_y_max = bev_y_max
        self.static_pitch = math.radians(static_pitch_deg)

        # Camera intrinsics
        fov_h_rad = math.radians(cam_fov_deg / 2.0)
        self.fx = (img_w / 2.0) / math.tan(fov_h_rad)
        self.fy = self.fx  # square pixels
        self.cx = img_w / 2.0
        self.cy = img_h / 2.0

        # Precompute BEV grid (world coordinates for each BEV pixel)
        bev_u = np.arange(bev_w)
        bev_v = np.arange(bev_h)
        bev_uu, bev_vv = np.meshgrid(bev_u, bev_v)

        self.world_x = (bev_uu - bev_w / 2.0) / bev_w * bev_x_range
        self.world_y = bev_y_max - (bev_vv / bev_h) * (bev_y_max - bev_y_min)

        self.m_per_px_x = bev_x_range / bev_w
        self.m_per_px_y = (bev_y_max - bev_y_min) / bev_h

        # Homography cache
        self._cached_pitch = None
        self._cached_img_u = None
        self._cached_img_v = None
        self._cached_valid = None
        self._pitch_cache_thresh = math.radians(0.5)

    def compute_homography(self, pitch_rad=0.0):
        total_pitch = self.static_pitch + pitch_rad
        cos_p = math.cos(total_pitch)
        sin_p = math.sin(total_pitch)

        x_cam = -self.world_x
        y_cam = self.cam_height * cos_p - self.world_y * sin_p
        z_cam = self.cam_height * sin_p + self.world_y * cos_p

        valid = z_cam > 0.5
        img_u = np.zeros_like(x_cam)
        img_v = np.zeros_like(y_cam)

        img_u[valid] = self.fx * x_cam[valid] / z_cam[valid] + self.cx
        img_v[valid] = self.fy * y_cam[valid] / z_cam[valid] + self.cy

        valid &= (img_u >= 0) & (img_u < self.img_w - 1)
        valid &= (img_v >= 0) & (img_v < self.img_h - 1)

        return img_u.astype(np.float32), img_v.astype(np.float32), valid

    def warp(self, img, pitch_rad=0.0):
        """Warp camera image to BEV using live pitch, with homography caching."""
        if (self._cached_pitch is None or
                abs(pitch_rad - self._cached_pitch) > self._pitch_cache_thresh):
            self._cached_img_u, self._cached_img_v, self._cached_valid = \
                self.compute_homography(pitch_rad)
            self._cached_pitch = pitch_rad

        bev = np.zeros((self.bev_h, self.bev_w, 3), dtype=np.uint8)
        iu = self._cached_img_u.astype(np.int32)
        iv = self._cached_img_v.astype(np.int32)
        bev[self._cached_valid] = img[iv[self._cached_valid], iu[self._cached_valid]]
        return bev, self._cached_valid

    def bev_pixel_to_meters(self, bev_x, bev_y):
        world_x = (bev_x - self.bev_w / 2.0) / self.bev_w * self.bev_x_range
        world_y = self.bev_y_max - (bev_y / self.bev_h) * (self.bev_y_max - self.bev_y_min)
        return world_x, world_y

    def meters_to_bev_pixel(self, world_x, world_y):
        bev_x = (world_x / self.bev_x_range + 0.5) * self.bev_w
        bev_y = (self.bev_y_max - world_y) / (self.bev_y_max - self.bev_y_min) * self.bev_h
        return bev_x, bev_y

    def diagnose_coverage(self, pitch_rad=0.0):
        """
        Report the actual world_y range (meters ahead) that the current
        cam_height/fov/pitch geometry maps into the real image frame, using
        only the CENTER column (world_x ~= 0) so a narrow-FOV mismatch on
        the sides doesn't mask a near-field problem straight ahead.
        """
        img_u, img_v, valid = self.compute_homography(pitch_rad)
        center_col = self.bev_w // 2
        col_valid = valid[:, center_col]
        col_world_y = self.world_y[:, center_col]

        if not col_valid.any():
            return None, None
        visible_y = col_world_y[col_valid]
        return float(visible_y.min()), float(visible_y.max())


# ═══════════════════════════════════════════════════════════════════════
# RANSACLaneFitter — unchanged from the sim node
# ═══════════════════════════════════════════════════════════════════════
class RANSACLaneFitter:
    """RANSAC 2nd-degree polynomial fitter: x = a*y^2 + b*y + c (BEV px)."""

    def __init__(self, n_iters=RANSAC_ITERS, threshold=RANSAC_THRESH,
                 min_inliers=RANSAC_MIN_INLIERS, seed=42):
        self.n_iters = n_iters
        self.threshold = threshold
        self.min_inliers = min_inliers
        # Seeded RNG (was the unseeded global np.random) so identical input
        # reproduces identical fits run-to-run instead of jittering — see
        # module docstring "TUNED VARIANT" note.
        self._rng = np.random.RandomState(seed)

    def fit(self, points_x, points_y):
        if len(points_x) < self.min_inliers:
            return None

        best_coeffs = None
        best_inliers = 0
        n = len(points_x)

        for _ in range(self.n_iters):
            idx = self._rng.choice(n, 3, replace=False)
            sample_y = points_y[idx]
            sample_x = points_x[idx]

            if (sample_y.max() - sample_y.min()) < 50:
                continue

            try:
                coeffs = np.polyfit(sample_y, sample_x, 2)
            except (np.linalg.LinAlgError, ValueError):
                continue

            predicted_x = np.polyval(coeffs, points_y)
            errors = np.abs(predicted_x - points_x)
            inlier_mask = errors < self.threshold
            n_inliers = inlier_mask.sum()

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_coeffs = coeffs

        if best_coeffs is not None and best_inliers >= self.min_inliers:
            predicted_x = np.polyval(best_coeffs, points_y)
            inlier_mask = np.abs(predicted_x - points_x) < self.threshold
            try:
                inlier_y = points_y[inlier_mask]
                inlier_x = points_x[inlier_mask]
                if len(inlier_y) > 200:
                    idx = self._rng.choice(len(inlier_y), 200, replace=False)
                    inlier_y = inlier_y[idx]
                    inlier_x = inlier_x[idx]
                best_coeffs = np.polyfit(inlier_y, inlier_x, 2)
            except (np.linalg.LinAlgError, ValueError):
                pass
            return best_coeffs

        return None


# ═══════════════════════════════════════════════════════════════════════
# LaneKalmanFilter — unchanged from the sim node
# ═══════════════════════════════════════════════════════════════════════
class LaneKalmanFilter:
    """Kalman filter on polynomial coefficients [a,b,c] for each lane."""

    def __init__(self):
        self.kf = cv2.KalmanFilter(6, 6)
        self.kf.transitionMatrix = np.eye(6, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(6, dtype=np.float32)

        process_noise = np.diag([
            3e-3, 1e-3, 5e-3,    # left lane:  a, b, c
            3e-3, 1e-3, 5e-3,    # right lane: a, b, c
        ]).astype(np.float32)
        self.kf.processNoiseCov = process_noise

        meas_noise = np.diag([
            0.05, 5e-2, 0.5,     # left lane:  a, b, c
            0.05, 5e-2, 0.5,     # right lane: a, b, c
        ]).astype(np.float32)
        self.kf.measurementNoiseCov = meas_noise

        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0

        self.initialized = False
        self.frames_without_detection = 0
        self.max_predict_only = MAX_PREDICT_ONLY_FRAMES

    def update(self, left_coeffs, right_coeffs, lane_width_bev):
        has_left = left_coeffs is not None
        has_right = right_coeffs is not None

        if not has_left and not has_right:
            self.frames_without_detection += 1
            if self.initialized and self.frames_without_detection <= self.max_predict_only:
                predicted = self.kf.predict()
                return (predicted[0:3, 0].copy(), predicted[3:6, 0].copy())
            if self.frames_without_detection > self.max_predict_only:
                # Coast window exhausted — the held state is now stale
                # enough (>1.5s old) that a fresh detection shouldn't be
                # pulled toward it via kf.correct(). Start clean next time
                # real points show up rather than reconciling against an
                # outdated prior.
                self.initialized = False
            return None, None

        self.frames_without_detection = 0
        meas = np.zeros((6, 1), np.float32)

        if has_left and has_right:
            meas[0:3, 0] = left_coeffs
            meas[3:6, 0] = right_coeffs
        elif has_left:
            meas[0:3, 0] = left_coeffs
            meas[3, 0] = left_coeffs[0]
            meas[4, 0] = left_coeffs[1]
            meas[5, 0] = left_coeffs[2] + lane_width_bev
        elif has_right:
            meas[0, 0] = right_coeffs[0]
            meas[1, 0] = right_coeffs[1]
            meas[2, 0] = right_coeffs[2] - lane_width_bev
            meas[3:6, 0] = right_coeffs

        if not self.initialized:
            self.kf.statePre = meas.copy()
            self.kf.statePost = meas.copy()
            self.initialized = True
            return meas[0:3, 0].copy(), meas[3:6, 0].copy()

        self.kf.predict()
        estimated = self.kf.correct(meas)
        return estimated[0:3, 0].copy(), estimated[3:6, 0].copy()

    def reset(self):
        self.initialized = False
        self.frames_without_detection = 0


# ═══════════════════════════════════════════════════════════════════════
# LKAPerceptionEngine — replaces the ROS Node; same math, no rclpy
# ═══════════════════════════════════════════════════════════════════════
class LKAPerceptionEngine:
    def __init__(self, img_w, img_h, cam_height=CAM_HEIGHT_DEFAULT,
                 cam_fov_deg=CAM_FOV_DEG_DEFAULT,
                 cam_pitch_deg=CAM_PITCH_DEG_DEFAULT,
                 lane_width_m=LANE_WIDTH_M_DEFAULT,
                 white_v_min=WHITE_V_MIN_DEFAULT,
                 white_s_max=WHITE_S_MAX_DEFAULT,
                 bev_y_min=BEV_Y_MIN, bev_y_max=BEV_Y_MAX):
        self.cam_height = cam_height
        self.white_v_min = white_v_min   # mutable — live-tunable via --tune
        self.white_s_max = white_s_max

        self.lateral_offset_bias = 0.0
        self.bias_calibration_active = False

        self.lateral_offset_filt = 0.0
        self.lateral_offset_alpha = 0.15

        self.curvature_filt = 0.0
        self.curvature_alpha = 0.08

        self.bev = DynamicBEV(
            cam_height=cam_height, cam_fov_deg=cam_fov_deg,
            img_w=img_w, img_h=img_h,
            bev_w=BEV_W, bev_h=BEV_H,
            bev_x_range=BEV_X_RANGE, bev_y_min=bev_y_min, bev_y_max=bev_y_max,
            static_pitch_deg=cam_pitch_deg,
        )
        self.ransac = RANSACLaneFitter()
        self.kalman = LaneKalmanFilter()
        self.lane_width_bev = lane_width_m / self.bev.m_per_px_x

        # See MIN_INLIER_SPAN_M above.
        self.min_inlier_span_px = MIN_INLIER_SPAN_M / self.bev.m_per_px_y
        self.max_extrapolation_gap_px = MAX_EXTRAPOLATION_GAP_M / self.bev.m_per_px_y

    def detect_lane_pixels(self, bev_img):
        """Detect lane marking pixels in BEV image via white segmentation."""
        hsv = cv2.cvtColor(bev_img, cv2.COLOR_RGB2HSV)
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, self.white_v_min]),
            np.array([180, self.white_s_max, 255]),
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        # Scene-mismatch guard: real lane paint never covers more than a
        # sliver of the BEV. If the mask floods past MAX_WHITE_FRAC (e.g.
        # camera has driven past the lane into a white-walled/bright area,
        # so the whole ground-plane homography is now sampling that instead
        # of ground), treat it as "no detection" rather than letting RANSAC
        # fit a polynomial to wall/glare noise.
        if white_mask.mean() / 255.0 > MAX_WHITE_FRAC:
            return None, None, None, None, white_mask

        lane_y, lane_x = np.where(white_mask > 0)
        if len(lane_x) < 20:
            return None, None, None, None, white_mask

        center_x = BEV_W / 2.0
        bottom_half = white_mask[BEV_H // 2:, :]
        histogram = np.sum(bottom_half, axis=0)

        left_hist = histogram[:int(center_x)]
        right_hist = histogram[int(center_x):]

        if left_hist.max() < 100 and right_hist.max() < 100:
            return None, None, None, None, white_mask

        left_peak = np.argmax(left_hist) if left_hist.max() > 100 else None
        right_peak = (np.argmax(right_hist) + int(center_x)) if right_hist.max() > 100 else None

        left_x_pts, left_y_pts = None, None
        right_x_pts, right_y_pts = None, None
        window = int(self.lane_width_bev * 0.4)

        if left_peak is not None:
            mask_l = np.abs(lane_x - left_peak) < window
            if mask_l.sum() >= RANSAC_MIN_INLIERS:
                left_x_pts = lane_x[mask_l]
                left_y_pts = lane_y[mask_l]
                if (left_y_pts.max() - left_y_pts.min()) < self.min_inlier_span_px:
                    left_x_pts, left_y_pts = None, None
                elif (BEV_H - left_y_pts.max()) > self.max_extrapolation_gap_px:
                    # Span is fine but the whole cluster sits far from the
                    # vehicle row — near-field position would be a multi-
                    # meter extrapolation, not a measurement.
                    left_x_pts, left_y_pts = None, None

        if right_peak is not None:
            mask_r = np.abs(lane_x - right_peak) < window
            if mask_r.sum() >= RANSAC_MIN_INLIERS:
                right_x_pts = lane_x[mask_r]
                right_y_pts = lane_y[mask_r]
                if (right_y_pts.max() - right_y_pts.min()) < self.min_inlier_span_px:
                    right_x_pts, right_y_pts = None, None
                elif (BEV_H - right_y_pts.max()) > self.max_extrapolation_gap_px:
                    right_x_pts, right_y_pts = None, None

        # NOTE: earlier version of this patch also nulled the OTHER side
        # whenever one was narrow-rejected, to avoid the single-lane Kalman
        # synthesis branch whiplashing when a side flickers in/out. Verified
        # against real footage that this was too aggressive — it broke
        # otherwise-healthy single-sided detections (e.g. right-lane span
        # 4.7m, perfectly good, discarded solely because left was 1.8m that
        # same frame) and stalled recovery for extended periods. Reverted to
        # per-side rejection only; the single-lane synthesis path in
        # LaneKalmanFilter.update() is left to do its designed job.

        return left_x_pts, left_y_pts, right_x_pts, right_y_pts, white_mask

    def _reject_if_implausible_curvature(self, coeffs):
        """Reject a raw (pre-Kalman) fit if either:
        (a) its own curvature implies a turn tighter than
            MAX_PHYSICAL_CURVATURE allows (see derivation above), or
        (b) its extrapolated position at the vehicle row lands off the
            visible ground plane entirely. A fit can have gentle,
            physically-fine curvature (small a) and still have a huge
            slope (b) — evaluating x = a*y^2+b*y+c at y_vehicle=BEV_H can
            land hundreds of pixels outside [0, BEV_W] even when a alone
            looks perfectly reasonable. Confirmed on real footage: a fit
            with curvature=0.0002 (far under the 0.132 bound) evaluated to
            x=-662px against a 600px-wide BEV — 1.5 frame-widths off the
            edge — and that single fit alone pushed lateral_offset to 6.4m.
        """
        if coeffs is None:
            return None
        lane_curvature = 2.0 * coeffs[0] * self.bev.m_per_px_x / (self.bev.m_per_px_y ** 2)
        if abs(lane_curvature) > MAX_PHYSICAL_CURVATURE:
            return None
        x_at_vehicle = np.polyval(coeffs, float(BEV_H))
        if x_at_vehicle < -BEV_W * MAX_OFF_FRAME_MARGIN_FRAC or \
           x_at_vehicle > BEV_W * (1.0 + MAX_OFF_FRAME_MARGIN_FRAC):
            return None
        return coeffs

    def compute_lane_output(self, left_coeffs, right_coeffs, frames_without_detection=0):
        """
        Compute lateral offset, curvature, relative yaw, confidence.

        Auto-bias calibration is DISABLED (root-cause fix carried over from
        the sim node): it was learning the vehicle's own transient
        off-centre position as "zero" and manufacturing a steady-state
        error. The AT-VEHICLE geometric offset is published directly.

        GRACEFUL DEGRADATION on lost detection: hold and decay the last
        known trajectory instead of snapping to "0 offset, 0 curvature."
        Reporting exactly zero the instant detection drops tells the
        controller "you are perfectly centered on a straight lane" even if
        the vehicle was mid-turn — confidently wrong, not just uncertain.
        Instead we exponentially decay the last good estimate toward zero
        AND ACTUALLY RETURN THE DECAYED VALUE (the original code computed
        this decay into self.curvature_filt/self.lateral_offset_filt but
        then discarded it, returning a hardcoded 0.0, 0.0 regardless —
        fixed here), and we report a confidence that drops with time spent
        coasting so the controller can tell "actively tracked" apart from
        "coasting on a stale estimate" instead of seeing a step to 0.0.
        """
        if left_coeffs is None or right_coeffs is None:
            self.curvature_filt *= 0.9
            self.lateral_offset_filt *= 0.9
            # Confidence decays over the coast window instead of stepping to
            # 0 immediately — gives the controller a smooth signal to act on
            # (e.g. reduce speed / steering aggressiveness) rather than a
            # binary detected/not-detected flag.
            coast_confidence = max(0.0, 1.0 - frames_without_detection / max(1, COAST_CONFIDENCE_FRAMES))
            return self.lateral_offset_filt, self.curvature_filt, 0.0, coast_confidence

        y_vehicle = float(BEV_H)
        y_preview = ((self.bev.bev_y_max - PREVIEW_DISTANCE_M)
                     / (self.bev.bev_y_max - self.bev.bev_y_min) * BEV_H)

        x_left_v = np.polyval(left_coeffs, y_vehicle)
        x_right_v = np.polyval(right_coeffs, y_vehicle)
        lane_center_v = (x_left_v + x_right_v) / 2.0
        vehicle_center_x = BEV_W / 2.0
        raw_offset_at_vehicle = (vehicle_center_x - lane_center_v) * self.bev.m_per_px_x

        self.lateral_offset_bias = 0.0
        self.bias_calibration_active = False
        lateral_offset_raw = raw_offset_at_vehicle - self.lateral_offset_bias

        self.lateral_offset_filt = (
            (1.0 - self.lateral_offset_alpha) * self.lateral_offset_filt
            + self.lateral_offset_alpha * lateral_offset_raw
        )
        lateral_offset = self.lateral_offset_filt

        a_avg = (left_coeffs[0] + right_coeffs[0]) / 2.0
        b_avg = (left_coeffs[1] + right_coeffs[1]) / 2.0
        curvature_raw = 2.0 * a_avg * self.bev.m_per_px_x / (self.bev.m_per_px_y ** 2)
        # Defense-in-depth: the per-lane rejection above should already
        # keep implausible fits out, but clamp the blended result too in
        # case Kalman smoothing between two individually-plausible-but-
        # opposed fits still produces something unphysical.
        curvature_raw = float(np.clip(curvature_raw, -MAX_PHYSICAL_CURVATURE, MAX_PHYSICAL_CURVATURE))

        self.curvature_filt = (
            (1.0 - self.curvature_alpha) * self.curvature_filt
            + self.curvature_alpha * curvature_raw
        )
        curvature = self.curvature_filt

        slope_px = 2.0 * a_avg * y_preview + b_avg
        slope_world = slope_px * (self.bev.m_per_px_x / self.bev.m_per_px_y)
        relative_yaw = -math.atan(slope_world)

        detected_width = x_right_v - x_left_v
        expected_width = self.lane_width_bev
        width_error = abs(detected_width - expected_width) / expected_width
        confidence = max(0.0, 1.0 - width_error)

        if frames_without_detection > 0:
            # smooth_left/right came from Kalman predict()-only coasting,
            # not a fresh measurement this frame — soften confidence in
            # proportion to how long we've been coasting so the controller
            # can see the difference between "actively tracked" and "held."
            coast_factor = max(0.0, 1.0 - frames_without_detection / max(1, COAST_CONFIDENCE_FRAMES))
            confidence *= coast_factor

        return lateral_offset, curvature, relative_yaw, confidence

    def draw_bev_visualization(self, bev_img, white_mask, left_coeffs, right_coeffs,
                                pitch_deg, lateral_offset, curvature):
        vis = bev_img.copy()
        overlay = vis.copy()
        overlay[white_mask > 0] = (255, 255, 0)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        y_range = np.arange(0, BEV_H)

        if left_coeffs is not None:
            x_left = np.polyval(left_coeffs, y_range).astype(int)
            for i in range(len(y_range) - 1):
                if 0 <= x_left[i] < BEV_W and 0 <= x_left[i + 1] < BEV_W:
                    cv2.line(vis, (x_left[i], y_range[i]),
                             (x_left[i + 1], y_range[i + 1]), (0, 0, 255), 2)

        if right_coeffs is not None:
            x_right = np.polyval(right_coeffs, y_range).astype(int)
            for i in range(len(y_range) - 1):
                if 0 <= x_right[i] < BEV_W and 0 <= x_right[i + 1] < BEV_W:
                    cv2.line(vis, (x_right[i], y_range[i]),
                             (x_right[i + 1], y_range[i + 1]), (0, 255, 0), 2)

        if left_coeffs is not None and right_coeffs is not None:
            x_center = ((np.polyval(left_coeffs, y_range) +
                         np.polyval(right_coeffs, y_range)) / 2).astype(int)
            for i in range(len(y_range) - 1):
                if 0 <= x_center[i] < BEV_W and 0 <= x_center[i + 1] < BEV_W:
                    cv2.line(vis, (x_center[i], y_range[i]),
                             (x_center[i + 1], y_range[i + 1]), (255, 0, 255), 1)

        cv2.putText(vis, f"Pitch: {pitch_deg:.2f} deg", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Offset: {lateral_offset:.3f} m", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"Curv: {curvature:.5f}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis

    def draw_cluster_debug(self, white_mask, left_x, left_y, right_x, right_y,
                            raw_left, raw_right):
        """
        Debug view for the "which points is the picker actually using"
        question: raw white mask, with the SELECTED left-window points in
        blue and right-window points in red, plus the raw (pre-Kalman,
        pre-smoothing) fit as a thin cyan/yellow curve. If a curve is
        clearly tracking a paver joint/seam instead of the painted line,
        it'll be obvious here — the mask alone (--debug-mask) can't show
        you the point ASSIGNMENT, only the raw threshold result.
        """
        vis = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR)
        if left_x is not None:
            vis[left_y, left_x] = (255, 80, 0)      # blue-ish = left selection
        if right_x is not None:
            vis[right_y, right_x] = (0, 80, 255)    # red-ish = right selection

        y_range = np.arange(0, BEV_H)
        if raw_left is not None:
            xl = np.polyval(raw_left, y_range).astype(int)
            for i in range(len(y_range) - 1):
                if 0 <= xl[i] < BEV_W and 0 <= xl[i + 1] < BEV_W:
                    cv2.line(vis, (xl[i], y_range[i]), (xl[i + 1], y_range[i + 1]),
                             (0, 255, 255), 1)
        if raw_right is not None:
            xr = np.polyval(raw_right, y_range).astype(int)
            for i in range(len(y_range) - 1):
                if 0 <= xr[i] < BEV_W and 0 <= xr[i + 1] < BEV_W:
                    cv2.line(vis, (xr[i], y_range[i]), (xr[i + 1], y_range[i + 1]),
                             (0, 255, 0), 1)
        return vis

    def draw_front_visualization(self, frame, left_coeffs, right_coeffs,
                                  lateral_offset, curvature, confidence, pitch_deg):
        vis = frame.copy()
        h, w = vis.shape[:2]
        y_bev = np.linspace(0, BEV_H - 1, 50)

        pitch_rad = math.radians(pitch_deg)
        cos_p = math.cos(self.bev.static_pitch + pitch_rad)
        sin_p = math.sin(self.bev.static_pitch + pitch_rad)

        def bev_lane_to_camera(coeffs, color, thickness=3):
            x_bev = np.polyval(coeffs, y_bev)
            pts = []
            for xb, yb in zip(x_bev, y_bev):
                wx, wy = self.bev.bev_pixel_to_meters(xb, yb)
                x_cam = -wx
                y_cam = self.cam_height * cos_p - wy * sin_p
                z_cam = self.cam_height * sin_p + wy * cos_p
                if z_cam > 0.5:
                    u = int(self.bev.fx * x_cam / z_cam + self.bev.cx)
                    v = int(self.bev.fy * y_cam / z_cam + self.bev.cy)
                    if 0 <= u < w and 0 <= v < h:
                        pts.append((u, v))
            for i in range(len(pts) - 1):
                cv2.line(vis, pts[i], pts[i + 1], color, thickness)

        if left_coeffs is not None:
            bev_lane_to_camera(left_coeffs, (0, 0, 255), 4)
        if right_coeffs is not None:
            bev_lane_to_camera(right_coeffs, (0, 255, 0), 4)

        y0 = 25
        cv2.putText(vis, f"Pitch: {pitch_deg:.2f} deg", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, f"Offset: {lateral_offset:.3f}m  Curv: {curvature:.5f}",
                    (10, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(vis, f"Confidence: {confidence:.2f}", (10, y0 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cx = w // 2
        cv2.line(vis, (cx, h), (cx, h - 40), (0, 255, 255), 3)
        lane_center_px = cx - int(lateral_offset / self.bev.m_per_px_x)
        cv2.circle(vis, (lane_center_px, h - 10), 8, (255, 0, 255), -1)

        return vis

    def process_frame(self, frame_rgb, pitch_deg):
        """
        Run the full pipeline on one RGB frame.
        Returns (front_vis_rgb, bev_vis_rgb, white_mask, cluster_debug_bgr,
                 lateral_offset, curvature, relative_yaw, confidence).
        cluster_debug_bgr is already BGR (ready for imshow), unlike the
        other two which are RGB — see draw_cluster_debug.
        """
        pitch_rad = math.radians(pitch_deg)

        bev_img, _ = self.bev.warp(frame_rgb, pitch_rad)
        left_x, left_y, right_x, right_y, white_mask = self.detect_lane_pixels(bev_img)

        raw_left = self.ransac.fit(left_x, left_y) if left_x is not None else None
        raw_right = self.ransac.fit(right_x, right_y) if right_x is not None else None

        # Reject any single-lane fit whose own curvature implies a turn
        # tighter than the track can physically require (see
        # MAX_PHYSICAL_CURVATURE derivation above). This catches a bad fit
        # — wrong-feature lock-on, narrow-cluster extrapolation — at the
        # source, before it ever reaches the Kalman filter, rather than
        # letting a corrupted trajectory get corrected-in and then have to
        # decay back out over the next several seconds. Only the offending
        # side is rejected (see the per-side-only lesson from the min-span
        # gate above — rejecting both sides broke legitimate recovery).
        raw_left = self._reject_if_implausible_curvature(raw_left)
        raw_right = self._reject_if_implausible_curvature(raw_right)

        smooth_left, smooth_right = self.kalman.update(raw_left, raw_right, self.lane_width_bev)

        lateral_offset, curvature, relative_yaw, confidence = self.compute_lane_output(
            smooth_left, smooth_right, self.kalman.frames_without_detection
        )

        bev_vis = self.draw_bev_visualization(
            bev_img, white_mask, smooth_left, smooth_right, pitch_deg, lateral_offset, curvature
        )
        front_vis = self.draw_front_visualization(
            frame_rgb, smooth_left, smooth_right, lateral_offset, curvature, confidence, pitch_deg
        )
        cluster_debug = self.draw_cluster_debug(
            white_mask, left_x, left_y, right_x, right_y, raw_left, raw_right
        )

        return (front_vis, bev_vis, white_mask, cluster_debug,
                lateral_offset, curvature, relative_yaw, confidence)


# ═══════════════════════════════════════════════════════════════════════
# RunLogger — auto-indexed output dir, mirrors the bbw_radar.py pattern
# ═══════════════════════════════════════════════════════════════════════
class RunLogger:
    def __init__(self, base_dir, save_video, save_bev_video, fps, front_size, bev_size):
        os.makedirs(base_dir, exist_ok=True)
        existing = [d for d in os.listdir(base_dir) if d.startswith('run_')]
        next_idx = len(existing) + 1
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(base_dir, f'run_{next_idx:04d}_{stamp}')
        os.makedirs(self.run_dir, exist_ok=True)

        csv_path = os.path.join(self.run_dir, 'lane_output_log.csv')
        self._csv_file = open(csv_path, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'frame_idx', 'timestamp_s', 'pitch_deg',
            'lateral_offset_m', 'curvature_1_per_m',
            'relative_yaw_rad', 'confidence',
        ])
        self.csv_path = csv_path

        self.front_writer = None
        self.bev_writer = None
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        if save_video:
            self.front_path = os.path.join(self.run_dir, 'annotated_front.mp4')
            self.front_writer = cv2.VideoWriter(self.front_path, fourcc, fps, front_size)
        if save_bev_video:
            self.bev_path = os.path.join(self.run_dir, 'annotated_bev.mp4')
            self.bev_writer = cv2.VideoWriter(self.bev_path, fourcc, fps, bev_size)

    def log(self, frame_idx, timestamp_s, pitch_deg, lateral_offset, curvature,
            relative_yaw, confidence):
        self._csv_writer.writerow([
            frame_idx, f'{timestamp_s:.3f}', f'{pitch_deg:.3f}',
            f'{lateral_offset:.4f}', f'{curvature:.6f}',
            f'{relative_yaw:.4f}', f'{confidence:.3f}',
        ])

    def write_front(self, frame_bgr):
        if self.front_writer is not None:
            self.front_writer.write(frame_bgr)

    def write_bev(self, frame_bgr):
        if self.bev_writer is not None:
            self.bev_writer.write(frame_bgr)

    def close(self):
        self._csv_file.close()
        if self.front_writer is not None:
            self.front_writer.release()
        if self.bev_writer is not None:
            self.bev_writer.release()


def load_pitch_csv(path):
    """Load an optional frame_idx,pitch_deg CSV into a dict."""
    pitch_map = {}
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    start = 1 if rows and not rows[0][0].strip().lstrip('-').isdigit() else 0
    for row in rows[start:]:
        if len(row) < 2:
            continue
        try:
            pitch_map[int(row[0])] = float(row[1])
        except ValueError:
            continue
    return pitch_map


def build_arg_parser():
    p = argparse.ArgumentParser(
        description='LKA perception on a recorded video file (Windows, no ROS).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--video', required=True,
                    help='Path to video file, or a webcam index (e.g. "0")')
    p.add_argument('--output-dir', default='runs', help='Base directory for run outputs')
    p.add_argument('--headless', action='store_true',
                    help='No display windows; just process and log/save')
    p.add_argument('--speed', type=float, default=1.0,
                    help='Playback speed multiplier for the display windows')
    p.add_argument('--save-video', action='store_true', help='Save annotated front-camera video')
    p.add_argument('--save-bev-video', action='store_true', help='Save annotated BEV video')
    p.add_argument('--tune', action='store_true',
                    help='Open an HSV trackbar UI to live-tune white-mask thresholds')
    p.add_argument('--debug-mask', action='store_true',
                    help='Open a window showing the raw binary white_mask')
    p.add_argument('--debug-clusters', action='store_true',
                    help='Open a window showing which points were assigned to the '
                         'left/right lane (blue/red) plus the raw pre-Kalman fit — '
                         'use this to see if the picker is locking onto the wrong '
                         'feature (e.g. a paver joint instead of the painted line)')

    p.add_argument('--cam-height', type=float, default=CAM_HEIGHT_DEFAULT,
                    help='Camera height above ground, meters')
    p.add_argument('--fov', type=float, default=CAM_FOV_DEG_DEFAULT,
                    help='Camera horizontal FOV, degrees')
    p.add_argument('--pitch', type=float, default=CAM_PITCH_DEG_DEFAULT,
                    help='Static camera pitch offset, degrees (used if no --pitch-csv)')
    p.add_argument('--pitch-csv', default=None,
                    help='Optional CSV of frame_idx,pitch_deg for logged-IMU playback')
    p.add_argument('--lane-width', type=float, default=LANE_WIDTH_M_DEFAULT,
                    help='Expected lane width, meters')
    p.add_argument('--bev-y-min', type=float, default=BEV_Y_MIN,
                    help='Near edge of BEV coverage, meters ahead')
    p.add_argument('--bev-y-max', type=float, default=BEV_Y_MAX,
                    help='Far edge of BEV coverage, meters ahead')
    p.add_argument('--white-v-min', type=int, default=WHITE_V_MIN_DEFAULT,
                    help='HSV V-channel lower bound for white lane markings')
    p.add_argument('--white-s-max', type=int, default=WHITE_S_MAX_DEFAULT,
                    help='HSV S-channel upper bound for white lane markings')

    p.add_argument('--start-frame', type=int, default=0, help='Skip to this frame before processing')
    p.add_argument('--max-frames', type=int, default=None, help='Stop after this many frames')
    return p


def main():
    args = build_arg_parser().parse_args()

    source = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'ERROR: could not open video source: {args.video}')
        return

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'Source: {args.video}  ({img_w}x{img_h} @ {fps:.1f} fps'
          + (f', {total_frames} frames)' if total_frames > 0 else ', live)'))
    print(f'Cam config: height={args.cam_height}m  fov={args.fov}deg  '
          f'pitch={args.pitch}deg  lane_width={args.lane_width}m')

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    pitch_map = load_pitch_csv(args.pitch_csv) if args.pitch_csv else None

    engine = LKAPerceptionEngine(
        img_w=img_w, img_h=img_h,
        cam_height=args.cam_height, cam_fov_deg=args.fov, cam_pitch_deg=0.0,
        lane_width_m=args.lane_width,
        white_v_min=args.white_v_min, white_s_max=args.white_s_max,
        bev_y_min=args.bev_y_min, bev_y_max=args.bev_y_max,
    )

    visible_min, visible_max = engine.bev.diagnose_coverage(math.radians(args.pitch))
    if visible_min is None:
        print('WARNING: with this cam-height/fov/pitch, NOTHING in the requested '
              f'{args.bev_y_min}-{args.bev_y_max}m range is visible to the camera.')
    else:
        print(f'Camera coverage (center column): visible from {visible_min:.2f}m to '
              f'{visible_max:.2f}m ahead.')
        if visible_min > args.bev_y_min + 0.05:
            print(f'  -> near-field gap: requested bev-y-min={args.bev_y_min}m but the '
                  f'nearest visible ground is {visible_min:.2f}m ahead.')

    logger = RunLogger(
        base_dir=args.output_dir,
        save_video=args.save_video, save_bev_video=args.save_bev_video,
        fps=fps, front_size=(img_w, img_h), bev_size=(BEV_W, BEV_H),
    )
    print(f'Run outputs -> {logger.run_dir}')

    if args.tune and not args.headless:
        cv2.namedWindow('HSV Tune')
        cv2.createTrackbar('white_v_min', 'HSV Tune', args.white_v_min, 255, lambda v: None)
        cv2.createTrackbar('white_s_max', 'HSV Tune', args.white_s_max, 255, lambda v: None)

    if args.debug_mask and not args.headless:
        cv2.namedWindow('LKA - Mask (debug)')
    if args.debug_clusters and not args.headless:
        cv2.namedWindow('LKA - Clusters (debug)')

    frame_idx = args.start_frame
    processed = 0
    t_start = time.time()
    wait_ms = max(1, int(1000.0 / (fps * max(args.speed, 0.01))))

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if args.max_frames is not None and processed >= args.max_frames:
                break

            if args.tune and not args.headless:
                engine.white_v_min = cv2.getTrackbarPos('white_v_min', 'HSV Tune')
                engine.white_s_max = cv2.getTrackbarPos('white_s_max', 'HSV Tune')

            if pitch_map is not None:
                pitch_deg = pitch_map.get(frame_idx, args.pitch)
            else:
                pitch_deg = args.pitch

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            try:
                (front_vis, bev_vis, white_mask, cluster_debug,
                 lateral_offset, curvature, relative_yaw, confidence) = \
                    engine.process_frame(frame_rgb, pitch_deg)
            except Exception as e:
                print(f'[frame {frame_idx}] processing error, skipping: {e}')
                frame_idx += 1
                continue

            timestamp_s = frame_idx / fps
            logger.log(frame_idx, timestamp_s, pitch_deg, lateral_offset,
                       curvature, relative_yaw, confidence)

            front_vis_bgr = cv2.cvtColor(front_vis, cv2.COLOR_RGB2BGR)
            bev_vis_bgr = cv2.cvtColor(bev_vis, cv2.COLOR_RGB2BGR)
            logger.write_front(front_vis_bgr)
            logger.write_bev(bev_vis_bgr)

            if not args.headless:
                cv2.imshow('LKA - Front', front_vis_bgr)
                cv2.imshow('LKA - BEV', bev_vis_bgr)
                if args.debug_mask:
                    cv2.imshow('LKA - Mask (debug)', white_mask)
                if args.debug_clusters:
                    cv2.imshow('LKA - Clusters (debug)', cluster_debug)
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord('q'):
                    print('Quit requested.')
                    break
                if key == ord('p'):
                    cv2.waitKey(0)

            frame_idx += 1
            processed += 1

    finally:
        elapsed = time.time() - t_start
        cap.release()
        logger.close()
        if not args.headless:
            cv2.destroyAllWindows()

        print(f'\nProcessed {processed} frames in {elapsed:.1f}s '
              f'({processed / elapsed if elapsed > 0 else 0:.1f} fps).')
        print(f'CSV log: {logger.csv_path}')
        if args.save_video:
            print(f'Front video: {logger.front_path}')
        if args.save_bev_video:
            print(f'BEV video: {logger.bev_path}')


if __name__ == '__main__':
    main()