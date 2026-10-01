"""
sort_tracker.py
----------------
Step 4: Apply object tracking using SORT (Simple Online and Realtime
Tracking) — a Kalman filter predicts each tracked object's next position,
and the Hungarian algorithm (scipy.optimize.linear_sum_assignment) matches
new detections to existing tracks by IoU (Intersection-over-Union).

This is a compact, dependency-light re-implementation of the classic SORT
algorithm (Bewley et al., 2016) using only numpy + scipy (no filterpy
required).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(box_a, box_b):
    """IoU between two [x, y, w, h] boxes."""
    ax1, ay1, aw, ah = box_a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = box_b
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class KalmanBoxTracker:
    """A constant-velocity Kalman filter tracking a single object's box,
    in state [cx, cy, s, r, vcx, vcy, vs] (center, area, aspect ratio + velocities)."""
    count = 0

    def __init__(self, bbox, class_name):
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        s, r = w * h, w / float(h) if h != 0 else 1.0

        self.state = np.array([cx, cy, s, r, 0, 0, 0], dtype=float)
        self.P = np.eye(7) * 10.0            # state covariance
        self.Q = np.eye(7) * 0.5             # process noise
        self.Q[4:, 4:] *= 0.1
        self.R = np.eye(4) * 1.0             # measurement noise

        self.F = np.eye(7)                    # constant-velocity motion model
        for i in range(3):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 7))             # measurement matrix (observe cx,cy,s,r)
        for i in range(4):
            self.H[i, i] = 1.0

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.class_name = class_name
        self.hits = 1
        self.time_since_update = 0

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self._state_to_box()

    def update(self, bbox, class_name):
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        s, r = w * h, w / float(h) if h != 0 else 1.0
        z = np.array([cx, cy, s, r], dtype=float)

        y_res = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_res
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.class_name = class_name
        self.hits += 1
        self.time_since_update = 0

    def _state_to_box(self):
        cx, cy, s, r = self.state[:4]
        s = max(s, 1e-3)
        w = np.sqrt(s * r)
        h = s / w if w != 0 else 0
        return [cx - w / 2, cy - h / 2, w, h]

    def get_box(self):
        return self._state_to_box()


class Sort:
    """Multi-object tracker: call update(detections) once per frame."""

    def __init__(self, max_age=15, min_hits=2, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        """
        detections: list of [x, y, w, h, confidence, class_id, class_name]
        returns: list of [x, y, w, h, track_id, class_name]
        """
        # 1. Predict new locations for existing trackers
        predicted_boxes = [t.predict() for t in self.trackers]

        # 2. Associate detections to trackers by IoU (Hungarian algorithm)
        matches, unmatched_dets, unmatched_trks = self._associate(
            detections, predicted_boxes
        )

        # 3. Update matched trackers with new detections
        for det_idx, trk_idx in matches:
            box = detections[det_idx][:4]
            class_name = detections[det_idx][6]
            self.trackers[trk_idx].update(box, class_name)

        # 4. Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            box = detections[det_idx][:4]
            class_name = detections[det_idx][6]
            self.trackers.append(KalmanBoxTracker(box, class_name))

        # 5. Remove dead trackers (not seen for too long)
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # 6. Output confirmed tracks
        results = []
        for t in self.trackers:
            if t.hits >= self.min_hits or t.time_since_update == 0:
                x, y, w, h = t.get_box()
                results.append([x, y, w, h, t.id, t.class_name])
        return results

    def _associate(self, detections, predicted_boxes):
        if len(predicted_boxes) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(predicted_boxes)))

        iou_matrix = np.zeros((len(detections), len(predicted_boxes)))
        for d, det in enumerate(detections):
            for t, trk_box in enumerate(predicted_boxes):
                iou_matrix[d, t] = iou(det[:4], trk_box)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)  # maximize IoU

        matches, unmatched_dets, unmatched_trks = [], [], []
        matched_det_idx, matched_trk_idx = set(), set()
        for d, t in zip(row_ind, col_ind):
            if iou_matrix[d, t] >= self.iou_threshold:
                matches.append((d, t))
                matched_det_idx.add(d)
                matched_trk_idx.add(t)

        unmatched_dets = [d for d in range(len(detections)) if d not in matched_det_idx]
        unmatched_trks = [t for t in range(len(predicted_boxes)) if t not in matched_trk_idx]
        return matches, unmatched_dets, unmatched_trks
