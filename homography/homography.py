import cv2
import numpy as np

# OF COURSE THIS IS AI GENERATED PROFESSOR :D
# WE TALKED WITH IT TO LEARN ABOUT HOW THE KEYPOINT DETECTION WORKS
PITCH_KEYPOINT_COORDS = np.array([
    [    0,    0],   #  0  top-left corner
    [    0,  138],   #  1  top-left of left 18-box (touchline)
    [    0,  248],   #  2  top-left of left 6-box (touchline)
    [    0,  431],   #  3  bottom-left of left 6-box (touchline)
    [    0,  541],   #  4  bottom-left of left 18-box (touchline)
    [    0,  680],   #  5  bottom-left corner
    [   55,  248],   #  6  top-right of left 6-box (inside)
    [   55,  431],   #  7  bottom-right of left 6-box (inside)
    [  110,  340],   #  8  left penalty spot
    [  165,  138],   #  9  top-right of left 18-box (inside)
    [  165,  268],   # 10  top of left penalty arc
    [  165,  412],   # 11  bottom of left penalty arc
    [  165,  541],   # 12  bottom-right of left 18-box (inside)
    [  525,    0],   # 13  top of halfway line (touchline)
    [  525,  249],   # 14  halfway line — top of centre circle
    [  525,  431],   # 15  halfway line — bottom of centre circle
    [  525,  680],   # 16  bottom of halfway line (touchline)
    [  885,  138],   # 17  top-left of right 18-box (inside)
    [  885,  268],   # 18  top of right penalty arc
    [  885,  412],   # 19  bottom of right penalty arc
    [  885,  541],   # 20  bottom-left of right 18-box (inside)
    [  940,  340],   # 21  right penalty spot
    [  995,  248],   # 22  top-left of right 6-box (inside)
    [  995,  431],   # 23  bottom-left of right 6-box (inside)
    [ 1050,    0],   # 24  top-right corner
    [ 1050,  138],   # 25  top-right of right 18-box (touchline)
    [ 1050,  248],   # 26  top-right of right 6-box (touchline)
    [ 1050,  431],   # 27  bottom-right of right 6-box (touchline)
    [ 1050,  541],   # 28  bottom-right of right 18-box (touchline)
    [ 1050,  680],   # 29  bottom-right corner
    [  201,  340],   # 30  left penalty arc midpoint
    [  849,  340],   # 31  right penalty arc midpoint
], dtype=np.float32)

_KPT_CONF_THRESHOLD: float = 0.5
_RANSAC_THRESHOLD:   float = 30.0
_MIN_INLIERS:        int   = 4


class HomographyTransformer:
    """
    Computes camera-to-pitch homography and tracks camera motion using optical flow.
    """
    PITCH_W: int = 1050
    PITCH_H: int = 680
    H_RECOMPUTE_EVERY: int = 90

    # Optical flow parameters
    _LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    _FEATURE_PARAMS = dict(
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7,
    )

    _REFRESH_INTERVAL = 30
    _MAX_FLOW_CORRECTION = 80.0

    def __init__(self, pitch_tracker) -> None:
        self.field_detector = pitch_tracker
        self.H: np.ndarray | None = None
        self.keyframe_H: dict = {}
        self.sorted_keyframes: list = []
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None
        self._frame_count: int = 0

    def compute(
        self,
        image_keypoints: np.ndarray,
        confidences: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """
        Estimate homography from camera space to pitch canvas using
        detected field keypoints.
        """
        if image_keypoints is None:
            return None

        image_keypoints = np.asarray(image_keypoints, dtype=np.float32)
        n = len(image_keypoints)
        if n == 0:
            return None

        if confidences is not None:
            confidences = np.asarray(confidences, dtype=np.float32)
            valid = confidences >= _KPT_CONF_THRESHOLD
        else:
            valid = ~((image_keypoints[:, 0] == 0) & (image_keypoints[:, 1] == 0))

        max_idx    = min(n, len(PITCH_KEYPOINT_COORDS))
        valid_mask = valid[:max_idx]

        src = image_keypoints[:max_idx][valid_mask]
        dst = PITCH_KEYPOINT_COORDS[:max_idx][valid_mask]

        n_valid = len(src)
        if n_valid < _MIN_INLIERS:
            return None

        H_ransac, mask = cv2.findHomography(src, dst, cv2.RANSAC, _RANSAC_THRESHOLD)
        inliers = int(mask.sum()) if mask is not None else 0

        if H_ransac is not None and inliers >= max(_MIN_INLIERS, n_valid // 2):
            inlier_mask = mask.ravel().astype(bool)
            H_final, _ = cv2.findHomography(src[inlier_mask], dst[inlier_mask], 0)
            if H_final is not None and self._sanity_check(H_final):
                return H_final

        H_ls, _ = cv2.findHomography(src, dst, 0)
        if H_ls is not None and self._sanity_check(H_ls):
            return H_ls

        return None

    def _build_composite_H(self, frames, sample_every=10):
        """
        Scan frames to build a composite best-confidence keypoint set
        and compute the best possible H from it.
        """
        N_KPT     = 32
        best_pts  = np.zeros((N_KPT, 2), dtype=np.float32)
        best_conf = np.zeros(N_KPT,      dtype=np.float32)

        for i in range(0, len(frames), sample_every):
            result = self.field_detector.detect_keypoints_with_confidence(frames[i])
            if result is None:
                continue
            pts, conf = result
            for idx in range(min(N_KPT, len(pts))):
                if conf[idx] > best_conf[idx]:
                    best_conf[idx] = conf[idx]
                    best_pts[idx]  = pts[idx]

            n_good = int((best_conf >= 0.5).sum())
            print(f"  [scan] frame {i:04d} — {n_good}/32 keypoints >= 0.5 conf")

            if n_good == N_KPT:
                print("  [scan] All 32 found — stopping early.")
                break

        H = self.compute(best_pts, confidences=best_conf)
        return H, best_pts, best_conf

    def _recompute_H(self, frame):
        """
        Recompute H from a single frame. Returns new H or None.
        """
        result = self.field_detector.detect_keypoints_with_confidence(frame)
        if result is None:
            return None
        pts, conf = result
        return self.compute(pts, confidences=conf)

    def _interpolate_H(self, H_prev, H_next, alpha):
        """
        Linearly interpolate between two homography matrices.
        """
        return (1.0 - alpha) * H_prev + alpha * H_next

    def precompute_keyframes(self, video_frames, sample_every=10):
        H_base, _, _ = self._build_composite_H(video_frames, sample_every=sample_every)
        self.H = H_base

        if H_base is not None:
            print("Base homography ready.")
        else:
            print("WARNING: Homography failed — bird's-eye view disabled.")

        print("Pre-computing homography keyframes...")
        keyframe_indices = list(range(0, len(video_frames), self.H_RECOMPUTE_EVERY))
        self.keyframe_H = {}

        for kf_idx in keyframe_indices:
            H_new = self._recompute_H(video_frames[kf_idx])
            if H_new is not None:
                self.keyframe_H[kf_idx] = H_new
                print(f"  keyframe {kf_idx:04d}: H recomputed.")
            else:
                # find the most recent valid H
                prev_valid = [k for k in self.keyframe_H if k < kf_idx]
                if prev_valid:
                    self.keyframe_H[kf_idx] = self.keyframe_H[max(prev_valid)]
                    print(f"  keyframe {kf_idx:04d}: H recomputation failed — keeping previous.")
                elif H_base is not None:
                    self.keyframe_H[kf_idx] = H_base
                else:
                    print(f"  keyframe {kf_idx:04d}: no valid H available.")

        self.sorted_keyframes = sorted(self.keyframe_H.keys())
        if self.keyframe_H:
            self.H = self.keyframe_H[keyframe_indices[0]]

    def update_for_frame(self, frame_num: int, prev_frame: np.ndarray | None, curr_frame: np.ndarray) -> None:
        if not self.keyframe_H:
            return

        # Interpolate H
        prev_kf = max((k for k in self.sorted_keyframes if k <= frame_num), default=self.sorted_keyframes[0])
        next_kf_candidates = [k for k in self.sorted_keyframes if k > frame_num]

        if next_kf_candidates:
            next_kf = next_kf_candidates[0]
            interval = next_kf - prev_kf
            alpha = (frame_num - prev_kf) / interval if interval > 0 else 0.0
            self.H = self._interpolate_H(
                self.keyframe_H[prev_kf],
                self.keyframe_H[next_kf],
                alpha,
            )
        else:
            self.H = self.keyframe_H[prev_kf]

        # Optical flow update
        if frame_num > 0 and prev_frame is not None:
            self.update_with_optical_flow(prev_frame, curr_frame)

    def update_with_optical_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> None:
        """
        Refine self.H using sparse optical flow between prev_frame and curr_frame.
        """
        if self.H is None:
            return

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if (
            self._prev_gray is None
            or self._prev_pts is None
            or self._frame_count % self._REFRESH_INTERVAL == 0
        ):
            self._prev_pts = self._detect_pitch_features(prev_frame)
            self._prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        if self._prev_pts is None or len(self._prev_pts) < 8:
            self._prev_gray = curr_gray
            self._frame_count += 1
            return

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,
            curr_gray,
            self._prev_pts,
            None,
            **self._LK_PARAMS,
        )

        if curr_pts is None:
            self._prev_gray = curr_gray
            self._frame_count += 1
            return

        good_prev = self._prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if len(good_prev) < 8:
            self._prev_pts = None
            self._prev_gray = curr_gray
            self._frame_count += 1
            return

        H_motion, _ = cv2.findHomography(
            good_prev, good_curr, cv2.RANSAC, 3.0
        )

        if H_motion is None:
            self._prev_gray = curr_gray
            self._frame_count += 1
            return

        correction = np.linalg.norm(H_motion - np.eye(3))
        if correction > self._MAX_FLOW_CORRECTION:
            self._prev_gray = curr_gray
            self._frame_count += 1
            return

        try:
            H_motion_inv = np.linalg.inv(H_motion)
            H_updated = self.H @ H_motion_inv

            if self._sanity_check(H_updated):
                self.H = H_updated
        except np.linalg.LinAlgError:
            pass

        self._prev_gray = curr_gray
        self._prev_pts  = good_curr.reshape(-1, 1, 2)
        self._frame_count += 1

    def transform(self, point: tuple[int, int]) -> tuple[int, int] | None:
        """Map a single camera-space point to pitch canvas coords."""
        if self.H is None:
            return None
        p = np.array(
            [[[float(point[0]), float(point[1])]]],
            dtype=np.float32,
        )
        mapped = cv2.perspectiveTransform(p, self.H)
        return (int(mapped[0][0][0]), int(mapped[0][0][1]))

    def _detect_pitch_features(
        self, frame: np.ndarray
    ) -> np.ndarray | None:
        """
        Detect good features to track on the pitch area of the frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))

        kernel = np.ones((15, 15), np.uint8)
        mask   = cv2.erode(mask, kernel, iterations=1)

        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self._FEATURE_PARAMS)
        return pts

    def _sanity_check(self, H: np.ndarray) -> bool:
        """Check H is not degenerate and maps pitch to a sensible image region."""
        if H is None:
            return False
        if abs(np.linalg.det(H)) < 1e-10:
            return False

        corners = np.array([
            [[0.,    0.]],
            [[float(self.PITCH_W), 0.]],
            [[float(self.PITCH_W), float(self.PITCH_H)]],
            [[0.,    float(self.PITCH_H)]],
        ], dtype=np.float32)

        try:
            H_inv  = np.linalg.inv(H)
            mapped = cv2.perspectiveTransform(corners, H_inv)
        except Exception:
            return False

        if not np.all(np.isfinite(mapped)):
            return False

        xs = mapped[:, 0, 0]
        ys = mapped[:, 0, 1]
        if (xs.max() - xs.min()) < 100 or (ys.max() - ys.min()) < 100:
            return False

        return True
