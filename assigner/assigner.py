import numpy as np
from sklearn.cluster import KMeans
import cv2

class Assigner:

    REVERIFY_INTERVAL: int = 10

    def __init__(self) -> None:
        self.model = KMeans(
            n_clusters=2,
            n_init=20,
            random_state=42,
        )
        self.fitted: bool = False

        self.memory: dict[int, np.ndarray] = {}

        self.player_team: dict[int, int] = {}


        self._frames_since_verify: dict[int, int] = {}

        self.team1_ids: set[int] = set()
        self.team2_ids: set[int] = set()
        self.team_colors: dict[int, tuple[int, int, int]] = {0: (0, 215, 255)}



    def assign_team(self, frame: np.ndarray, players: dict) -> None:
        features: list[np.ndarray] = []
        track_ids: list[int] = []

        for track_id, p in players.items():
            f = self._get_feature(frame, p["bounding_box"])
            if self._valid(f):
                features.append(f)
                track_ids.append(track_id)
                self.memory[track_id] = f

        if len(features) < 2:
            print("[Assigner] init_teams: not enough valid players to cluster.")
            return

        features_arr = np.array(features, dtype=np.float64)
        labels = self.model.fit_predict(features_arr)
        self.fitted = True

        for track_id, label in zip(track_ids, labels):
            team = 1 if label == 0 else 2
            self._set_team(track_id, team)

        for cluster_idx in range(2):
            center = self.model.cluster_centers_[cluster_idx]
            hsv_pixel = np.uint8([[center]])
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
            bgr_tuple = (int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2]))
            team_num = 1 if cluster_idx == 0 else 2
            self.team_colors[team_num] = bgr_tuple

        print(
            f"[Assigner] init_teams: "
            f"team1={len(self.team1_ids)} "
            f"team2={len(self.team2_ids)} players"
        )

    def get_team_color(self, team: int) -> tuple:
        return self.team_colors.get(team, self.team_colors.get(0, (0, 215, 255)))


    def get_player_team(
        self,
        frame: np.ndarray,
        bbox: list | np.ndarray,
        track_id: int,
    ) -> int | None:

        feature = self._get_feature(frame, bbox)

        if not self._valid(feature):
            return None

        if track_id not in self.memory:
            self.memory[track_id] = feature
            self._frames_since_verify[track_id] = 0
            self._assign_new_player(track_id)
        else:
            self.memory[track_id] = (
                0.85 * self.memory[track_id] + 0.15 * feature
            )


            self._frames_since_verify[track_id] = (
                self._frames_since_verify.get(track_id, 0) + 1
            )

            if self._frames_since_verify[track_id] >= self.REVERIFY_INTERVAL:
                self._frames_since_verify[track_id] = 0
                self._reverify(track_id)

        return track_id


    def get_team(self, player_id: int) -> int:
        """Return 1, 2, or 0 (unknown). Never raises KeyError."""
        return self.player_team.get(player_id, 0)


    def _set_team(self, track_id: int, team: int) -> None:
        """Central place to record a team assignment."""
        self.team1_ids.discard(track_id)
        self.team2_ids.discard(track_id)

        self.player_team[track_id] = team

        if team == 1:
            self.team1_ids.add(track_id)
        else:
            self.team2_ids.add(track_id)

    def _assign_new_player(self, pid: int) -> None:
        if not self.fitted:
            self._set_team(pid, 1)
            return

        feature = self.memory[pid]
        label = int(self.model.predict([feature.astype(np.float64)])[0])
        self._set_team(pid, 1 if label == 0 else 2)

    def _reverify(self, pid: int) -> None:
        """
        re-run KMeans prediction on the current smoothed feature and
        update the team if it has changed.
        """
        if not self.fitted:
            return

        feature = self.memory[pid]
        label = int(self.model.predict([feature.astype(np.float64)])[0])
        new_team = 1 if label == 0 else 2
        old_team = self.player_team.get(pid, 0)

        if new_team != old_team:
            print(
                f"[Assigner] ID {pid} re-verified: "
                f"team {old_team} → {new_team}"
            )
            self._set_team(pid, new_team)

    def _get_feature(
        self, frame: np.ndarray, bbox: list | np.ndarray
    ) -> np.ndarray:

        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros(3)

        h, w = crop.shape[:2]

        region = crop[
            int(h * 0.20): int(h * 0.55),
            int(w * 0.25): int(w * 0.75),
        ]

        if region.size == 0:
            return np.zeros(3)

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        pixels = hsv.reshape(-1, 3).astype(np.float32)


        v_channel = pixels[:, 2]
        pixels = pixels[(v_channel > 40) & (v_channel < 220)]

        if len(pixels) < 10:
            return np.zeros(3)

        return np.mean(pixels, axis=0)

    def _valid(self, f: np.ndarray) -> bool:
        return bool(np.linalg.norm(f) > 10)
