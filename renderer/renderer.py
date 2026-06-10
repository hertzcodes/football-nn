import cv2
import numpy as np

class PitchDrawer:
    WIDTH  = 1050
    HEIGHT = 680
    _L_PEN_X  = 165
    _L_GA_X   = 220
    _PEN_Y1   = 138
    _PEN_Y2   = 541
    _GA_Y1    = 248
    _GA_Y2    = 431
    _L_SPOT_X = 110
    _R_SPOT_X = 940

    WHITE = (255, 255, 255)
    LINE_W = 2

    def __init__(self) -> None:
        self.width  = self.WIDTH
        self.height = self.HEIGHT

    def create_pitch(self) -> np.ndarray:
        """Return a freshly drawn 1050 x 680 BGR pitch image."""
        pitch = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)

        for i in range(0, self.WIDTH, 80):
            color = (40, 140, 40) if (i // 80) % 2 == 0 else (30, 120, 30)
            cv2.rectangle(pitch, (i, 0), (i + 80, self.HEIGHT), color, -1)

        W = self.WHITE
        LW = self.LINE_W

        cv2.rectangle(pitch, (0, 0), (self.WIDTH - 1, self.HEIGHT - 1), W, LW + 1)

        mid_x = self.WIDTH // 2
        cv2.line(pitch, (mid_x, 0), (mid_x, self.HEIGHT), W, LW)

        centre = (mid_x, self.HEIGHT // 2)
        cv2.circle(pitch, centre, 92, W, LW)
        cv2.circle(pitch, centre, 4,  W, -1)

        cv2.rectangle(
            pitch,
            (0,              self._PEN_Y1),
            (self._L_PEN_X,  self._PEN_Y2),
            W, LW,
        )

        cv2.rectangle(
            pitch,
            (0,             self._GA_Y1),
            (self._L_GA_X,  self._GA_Y2),
            W, LW,
        )

        cv2.circle(pitch, (self._L_SPOT_X, self.HEIGHT // 2), 4, W, -1)

        cv2.ellipse(
            pitch,
            (self._L_SPOT_X, self.HEIGHT // 2),
            (92, 92),
            0,
            -53, 53,   
            W, LW,
        )

        cv2.rectangle(
            pitch,
            (self.WIDTH - self._L_PEN_X, self._PEN_Y1),
            (self.WIDTH,                  self._PEN_Y2),
            W, LW,
        )

        cv2.rectangle(
            pitch,
            (self.WIDTH - self._L_GA_X, self._GA_Y1),
            (self.WIDTH,                 self._GA_Y2),
            W, LW,
        )

        cv2.circle(pitch, (self._R_SPOT_X, self.HEIGHT // 2), 4, W, -1)

        cv2.ellipse(
            pitch,
            (self._R_SPOT_X, self.HEIGHT // 2),
            (92, 92),
            0,
            127, 233,
            W, LW,
        )

        goal_h = 73   
        goal_y1 = (self.HEIGHT - goal_h) // 2
        goal_y2 = goal_y1 + goal_h
        goal_d  = 20
        cv2.rectangle(pitch, (0,                    goal_y1), (goal_d, goal_y2), W, LW)
        cv2.rectangle(pitch, (self.WIDTH - goal_d,  goal_y1), (self.WIDTH, goal_y2), W, LW)

        return pitch


class Renderer:
    def __init__(self, homography=None):
        self.homography = homography
        self.pitch_drawer = PitchDrawer()

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame
    
    def draw_birds_eye_view(
        self,
        frame: np.ndarray,
        player_dict: dict,
        ball_dict: dict,
        goalkeeper_dict: dict | None = None,
    ) -> np.ndarray:
        if self.homography is None or self.homography.H is None:
            return frame

        pitch = self.pitch_drawer.create_pitch()

        PITCH_W = self.homography.PITCH_W
        PITCH_H = self.homography.PITCH_H
        MARGIN = 15

        def in_bounds(x, y):
            return (
                -MARGIN <= x <= PITCH_W + MARGIN and
                -MARGIN <= y <= PITCH_H + MARGIN
            )

        def clamp(x, y):
            return (
                max(0, min(PITCH_W - 1, x)),
                max(0, min(PITCH_H - 1, y)),
            )

        # Draw players (note: goalkeepers are merged into player_dict in the main project)
        for track_id, player in player_dict.items():
            foot   = get_foot_position(player["bounding_box"])
            mapped = self.homography.transform(foot)
            if mapped is None:
                continue

            if not in_bounds(mapped[0], mapped[1]):
                continue   

            x, y = clamp(mapped[0], mapped[1])
            color = player.get("team_color", (0, 0, 255))
            cv2.circle(pitch, (x, y), 12, color,           -1)
            cv2.circle(pitch, (x, y), 12, (255, 255, 255),  2)

        # Draw extra goalkeepers if passed separately
        if goalkeeper_dict:
            for track_id, gk in goalkeeper_dict.items():
                foot   = get_foot_position(gk["bounding_box"])
                mapped = self.homography.transform(foot)
                if mapped is None:
                    continue

                if not in_bounds(mapped[0], mapped[1]):
                    continue

                x, y = clamp(mapped[0], mapped[1])
                color = gk.get("team_color", (180, 0, 180))
                cv2.circle(pitch, (x, y), 12, color,           -1)
                cv2.circle(pitch, (x, y), 12, (255, 255, 255),  2)

        # Draw balls
        for _, ball in ball_dict.items():
            center = get_center_of_bbox(ball["bounding_box"])
            mapped = self.homography.transform(center)
            if mapped is None:
                continue

            if not in_bounds(mapped[0], mapped[1]):
                continue

            bx, by = clamp(mapped[0], mapped[1])
            cv2.circle(pitch, (bx, by), 8, (0, 255, 255), -1)
            cv2.circle(pitch, (bx, by), 8, (255, 255, 255), 2)

        mini = cv2.resize(pitch, (600, 380))
        h, w = mini.shape[:2]
        cv2.rectangle(mini, (0, 0), (w - 1, h - 1), (0, 0, 0), 4)

        ox = frame.shape[1] - w - 20
        oy = frame.shape[0] - h - 20

        if ox < 0 or oy < 0:
            return frame

        overlay = frame.copy()
        overlay[oy: oy + h, ox: ox + w] = mini
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        return frame

    def render_items(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            # Update homography matrix with interpolation and optical flow for this frame
            if self.homography is not None:
                prev_frame = video_frames[frame_num - 1] if frame_num > 0 else None
                self.homography.update_for_frame(frame_num, prev_frame, frame)

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["others"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bounding_box"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bounding_box"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bounding_box"], (0, 255, 255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bounding_box"], (0, 255, 0))
            
            # Draw bird's eye view minimap
            frame = self.draw_birds_eye_view(
                frame, player_dict, ball_dict
            )

            output_video_frames.append(frame)

        return output_video_frames


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    return bbox[2] - bbox[0]


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)