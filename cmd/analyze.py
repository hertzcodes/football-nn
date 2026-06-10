from config import Config
import utils

from tracking import Tracker, PitchTracker
from assigner import Assigner
from renderer import Renderer
from homography import HomographyTransformer


def run_analyzer(args, config: Config) -> None:
    """
    Full analysis pipeline:
      1. Read video frames.
      2. Scan every 10 frames to build composite best-confidence keypoints → H.
      3. Track players, goalkeepers, ball, and others across all frames.
      4. Assign every player / goalkeeper to a team via jersey colour.
      5. Annotate frames and write output video.
    """

    video_frames = utils.read_video(config.input_video_path)
    print(f"Loaded {len(video_frames)} frames from {config.input_video_path}")

    tracker = Tracker(
        config.Analyzer.player_model_path,
        config.device,
    )

    print("Scanning video for best keypoint detections...")
    field_tracker = PitchTracker(config.Analyzer.field_model_path, config.device)

    homography = HomographyTransformer(field_tracker)
    homography.precompute_keyframes(video_frames)

    tracks = tracker.track_detections(video_frames)

    assigner = Assigner()

    bootstrap_players = {}
    for frame_idx in range(min(30, len(video_frames))):
        for track_id, player in tracks["players"][frame_idx].items():
            if track_id not in bootstrap_players:
                bootstrap_players[track_id] = player

    assigner.assign_team(video_frames[0], bootstrap_players)

    for frame_num, player_track in enumerate(tracks["players"]):
        frame = video_frames[frame_num]
        for track_id, track in player_track.items():
            pid = assigner.get_player_team(
                frame, track["bounding_box"], track_id,
            )
            if pid is None:
                continue
            team = assigner.get_team(pid)
            track["global_id"]  = pid
            track["team"]       = team
            track["team_color"] = assigner.get_team_color(team)

    renderer = Renderer(homography)
    output_frames = renderer.render_items(video_frames, tracks)
    utils.save_video(output_frames, config.output_video_path)
    print(f"Saved: {config.output_video_path}")
