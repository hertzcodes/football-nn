from config import Config
import utils
from tracking import Tracker

def parse_args(argv, config: Config):
    """
    Docstring for parse_args
    
    :param argv: Description
    :param config: Description
    :type config: Config
    """

    return

def run_analyzer(args, config: Config):
    """
    Docstring for analyze
    
    :param args: Description
    :param config: Description
    :type config: Config
    """

    video_frames = utils.read_video(config.input_video_path)

    tracker = Tracker(config.Analyzer.player_model_path, config.device)

    tracks = tracker.track_detections(video_frames)

    print(tracks)
    return
