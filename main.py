import utils
from config import Config

CONFIG = Config()

def main():
    """
    Docstring for main
    """


    input_frames = utils.read_video(CONFIG.input_video_path)
    