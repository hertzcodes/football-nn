from typing import List, Dict, Any
from ultralytics import YOLO
import supervision as sv
import numpy as np

class Tracker:
    """
    Docstring for Tracker
    """
    def __init__(self, model_path: str, device: str):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.tracker: sv.ByteTrack = sv.ByteTrack()

    def __detect(self, frames: List[np.ndarray], conf: float,batch_size: int = 16) -> List[Any]:
        """
        Docstring for detect
        
        :param self: Description
        :param frames: Description
        :type frames: List[np.ndarray]
        :param batch_size: Description
        :type batch_size: int
        :return: Description
        :rtype: List[Any]
        """

        detections = []
        for i in range(0 , len(frames), batch_size):
            batch = frames[i:i+batch_size]
            result = self.model.predict(batch, conf=conf)
            detections += result

        return detections

    def track_detections(self, frames: List[np.ndarray]) -> Dict[str, List[Dict[int, Dict[str, Any]]]]:
        """
        Docstring for track_detections
        
        :param self: Description
        :param frames: Description
        :type frames: List[np.ndarray]
        :return: Description
        :rtype: Dict[str, List[Dict[int, Dict[str, Any]]]]
        """           

        detections = self.__detect(frames,0.1, 20)

        objects = {
            'ball': [],
            'players': [],
            'others': [], # refree or coaches
        }

        for idx, detection in enumerate(detections):
            classficiations: Dict[int, str] = detection.names

            detection_sv = sv.Detections.from_ultralytics(detection)

            objects['players'].append(dict())
            objects['ball'].append(dict())
            objects['others'].append(dict())

            for det in detection_sv:
                classification = det[3]
                bounding_box = det[0]
                class_name = classficiations.get(classification, "")

                if class_name == 'ball':
                    objects['ball'][idx][1] = {"bounding_box": bounding_box}

            detection_frame_tracks = self.tracker.update_with_detections(detection_sv)

            for det in detection_frame_tracks:
                classification = det[3]
                track_id = det[4]
                bounding_box = det[0].tolist()

                class_name = classficiations.get(classification, "")

                match class_name:
                    case "player":
                        objects['players'][idx][track_id] = {"bounding_box": bounding_box}
                    case "goalkeeper":
                        objects['players'][idx][track_id] = {"bounding_box": bounding_box}
                    case _:
                        objects['others'][idx][track_id] = {"bounding_box": bounding_box}

        return objects
