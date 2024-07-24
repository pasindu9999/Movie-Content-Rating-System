import cv2
import numpy as np
from collections import deque, defaultdict
from tensorflow.keras.models import load_model

class ActionRecognizer:
    def __init__(self, model_path, sequence_length, image_height=64, image_width=64, classes_list=None):
        self.model = load_model(model_path)
        self.sequence_length = sequence_length
        self.image_height = image_height
        self.image_width = image_width
        self.classes_list = classes_list if classes_list else ["snort", "nonsnort"]

    def recognize_actions(self, video_path, output_path):
        video_reader = cv2.VideoCapture(video_path)
        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_reader.get(cv2.CAP_PROP_FPS)

        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, (original_video_width, original_video_height))
        frames_queue = deque(maxlen=self.sequence_length)
        predicted_class_name = ''
        class_duration = defaultdict(int)

        while video_reader.isOpened():
            ok, frame = video_reader.read()
            if not ok:
                break

            resized_frame = cv2.resize(frame, (self.image_height, self.image_width))
            normalized_frame = resized_frame / 255
            frames_queue.append(normalized_frame)

            if len(frames_queue) == self.sequence_length:
                predicted_labels_probabilities = self.model.predict(np.expand_dims(frames_queue, axis=0))[0]
                predicted_label = np.argmax(predicted_labels_probabilities)
                predicted_class_name = self.classes_list[predicted_label]
                class_duration[predicted_class_name] += 1

            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            video_writer.write(frame)

        video_reader.release()
        video_writer.release()

        class_duration_in_seconds = {cls: (frames / fps) for cls, frames in class_duration.items()}
        return class_duration_in_seconds