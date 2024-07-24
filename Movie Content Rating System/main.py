import os
from ObjectDetector import ObjectDetector
from ActionRecognizer import ActionRecognizer


def main():
    # Define paths and parameters
    VIDEOS_DIR = os.path.join('.', 'videos')
    
    video_path = "/content/gdrive/MyDrive/fypDemo/videos/Tips to use when trying to quit smoking. - YouTube - Google Chrome 2024-07-05 19-14-32.mp4"
    object_detection_output_path = '{}_object_detection_out.mp4'.format(video_path)
    action_recognition_output_path = '{}_action_recognition_out.mp4'.format(video_path)

    # Object Detection
    object_detector = ObjectDetector(model_path='/content/gdrive/MyDrive/fypDemo/marijuanna/train6/weights/best.pt')
    object_detection_durations = object_detector.detect_objects(video_path, object_detection_output_path)

    # Action Recognition
    action_recognizer = ActionRecognizer(model_path='/content/gdrive/MyDrive/fypDemo/LRCN_model___Date_Time_2024_07_14__20_32_16___Loss_0.2624717652797699___Accuracy_0.910614550113678.h5', sequence_length=20)
    action_recognition_durations = action_recognizer.recognize_actions(video_path, action_recognition_output_path)

    print("Object Detection Durations:", object_detection_durations)
    print("Action Recognition Durations:", action_recognition_durations)


if __name__ == "__main__":
    main()
