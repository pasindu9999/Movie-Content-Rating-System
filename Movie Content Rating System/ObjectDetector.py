from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path, threshold=0.5):
        self.model = YOLO(model_path)
        self.threshold = threshold

    def detect_objects(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        H, W, _ = frame.shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        detection_durations = {}
        frame_count = 0

        while ret:
            if frame_count % fps == 0:
                results = self.model(frame)[0]
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result
                    if score > self.threshold:
                        class_name = results.names[int(class_id)].upper()
                        if class_name not in detection_durations:
                            detection_durations[class_name] = 0
                        detection_durations[class_name] += 1

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                        cv2.putText(frame, class_name, (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

            out.write(frame)
            ret, frame = cap.read()
            frame_count += 1

        cap.release()
        out.release()

        for class_name, duration in detection_durations.items():
            print(f"Object: {class_name}, Total Detection Duration: {duration:.2f} seconds")

        return detection_durations