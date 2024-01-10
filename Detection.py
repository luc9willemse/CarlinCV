import cv2
import numpy as np
import Config, Tracker
import time


class Detection:

    def __init__(self, video_path, reference_image_path):
        self.video_path = video_path
        self.conf = Config.Config()
        self.tracker = Tracker.CustomTracker(self.conf.max_dis)
        self.object_detector = self.set_object_detector()
        self.reference_contour = self.create_reference_contour(reference_image_path)

    
    def create_reference_contour(self, reference_image_path):
        # Load the reference image
        reference_image = cv2.imread(reference_image_path, 0)
        # Apply a threshold to get a binary image
        _, binary_image = cv2.threshold(reference_image, 200, 255, cv2.THRESH_BINARY)
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Assuming the largest contour is the object
        reference_contour = max(contours, key=cv2.contourArea)
        return reference_contour


    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections, mask = self.object_detection(frame)
            tracked_objects = self.tracker.update(detections)
            for track_id, centroid in tracked_objects.items():
                x, y = centroid

                cv2.circle(frame, (int(x), int(y)), radius=5, color=(150, 150, 200), thickness=-1)
                cv2.putText(frame, str(track_id), (int(x) + 20, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 200), 2)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # time.sleep(0.1)

        cap.release()
        cv2.destroyAllWindows()

    
    def set_object_detector(self):
        return cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=100, detectShadows=False)

    
    def object_detection(self, frame):
        mask = self.object_detector.apply(frame)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.conf.min_area and area < self.conf.max_area:
                similarity = cv2.matchShapes(self.reference_contour, contour, cv2.CONTOURS_MATCH_I1, 0)
                # print(similarity)
                if similarity < self.conf.some_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    detections.append((x, y, w, h))

        return detections, mask
    



