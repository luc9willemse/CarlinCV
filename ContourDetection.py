import cv2
import numpy as np

class CustomContourDetector:
    def __init__(self):
        pass

    def preprocess(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply a binary threshold
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return thresh

    def detect_edges(self, image):
        # Sobel operators
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        # Combine the two operators
        edge_image = cv2.magnitude(sobelx, sobely)
        _, edges = cv2.threshold(edge_image, 50, 255, cv2.THRESH_BINARY)
        return np.uint8(edges)

    def trace_contours(self, edge_image):
        # This is a very basic contour tracing method
        contours = []
        visited = np.zeros(edge_image.shape, np.uint8)

        for y in range(edge_image.shape[0]):
            for x in range(edge_image.shape[1]):
                if edge_image[y, x] != 0 and visited[y, x] == 0:
                    contour = self._trace_single_contour(edge_image, (x, y), visited)
                    contours.append(contour)
        return contours

    def _trace_single_contour(self, edge_image, start, visited):
        # A simple method to trace a single contour from a starting point
        contour = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-connectivity
        current = start
        prev_direction = 0

        while True:
            contour.append(current)
            visited[current[1], current[0]] = 1
            found_next = False

            for i in range(4):
                next_dir = (prev_direction + i) % 4
                next_point = (current[0] + directions[next_dir][0], current[1] + directions[next_dir][1])

                if edge_image[next_point[1], next_point[0]] != 0 and visited[next_point[1], next_point[0]] == 0:
                    current = next_point
                    prev_direction = next_dir
                    found_next = True
                    break

            if not found_next:
                break

        return contour

    def find_contours(self, image):
        # preprocessed = self.preprocess(image)
        edges = self.detect_edges(image)
        contours = self.trace_contours(edges)
        print(contours)
        return contours
