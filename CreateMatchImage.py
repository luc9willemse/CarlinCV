import cv2
import numpy as np

# Load your object image
object_image_path = r'C:\Users\User\Documents\Code\Carlin\CarlinCV\CarlinCV\videos\shape.png'  # Replace with your image path
object_image = cv2.imread(object_image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to get a binary image
# You may need to adjust the threshold value based on your image
_, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

# Find contours from the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming the largest contour in the image is your object
reference_contour = max(contours, key=cv2.contourArea)

# Create an empty black canvas with the same dimensions as the original image
canvas = np.zeros_like(object_image)

# Draw the contour on the canvas
# cv2.drawContours expects a list of contours, hence we wrap reference_contour in a list
cv2.drawContours(canvas, [reference_contour], -1, (0, 255, 0), 2)  # Draw in green with a thickness of 2

# Display the image with the drawn contour
cv2.imshow('Reference Contour', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
