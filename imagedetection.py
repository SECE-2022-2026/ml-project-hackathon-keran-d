import cv2
import numpy as np
import os


# 1. Region of Interest
def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    # Define a trapezoidal region
    polygon = np.array([[
        (int(0.1 * width), height),  # Bottom-left
        (int(0.9 * width), height),  # Bottom-right
        (int(0.6 * width), int(0.6 * height)),  # Top-right
        (int(0.4 * width), int(0.6 * height))   # Top-left
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# 2. Detect Edges
def detect_edges(img, canny_low=50, canny_high=150):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    return edges


# 3. Hough Transform for Line Detection
def hough_lines(edges, frame_shape, threshold=20, min_line_length=30, max_line_gap=100):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    line_img = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)  # Color image
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 10)  # Green color for lanes
    return line_img, lines


def overlay_lines(original, lines):
    overlay = cv2.addWeighted(original, 0.8, lines, 1, 0)
    return overlay


# Dynamic Parameter Adjustment
def adjust_parameters(video_path):
    def nothing(x):
        pass
    
    cap = cv2.VideoCapture(video_path)
    cv2.namedWindow('Adjustments')
    cv2.createTrackbar('Canny Low', 'Adjustments', 50, 255, nothing)
    cv2.createTrackbar('Canny High', 'Adjustments', 150, 255, nothing)
    cv2.createTrackbar('Hough Threshold', 'Adjustments', 20, 100, nothing)
    cv2.createTrackbar('Min Line Length', 'Adjustments', 30, 200, nothing)
    cv2.createTrackbar('Max Line Gap', 'Adjustments', 100, 200, nothing)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roi = region_of_interest(frame)
        
        # Get dynamic parameters
        canny_low = cv2.getTrackbarPos('Canny Low', 'Adjustments')
        canny_high = cv2.getTrackbarPos('Canny High', 'Adjustments')
        hough_threshold = cv2.getTrackbarPos('Hough Threshold', 'Adjustments')
        min_line_length = cv2.getTrackbarPos('Min Line Length', 'Adjustments')
        max_line_gap = cv2.getTrackbarPos('Max Line Gap', 'Adjustments')
        
        edges = detect_edges(roi, canny_low, canny_high)
        lines_img, lines = hough_lines(edges, frame.shape, hough_threshold, min_line_length, max_line_gap)
        overlay = overlay_lines(frame, lines_img)
        
        cv2.imshow('Adjustments', overlay)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path, output_path):
    img = cv2.imread(image_path)

    # Apply lane detection pipeline
    roi = region_of_interest(img)
    edges = detect_edges(roi)
    lines = hough_lines(edges, img.shape)
    overlay = overlay_lines(img, lines)

    # Save the output image
    cv2.imwrite(output_path, overlay)

# Example usage
# input_folder = r'C:\Users\keran\OneDrive\Documents\Desktop\lane_detection_project\data\images\road_line_images\road_line_images\dc_auto_000177_QW8Pqnw0.jpg'
input_folder = r'C:\Users\keran\OneDrive\Documents\Desktop\lane_detection_project\data\images\road_line_images'

output_folder = 'output/images'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for img_name in os.listdir(input_folder):
    process_image(os.path.join(input_folder, img_name), os.path.join(output_folder, img_name))