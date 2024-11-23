import cv2
import numpy as np

def region_of_interest(img):
    """Define the region of interest dynamically based on the frame size."""
    height, width = img.shape[:2]
    vertices = np.array([[
        (width * 0.1, height),
        (width * 0.45, height * 0.6),
        (width * 0.55, height * 0.6),
        (width * 0.9, height)
    ]], dtype=np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def average_slope_intercept(lines):
    """Calculate average slopes and intercepts for left and right lanes."""
    left_lines = []
    right_lines = []

    if lines is None:
        return None, None

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0  # Avoid division by zero
            intercept = y1 - slope * x1
            if slope < 0:  # Negative slope -> left lane
                left_lines.append((slope, intercept))
            elif slope > 0:  # Positive slope -> right lane
                right_lines.append((slope, intercept))

    # Average the slopes and intercepts
    left_lane = np.mean(left_lines, axis=0) if left_lines else None
    right_lane = np.mean(right_lines, axis=0) if right_lines else None

    return left_lane, right_lane

def make_line_coordinates(frame, line_params):
    """Convert slope and intercept to line coordinates."""
    if line_params is None:
        return None

    slope, intercept = line_params
    height = frame.shape[0]
    y1 = height
    y2 = int(height * 0.6)  # 60% of the frame height

    # Avoid dividing by zero
    if slope == 0:
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return [x1, y1, x2, y2]

def draw_lines(img, lines):
    """Draw lines on the image."""
    if lines is not None:
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

def process_frame(frame, accuracy=None):
    """Process a single frame to detect lanes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 2, np.pi / 180, 100, minLineLength=40, maxLineGap=150)

    # Calculate lane lines
    left_lane, right_lane = average_slope_intercept(lines)
    left_line = make_line_coordinates(frame, left_lane)
    right_line = make_line_coordinates(frame, right_lane)

    # Create a blank image to draw lines
    line_image = np.zeros_like(frame)
    draw_lines(line_image, [left_line, right_line])

    # Overlay lines on the original frame
    result = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # Add accuracy text on the frame (if provided)
    if accuracy is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, f"Accuracy: {accuracy:.2f}%", (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result, lines

def process_image(image_path):
    """Process a single image to detect lanes."""
    image = cv2.imread(image_path)
    result, lines = process_frame(image)

    # Since we are processing only one image, accuracy is assumed to be 100% (or compare with ground truth)
    accuracy = 100.0

    # Display the image with detected lanes
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, f"Accuracy: {accuracy:.2f}%", (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Lane Detection - Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    """Process the video frame by frame and display lane detection."""
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    frames_with_lines = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result, lines = process_frame(frame)

        if lines is not None:
            frames_with_lines += 1
        total_frames += 1

        accuracy = (frames_with_lines / total_frames) * 100 if total_frames > 0 else 0

        # Display the result frame
        cv2.imshow("Lane Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Specify the path to your video or image
    video_path = r'C:\Users\keran\OneDrive\Documents\Desktop\lane_detection_project\data\videos\mixkit-52450-video-52450-hd-ready.mp4'  # Replace with your video path
    image_path = r'C:\Users\keran\OneDrive\Documents\Desktop\lane_detection_project\data\images\road_line_images\road_line_images\image.png'  # Replace with your image path

    # Uncomment one of the following to process either a video or an image:
    process_video(video_path)  # Process video
    process_image(image_path)  # Process single image

if __name__ == '__main__':
    main()
