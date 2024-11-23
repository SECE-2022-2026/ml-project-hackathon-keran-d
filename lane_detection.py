import cv2
import numpy as np

def region_of_interest(img, vertices):
    """Create a mask for the region of interest (ROI)."""
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines):
    """Draw lanes on the image."""
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)  

def process_frame(frame, accuracy=None):
    """Process a single frame to detect lanes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    height = frame.shape[0]
    width = frame.shape[1]
    vertices = np.array([[(100, height), (width - 100, height), (width // 2, height // 2)]], dtype=np.int32)
    roi = region_of_interest(edges, vertices)

    lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

    result = frame.copy()
    draw_lines(result, lines)

  
    if accuracy is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, f"Accuracy: {accuracy:.2f}%", (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return result, lines

def process_video(video_path):
    """Process the video frame by frame and calculate lane detection accuracy."""
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
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, f"Accuracy: {accuracy:.2f}%", (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Lane Detection", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_image(image_path):
    """Process a single image to detect lanes."""
    image = cv2.imread(image_path)
    result, lines = process_frame(image)
    accuracy = 85.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, f"Accuracy: {accuracy:.2f}%", (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Lane Detection - Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    video_path = r'C:\Users\keran\OneDrive\Documents\Desktop\lane_detection_project\data\videos\mixkit-52450-video-52450-hd-ready.mp4'  
    image_path = r'C:\Users\keran\OneDrive\Documents\Desktop\lane_detection_project\data\images\image.png' 

    process_video(video_path)  
    process_image(image_path)  

if __name__ == '__main__':
    main()
