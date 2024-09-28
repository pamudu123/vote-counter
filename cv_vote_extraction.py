import cv2
import numpy as np
import os
import utils


# Constants
TEMPLATE_DIR = "templates"
THRESHOLD = 0.6
RESIZE_DIM = (480, 640)
GAUSSIAN_BLUR_KERNEL = (5, 5)
CANNY_THRESH = (0, 125)
KERNEL_SIZE = (3, 3)
MIN_LINE_WIDTH = 300
CONTOUR_COLOR = (0, 255, 0)
RECT_COLOR = (0, 255, 255)
LINE_COLOR = (36, 255, 12)

# Utility class for image operations
class ImageProcessor:
    def __init__(self, template_dir, threshold=THRESHOLD):
        self.templates = self._load_templates(template_dir)
        self.threshold = threshold

    def _load_templates(self, template_dir):
        """Load template images for symbol matching."""
        templates = {}
        for filename in os.listdir(template_dir):
            if filename.endswith('.png'):
                symbol = filename.split('.')[0]
                template = cv2.imread(os.path.join(template_dir, filename), 0)
                templates[symbol] = template
        return templates

    def identify_symbol(self, image):
        """Identify the symbol in the given image using template matching."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        best_match = None
        best_val = -np.inf
        
        for symbol, template in self.templates.items():
            if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
                scale = min(image.shape[0] / template.shape[0], image.shape[1] / template.shape[1])
                new_size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
                template = cv2.resize(template, new_size, interpolation=cv2.INTER_AREA)

            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val and max_val > self.threshold:
                best_match = symbol
                best_val = max_val
                best_loc = max_loc

        if best_match:
            return best_match, best_loc, best_val
        else:
            return "Unknown", None, None

# Function to process the image and detect horizontal lines
def detect_horizontal_lines(image):
    """Detect horizontal lines in the warped binary image."""
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    contours, _ = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    horizontal_lines = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > MIN_LINE_WIDTH:
            horizontal_lines.append((x, y, x + w, y + h))
    return horizontal_lines

# Function to divide the image based on detected lines
def divide_image_by_lines(image, lines):
    """Divide the image into sub-images based on the provided horizontal lines."""
    lines.sort(key=lambda line: line[1])  # Sort by y-coordinate
    height, width = image.shape[:2]
    lines = [(0, 0, width, 0)] + lines + [(0, height, width, height)]

    sub_images = []
    for i in range(len(lines) - 1):
        y1 = lines[i][1]
        y2 = lines[i + 1][1]
        sub_image = image[y1:y2, 0:width]
        sub_images.append(sub_image)
    return sub_images

# Function to process and warp the perspective of the image
def process_image(image_path):
    """Read, resize, and process the image for further operations."""
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, RESIZE_DIM)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, GAUSSIAN_BLUR_KERNEL, 1)
    img_edges = cv2.Canny(img_blur, *CANNY_THRESH)
    
    kernel = np.ones(KERNEL_SIZE)
    img_dilated = cv2.dilate(img_edges, kernel, iterations=2)
    img_threshold = cv2.erode(img_dilated, kernel, iterations=1)

    return img_resized, img_threshold, img_gray

# Function to find and warp the largest contour
def warp_perspective(image, threshold_image):
    """Find the largest contour and warp the perspective of the image."""
    contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [cv2.contourArea(c) for c in contours]
    max_area_contour = contours[np.argmax(contour_areas)]

    x, y, w, h = cv2.boundingRect(max_area_contour)
    pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    pts2 = np.float32([[0, 0], [RESIZE_DIM[0], 0], [0, RESIZE_DIM[1]], [RESIZE_DIM[0], RESIZE_DIM[1]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    img_warped = cv2.warpPerspective(image, matrix, RESIZE_DIM)
    img_warped_bw = cv2.warpPerspective(threshold_image, matrix, RESIZE_DIM)

    return img_warped, img_warped_bw

def main():
    # Load and process the image
    image_path = "sample_ballot_papers/vote_5.png"
    img_resized, img_threshold, img_gray = process_image(image_path)
    
    # Warp perspective
    img_warped, img_warped_bw = warp_perspective(img_resized, img_threshold)
    
    # Detect horizontal lines
    horizontal_lines = detect_horizontal_lines(img_warped_bw)

    # Divide image into sub-images
    extracted_img_rows = divide_image_by_lines(img_warped, horizontal_lines)

    # Initialize symbol matcher and process each row
    processor = ImageProcessor(TEMPLATE_DIR)
    vote_dict = {}
    row = 0

    for img_row in extracted_img_rows:
        if img_row.shape[0] > 40 and img_row.shape[1] > 40:
            row += 1
            symbol, location, confidence = processor.identify_symbol(img_row)
            vote_dict[row] = symbol
            print(f"Row {row}: {symbol}, {location}, {confidence}")


    # Display results
    imageArray = ([img_resized, img_threshold, img_gray],
                [img_warped, img_warped_bw, img_resized])

    stackedImage = utils.stackImages(imageArray, 0.6)
    cv2.imshow("Result",stackedImage)

    # SAVE IMAGE WHEN 's' key is pressed
    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
