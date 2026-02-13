import cv2
import numpy as np
import sys
import os

def prepare_bandaid(img):
    """Refined extraction of the band-aid asset."""
    if img is None: return None
    if img.shape[2] == 4:
        return img
    
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    img_bgra[:, :, 3] = cv2.medianBlur(mask, 5)
    return img_bgra

def detect_and_fit_bandaid(image, ba_img):
    """Detects red/wound areas and fits a band-aid."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red Detection
    lower1 = np.array([0, 100, 50])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 50])
    upper2 = np.array([180, 255, 255])

    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))

    # Morphological operations to bridge gaps in the cut
    kernel = np.ones((15,15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 100: return None

    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), angle = rect

    if w < h:
        w, h = h, w
        angle += 90

    # Ensure band-aid is significantly larger than the wound
    target_w = int(w * 2.0)
    target_w = max(min(target_w, int(image.shape[1] * 0.5)), 150)

    aspect_ratio = ba_img.shape[0] / ba_img.shape[1]
    target_h = int(target_w * aspect_ratio)
    
    ba_resized = cv2.resize(ba_img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    # Rotation
    pad = max(target_w, target_h)
    ba_padded = cv2.copyMakeBorder(ba_resized, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0,0,0,0))
    M = cv2.getRotationMatrix2D((ba_padded.shape[1]//2, ba_padded.shape[0]//2), angle, 1.0)
    ba_rotated = cv2.warpAffine(ba_padded, M, (ba_padded.shape[1], ba_padded.shape[0]))

    sx = int(cx - ba_rotated.shape[1]/2)
    sy = int(cy - ba_rotated.shape[0]/2)
    
    return overlay_alpha(image.copy(), ba_rotated, sx, sy)

def overlay_alpha(background, overlay, x, y):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]
    x1, x2 = max(x, 0), min(x + ow, bw)
    y1, y2 = max(y, 0), min(y + oh, bh)
    if x1 >= x2 or y1 >= y2: return background
    ox1, oy1 = max(0, -x), max(0, -y)
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    overlay_chunk = overlay[oy1:oy2, ox1:ox2]
    background_chunk = background[y1:y2, x1:x2]
    mask = overlay_chunk[:, :, 3] / 255.0
    for c in range(3):
        background[y1:y2, x1:x2, c] = (mask * overlay_chunk[:, :, c] + (1.0 - mask) * background_chunk[:, :, c])
    return background

def create_comparison(original, processed):
    """Creates a side-by-side image with labels."""
    # Add labels to the images
    before = original.copy()
    after = processed.copy()
    
    cv2.putText(before, "BEFORE (Wound Detected)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(after, "AFTER (Band-Aid Applied)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Stack horizontally
    comparison = np.hstack((before, after))
    
    # Scale down if too large for typical screens
    max_width = 1400
    if comparison.shape[1] > max_width:
        scale = max_width / comparison.shape[1]
        comparison = cv2.resize(comparison, None, fx=scale, fy=scale)
        
    return comparison

def start_live_capture(ba_img):
    cap = cv2.VideoCapture(0)
    print("Live Feed Started. [S] Snap Comparison | [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        preview = cv2.flip(frame, 1)
        cv2.putText(preview, "Press 'S' to see Before/After", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Webcam Live Feed", preview)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            result = detect_and_fit_bandaid(frame, ba_img)
            if result is not None:
                # Create the side-by-side comparison
                comp = create_comparison(frame, result)
                cv2.imshow("Before vs After", comp)
                cv2.imwrite("comparison_result.png", comp)
                print("Comparison saved as comparison_result.png")
            else:
                print("No wound detected. Try closer or better light.")
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    ba_path = "bandaid.png"
    if not os.path.exists(ba_path):
        print(f"Error: {ba_path} not found.")
        return

    ba_img = cv2.imread(ba_path, cv2.IMREAD_UNCHANGED)
    ba_img = prepare_bandaid(ba_img)

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is not None:
            result = detect_and_fit_bandaid(img, ba_img)
            if result is not None:
                comp = create_comparison(img, result)
                cv2.imshow("Comparison", comp)
                cv2.waitKey(0)
            else:
                print("No wound detected.")
        else:
            print("Invalid image.")
    else:
        start_live_capture(ba_img)

if __name__ == "__main__":
    main()
