
#gesture_index = 0
#GESTURE_NAME = "thumbsup"

#SAVE_ROOT = "my_dataset/data"
#os.makedirs(SAVE_ROOT, exist_ok=True)


# === record_gestures.py (uppdaterad för att spara blank) ===
import cv2
import numpy as np
import os

# === Gestklass (hårdkodat för exempelvis 'blank') ===
GESTURE_NAME = "thumbsup"
SAVE_ROOT = "my_dataset/data/thumbsup"
os.makedirs(SAVE_ROOT, exist_ok=True)

bg = None
accumWeight = 0.5
top, right, bottom, left = 10, 350, 225, 590

saving = False
save_count = 0
save_target = 100
frame_counter = 0

# === Segmenteringsfunktion: returnerar alltid mask ===
def segment(roi_bgr, bg, threshold=25, min_area=1000):
    if bg is None:
        return None

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_skin, upper_skin)

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    diff = cv2.absdiff(bg.astype("uint8"), gray)
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    combined_mask = cv2.bitwise_and(mask_hsv, thresholded)

    kernel = np.ones((4, 4), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    contours, _ = cv2.findContours(combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return (combined_mask, None)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return (combined_mask, None)

    return (combined_mask, largest)

# === Bakgrundsmodell ===
def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)

camera = cv2.VideoCapture(0)
num_frames = 0

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = cv2.resize(frame, (700, 500))
    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    roi = frame[top:bottom, right:left]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if num_frames < 30:
        run_avg(gray, accumWeight)
        if num_frames == 1:
            print("[STATUS] Kalibrerar bakgrund... håll undan handen.")
        elif num_frames == 29:
            print("[STATUS] Klar med bakgrund.")
    else:
        hand = segment(roi, bg)
        if hand is not None:
            thresholded, segmented = hand

            if segmented is not None:
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255), 2)

            if saving:
                if frame_counter % 2 == 0:
                    count = len(os.listdir(SAVE_ROOT))
                    filename = os.path.join(SAVE_ROOT, f"{GESTURE_NAME}_{count:04d}.png")
                    cv2.imwrite(filename, thresholded)
                    print(f"[SAVED] {filename}")
                    save_count += 1
                if save_count >= save_target:
                    saving = False
                    save_count = 0
                    print(f"[INFO] Sparat {save_target} bilder till '{GESTURE_NAME}'")

            cv2.imshow("Thresholded", thresholded)

    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(clone, f"Gesture: {GESTURE_NAME}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    if saving:
        cv2.putText(clone, f"Saving: {save_count}/{save_target}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("Video Feed", clone)
    keypress = cv2.waitKey(1) & 0xFF

    if keypress == ord("s") and not saving:
        print(f"[INFO] Startar batch-inspelning för: {GESTURE_NAME}")
        saving = True
        save_count = 0

    if keypress == ord("q"):
        break

    num_frames += 1
    frame_counter += 1

camera.release()
cv2.destroyAllWindows()