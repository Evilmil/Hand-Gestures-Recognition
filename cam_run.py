# cam_run.py (uppdaterad)
import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model

# === Globala variabler ===
bg = None
accumWeight = 0.5
top, right, bottom, left = 10, 350, 225, 590

# === Bakgrundsmodell ===
def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, accumWeight)

# === Segmenteringsfunktion ===
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
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    return (combined_mask, largest)

# === Ladda modell ===
def _load_weights():
    try:
        model = load_model("hand_gesture_model.h5")
        print(model.summary())
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# === Prediktion ===
def getPredictedClass(model):
    image = cv2.imread('Temp.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 120))
    image = image.astype("float32") / 255.0
    image = image.reshape(1, 120, 100, 1)

    probs = model.predict(image)[0]
    predicted_class = np.argmax(probs)

    print ("Sannolikhet:", probs)

    classes = ["blank", "fist", "five", "ok", "thumbsdown", "thumbsup"]
    return classes[predicted_class]

# === Huvudprogram ===
if __name__ == "__main__":
    model = _load_weights()
    camera = cv2.VideoCapture(0)
    num_frames = 0
    k = 0

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        roi = frame[top:bottom, right:left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] Kalibrerar bakgrund... Håll undan handen.")
            elif num_frames == 29:
                print("[STATUS] Bakgrund kalibrerad!")
        else:
            hand = segment(roi, bg)
            if hand is not None:
                thresholded, segmented = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255), 2)

                if k % 5 == 0:
                    x, y, w, h = cv2.boundingRect(segmented)
                    hand_roi = thresholded[y:y+h, x:x+w]
                    #roi_resized = cv2.resize(hand_roi, (100, 120))
                    cv2.imwrite("Temp.png", hand_roi)

                    predictedClass = getPredictedClass(model)
                    cv2.putText(clone, str(predictedClass), (70, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Thresholded", thresholded)

        k += 1
        num_frames += 1
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.imshow("Video Feed", clone)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()