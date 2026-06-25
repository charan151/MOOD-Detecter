# MOOD-Detecter
# this is written in python code and also it wont work perfectlly
import cv2

# Load the face and smile detection models built straight into OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
print("Starting lightweight detector... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Isolate the face region to look for a smile inside it
        roi_gray = gray[y:y+h, x:x+w]
        
        # smile parameters: image, scaleFactor, minNeighbors, minSize
        # Tweaking minNeighbors higher (e.g., 25) avoids false smile triggers
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)

        # Basic mood logic rule
        mood = "HAPPY" if len(smiles) > 0 else "NEUTRAL / THOUGHTFUL"

        # Display the mood text over your face bounding box
        cv2.putText(frame, f"Mood: {mood}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Lightweight Mood Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
