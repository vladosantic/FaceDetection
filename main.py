import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

WebCamFootage = cv2.VideoCapture(0)

while True:
    ret, img = WebCamFootage.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        rec_gray = gray_img[y:y + h, x:x + w]
        rec_color = img[y:y + h, x:x + w]

    cv2.imshow('Face Recognition', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

WebCamFootage.release()
cv2.destroyAllWindows()