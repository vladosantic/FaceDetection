import cv2

kaskade_lica = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
kaskade_oci = cv2.CascadeClassifier("haarcascade_eye.xml")
kaskade_osmijeh = cv2.CascadeClassifier("haarcascade_smile.xml")

WebKamera = cv2.VideoCapture(0)

while True:
    ret, img = WebKamera.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lica = kaskade_lica.detectMultiScale(gray_img, 1.25, 4)

    for (x, y, w, h) in lica:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(img, 'Lice', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        snimka_siva_boja = gray_img[y:y + h, x:x + w]
        snimka_u_boji = img[y:y + h, x:x + w]

    oci = kaskade_oci.detectMultiScale(snimka_siva_boja)
    for (ex, ey, ew, eh) in oci:
        cv2.rectangle(snimka_u_boji, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

    cv2.imshow('Detekcija lica', img)

    if cv2.waitKey(30) & 0xff == ord('q'):
        break

WebKamera.release()
cv2.destroyAllWindows()