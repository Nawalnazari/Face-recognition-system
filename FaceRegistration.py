import cv2
import numpy as np
import os
from PIL import Image
import time

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
face_detector = cv2.CascadeClassifier("C:/Users/SamChan/Documents/FYP/FYP/FYP Project/Cascades/haarcascade_frontalface_default.xml")
# Setiap user perlu daftar nama
name_id = str(input('\n Masukkan nama anda dan tekan (return) ==>  ')).lower()
folder1 = "C:/Users/SamChan/Documents/FYP/FYP/FYP Project/data/" + name_id


isExist = os.path.exists(folder1)

if isExist:
    print("nama sudah sedia ada !")
    name_id = str(input("Sila masukkan nama berbeza : "))
else :
    os.mkdir(folder1)



print("\n [INFO] Proses pengecaman wajah. Sila lihat kamera dan tunggu...")
# Initialize individual sampling face count
count = 0
while (True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, "Please make angry face", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        count += 1
        name1 = 'C:/Users/SamChan/Documents/FYP/FYP/FYP Project/data/' + name_id + '/' + str(count) + '.jpg'

        print("gambar sedang diambil........." + name1)

        # Menyimpan gambar yang diambil ke dalam folder dataset
        cv2.imwrite(name1, img[y:y + h, x:x + w])
        #cv2.imwrite(name2, gray[y:y + h, x:x + w])
        cv2.imshow('face detection', img)
        k = cv2.waitKey(100) & 0xff  # Tekan 'ESC' untuk keluar
        if k == 27:
            break

        elif count == 50:
            break

 # menutup system
print("\n [INFO] system ditutup....")
cam.release()
cv2.destroyAllWindows()
