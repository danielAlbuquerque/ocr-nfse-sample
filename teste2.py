import enum
import cv2
import numpy as np
import pytesseract
import os

roi = [[(374, 12), (469, 25), 'text', 'NUM']]

per = 25

imgQ = cv2.imread("sample.jpeg")
h, w, c = imgQ.shape
imgQ = cv2.resize(imgQ, (w//1,h//1))

orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ, None)
# impKp1 = cv2.drawKeypoints(imgQ, kp1, None)

path = 'nfse'

myPicList = os.listdir(path)
print(myPicList)

for j, y in enumerate(myPicList):
    img = cv2.imread(path + '/' + y)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)

    good = matches[:int(len(matches)*(per/100))]
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good, None, flags=2)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w,h))

    # cv2.imshow(y, imgScan)

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, ((r[0][0]), r[0][1]), ((r[1][0]), r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1,0)
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        cv2.imshow(str(x), imgCrop)

        if r[2] == 'text':
            print(f'{r[3]} : {pytesseract.image_to_string(imgCrop)}' )
            myData.append(pytesseract.image_to_string(imgCrop))
        
        cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]), cv2.FONT_HERSHEY_PLAIN, 2.5, (0,0,255), 4)

    with open('DataOutput.csv', 'a+') as f:
        for data in myData:
            f.write((str(data)+','))
        f.write('\n')
        
    cv2.imshow(y, imgShow)

# cv2.imshow("Output1", impKp1)
cv2.imshow("Output", imgQ)
cv2.waitKey()