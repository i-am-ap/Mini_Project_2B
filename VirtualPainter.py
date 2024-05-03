import cv2
import numpy as np
import os
import HandTrackingModule as htm
from flask import Blueprint, render_template
from keras.models import load_model
import keyboard
import pygame
import time

VirtualPainter = Blueprint("HandTrackingModule", __name__, static_folder="static", template_folder="templates")

@VirtualPainter.route("/feature")
def strt():
    ############## Color Attributes ###############
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    GREEN = (0, 255, 0)
    BACKGROUND = (255, 255, 255)
    FORGROUND = (0, 255, 0)
    BORDER = (0, 255, 0)
    lastdrawColor = (0, 0, 1)
    drawColor = (0, 0, 255)
    BOUNDRYINC = 5

    ############## CV2 Attributes ###############
    cap = cv2.VideoCapture(0)
    width, height = 1280, 720
    cap.set(3, width)
    cap.set(4, height)
    imgCanvas = np.zeros((height, width, 3), np.uint8)

    ############## PyGame Attributes ###############
    pygame.init()
    DISPLAYSURF = pygame.display.set_mode((width, height), flags=pygame.HIDDEN)
    pygame.display.set_caption("Digit Board")
    number_xcord = []
    number_ycord = []

    ############## Header Files Attributes ###############
    folderPath = "Header"
    myList = os.listdir(folderPath)
    overlayList = []

    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)
    header = overlayList[0]

    ############## Predication Model Attributes ###############
    label = ""
    PREDICT = "off"
    AlphaMODEL = load_model("bModel.h5")
    NumMODEL = load_model("bestmodel.h5")
    AlphaLABELS = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                   10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
                   20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ''}
    # NumLABELS = {0: '0', 1: '1',
    #              2: '2', 3: '3',
    #              4: '4', 5: '5',
    #              6: '6', 7: '7',
    #              8: '8', 9: '9'}

    NumLABELS = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9'
                }

    rect_min_x, rect_max_x = 0, 0
    rect_min_y, rect_max_y = 0, 0

    ############## HandDetection Attributes ###############
    detector = htm.handDetector(detectionCon=0.85)
    xp, yp = 0, 0
    brushThickness = 15
    eraserThickness = 60   #Eraser
    modeValue = "OFF"
    modeColor = RED

    while True:
        SUCCESS, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        cv2.putText(img, "Press A for Alphabate Recognisition Mode ", (0, 145), 3, 0.5, (255, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "Press N for Digit Recognisition Mode ", (0, 162), 3, 0.5, (255, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img, "Press O for Turn Off Recognisition Mode ", (0, 179), 3, 0.5, (255, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img, f'{"RECOGNISITION IS "}{modeValue}', (0, 196), 3, 0.5, modeColor, 1, cv2.LINE_AA)

        if keyboard.is_pressed('a'):
            if PREDICT != "alpha":
                PREDICT = "alpha"
                modeValue, modeColor = "ALPHABATES", GREEN

        if keyboard.is_pressed('n'):
            if PREDICT != "num":
                PREDICT = "num"
                modeValue, modeColor = "NUMBER", YELLOW

        if keyboard.is_pressed('o'):
            if PREDICT != "off":
                PREDICT = "off"
                modeValue, modeColor = "OFF", RED

            xp, yp = 0, 0
            label = ""
            rect_min_x, rect_max_x = 0, 0
            rect_min_y, rect_max_y = 0, 0
            number_xcord = []
            number_ycord = []
            time.sleep(0.5)

        if len(lmList) > 0:

            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            fingers = detector.fingersUp()
            # print(fingers)

            if fingers[1] and fingers[2]:

                # add
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)

                if (len(number_xcord) > 0 and len(number_ycord) > 0 and PREDICT != "off"):
                    if drawColor != (0, 0, 0) and lastdrawColor != (0, 0, 0):
                        rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0), min(width,
                                                                                            number_xcord[-1] + BOUNDRYINC)
                        rect_min_y, rect_max_y = max(0, number_ycord[0] - BOUNDRYINC), min(
                            number_ycord[-1] + BOUNDRYINC, height)
                        number_xcord = []
                        number_ycord = []

                        img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[
                                  rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

                        cv2.rectangle(imgCanvas, (rect_min_x, rect_min_y), (rect_max_x, rect_max_y), BORDER, 3)
                        image = cv2.resize(img_arr, (28, 28))
                        # cv2.imshow("Tmp",image)
                        image = np.pad(image, (10, 10), 'constant', constant_values=0)
                        image = cv2.resize(image, (28, 28)) / 255
                        # cv2.imshow("Tmp",image)

                        if PREDICT == "alpha":
                            label = str(AlphaLABELS[np.argmax(AlphaMODEL.predict(image.reshape(1, 28, 28, 1)))])
                        if PREDICT == "num":
                            label = str(NumLABELS[np.argmax(NumMODEL.predict(image.reshape(1, 28, 28, 1)))])
                        pygame.draw.rect(DISPLAYSURF, BLACK, (0, 0, width, height))

                        cv2.rectangle(imgCanvas, (rect_min_x + 50, rect_min_y - 50), (rect_min_x, rect_min_y),
                                      BACKGROUND, -1)
                        cv2.putText(imgCanvas, label, (rect_min_x, rect_min_y - 5), 3, 2, FORGROUND, 1, cv2.LINE_AA)
                    else:
                        number_xcord = []
                        number_ycord = []

                xp, yp = 0, 0
                if y1 < 125:
                    lastdrawColor = drawColor
                    if 0 < x1 < 200:
                        imgCanvas = np.zeros((height, width, 3), np.uint8)
                    elif 210 < x1 < 320:
                        header = overlayList[0]
                        drawColor = (0, 0, 255)
                    elif 370 < x1 < 470:
                        header = overlayList[1]
                        drawColor = (0, 255, 255)
                    elif 520 < x1 < 630:
                        header = overlayList[2]
                        drawColor = (0, 255, 0)
                    elif 680 < x1 < 780:
                        header = overlayList[3]
                        drawColor = (255, 0, 0)
                    elif 890 < x1 < 1100:
                        header = overlayList[4]
                        drawColor = (0, 0, 0)
                    elif 1160 < x1 < 1250:
                        return render_template("index.html")

                cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

            elif fingers[1] and fingers[2] == False:

                # add
                number_xcord.append(x1)
                number_ycord.append(y1)
                # addEnd

                cv2.circle(img, (x1, y1 - 15), 15, drawColor, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                    pygame.draw.line(DISPLAYSURF, WHITE, (xp, yp), (x1, y1), brushThickness)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        img[0:132, 0:1280] = header
        pygame.display.update()
        cv2.imshow("Image", img)
        cv2.waitKey(1)

strt()
