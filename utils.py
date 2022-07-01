import cv2
import numpy


# Preprocessing Image
def preProcess(src):
    grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.GaussianBlur(grayScale, (5, 5), 1)
    thresholdImg = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
    return thresholdImg


# Reorder points for Warp Perspective
def reorder(points):
    points = points.reshape((4, 2))
    newPoints = numpy.zeros((4, 1, 2), dtype=numpy.int32)
    add = points.sum(1)
    newPoints[0] = points[numpy.argmin(add)]
    newPoints[3] = points[numpy.argmax(add)]
    diff = numpy.diff(points, axis=1)
    newPoints[1] = points[numpy.argmin(diff)]
    newPoints[2] = points[numpy.argmax(diff)]
    return newPoints


# FINDING THE BIGGEST CONTOUR
def biggestContur(contours):
    biggest = numpy.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


# SPLIT THE IMAGE INTO 81 DIFFERENT IMAGES
def splitBoxes(src):
    rows = numpy.vsplit(src, 9)
    boxes = []
    for i in rows:
        cols = numpy.hsplit(i, 9)
        for j in cols:
            boxes.append(j)
    return boxes


# GET PREDICTIONS ON ALL IMAGES
def getPredictions(boxes, model):
    result = []
    for image in boxes:
        img = numpy.asarray(image)
        img = img[4:img.shape[0]-4, 4:img.shape[1]-4]
        img = cv2.resize(img, (28, 28))
        img = img / 255
        img = img.reshape(1, 28, 28, 1)
        predictions = model.predict(img)
        classIndex = numpy.argmax(model.predict(img), axis=-1)
        probability = numpy.amax(predictions)
        if probability > 0.8:
            result.append(classIndex[0])
        else:
            result.append(0)
    return result


# DISPLAY NUMBERS ON AN IMAGE
def displayNumbers(src, numbers, color=(255, 0, 0)):
    secW = int(src.shape[1]/9)
    secH = int(src.shape[0]/9)
    for x in range(0, 9):
        for y in range(0, 9):
            if numbers[(y*9)+x] != 0:
                cv2.putText(src, str(numbers[(y * 9) + x]),
                            (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return src


def drawGrid(src):
    secW = int(src.shape[1]/9)
    secH = int(src.shape[0] / 9)
    for i in range(0, 9):
        pt1 = (0, secH*i)
        pt2 = (src.shape[1], secH*i)
        pt3 = (secW*i, 0)
        pt4 = (secW*i, src.shape[0])
        cv2.line(src, pt1, pt2, (255, 255, 0), 2)
        cv2.line(src, pt3, pt4, (255, 255, 0), 2)
    return src
