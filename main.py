import os
import cv2.cv2
import utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras.models import load_model
from utils import *
import solution


def solveSudokuImg():
    # 1. PREPARE THE IMAGE
    src = cv2.cv2.resize(cv2.cv2.imread(pathImg), (height, width))
    blankImg = numpy.zeros((height, width, 3), numpy.uint8)
    thresholdImg = preProcess(src)

    # 2. FIND ALL CONTOURS
    srcContur = src.copy()
    srcBigContur = src.copy()
    contours, hierarchy = cv2.cv2.findContours(thresholdImg, cv2.cv2.RETR_EXTERNAL, cv2.cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(srcContur, contours, -1, (0, 0, 255), 3)

    # 3. FIND THE BIGGEST CONTOUR AND USE IT AS SUDOKU
    biggestContur, maxArea = utils.biggestContur(contours)
    biggestContur = reorder(biggestContur)
    cv2.drawContours(srcBigContur, biggestContur, -1, (0, 0, 255), 25)
    pts1 = numpy.float32(biggestContur)
    pts2 = numpy.float32([[0, 0], [width, 0], [0, height], [width, height]])
    mat = cv2.getPerspectiveTransform(pts1, pts2)
    srcWarpColored = cv2.warpPerspective(src, mat, (width, height))
    srcDetectDigits = blankImg.copy()
    srcWarpColored = cv2.cvtColor(srcWarpColored, cv2.cv2.COLOR_RGB2GRAY)

    # 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
    srcSolveDigits = blankImg.copy()
    boxes = splitBoxes(srcWarpColored)
    numbers = getPredictions(boxes, model)
    srcDetectedDigits = displayNumbers(srcDetectDigits, numbers, color=(255, 0, 0))
    numbers = numpy.asarray(numbers)
    posArray = numpy.where(numbers > 0, 0, 1)

    # 5. FIND SOLUTION OF THE BOARD
    board = numpy.array_split(numbers, 9)
    try:
        solution.solveSudoku(board)
    except:
        print("The solution not found")
        pass
    flatList = []
    for sublist in board:
        for item in sublist:
            flatList.append(item)
    srcSolveDigits = displayNumbers(srcSolveDigits, flatList*posArray)

    # 6. OVERLAY SOLUTION
    pts2 = numpy.float32(biggestContur)
    pts1 = numpy.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    srcInvWarpColored = src.copy()
    srcInvWarpColored = cv2.warpPerspective(srcSolveDigits, matrix, (width, height))
    inv_perspective = cv2.addWeighted(srcInvWarpColored, 1, src, 0.5, 1)
    srcDetectedDigits = drawGrid(srcDetectDigits)
    srcSolveDigits = drawGrid(srcSolveDigits)

    # DISPLAY IMAGES
    cv2.cv2.imshow('Source Image', src)
    cv2.cv2.imshow('Threshold Image', thresholdImg)
    cv2.cv2.imshow('Contours', srcContur)
    cv2.cv2.imshow('The Biggest Contour', srcBigContur)
    cv2.cv2.imshow('Matrix Sudoku', srcDetectDigits)
    cv2.cv2.imshow('Solution', srcSolveDigits)
    cv2.cv2.imshow('Final Result', inv_perspective)
    cv2.cv2.waitKey(0)


value = 1
while value != 0:
    value = input("Please introduce number for a photo:\n")
    if value == "0":
        break
    else:
        height = 450
        width = 450
        model = load_model('Resurse/model.h5')
        pathImg = "Resurse/" + value + ".jpg"
        solveSudokuImg()
