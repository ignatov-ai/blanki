# импортирование необходимых библиотек
import numpy as np
import cv2
import imutils

# подготовка изображения к распознаванию
def imageProcessing(image):
    # конвертация изображения в градации серого. Это уберёт цветовой шум
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # размытие картинки, чтобы убрать высокочастотный шум
    # это помогает определить контур в сером изображении
    grayImageBlur = cv2.blur(grayImage,(3,3))

    # теперь производим определение границы по методу Canny
    edgedImage = cv2.Canny(grayImageBlur, 100, 300, 3)

    # показать серое изображение с определенными границами
    #cv2.imshow("grayBlur", grayImageBlur)
    #cv2.imshow("Edge Detected Image", edgedImage)

    # найти контуры на обрезанном изображении, рационально организовать область
    # оставить только большие варианты
    allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    allContours = imutils.grab_contours(allContours)

    # сортировка контуров области по уменьшению и сохранение топ-1
    allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]

    # aппроксимация контура
    perimeter = cv2.arcLength(allContours[0], True)
    ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02*perimeter, True)

    # показать контуры на изображении
    cv2.drawContours(image, [ROIdimensions], -1, (0,0,255), 2)
    #cv2.imshow('Contour Outline', image)

    # изменение массива координат
    ROIdimensions = ROIdimensions.reshape(4,2)

    # список удержания координат ROI
    rect = np.zeros((4,2), dtype='float32')

    # наименьшая сумма будет у верхнего левого угла,
    # наибольшая - у нижнего правого угла
    s = np.sum(ROIdimensions, axis=1)
    rect[0] = ROIdimensions[np.argmin(s)]
    rect[2] = ROIdimensions[np.argmax(s)]

    # верх-право будет с минимальной разницей
    # низ-лево будет иметь максимальную разницу
    diff = np.diff(ROIdimensions, axis=1)
    rect[1] = ROIdimensions[np.argmin(diff)]
    rect[3] = ROIdimensions[np.argmax(diff)]

    # верх-лево, верх-право, низ-право, низ-лево
    (tl, tr, br, bl) = rect

    # вычислить ширину ROI
    widthA = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
    widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
    maxWidth = max(int(widthA), int(widthB))

    # вычислить высоту ROI
    heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
    heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
    maxHeight = max(int(heightA), int(heightB))

    # набор итоговых точек для обзора всего документа
    # размер нового изображения
    dst = np.array([
        [0,0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]], dtype="float32")

    # вычислить матрицу перспективного преобразования и применить её
    transformMatrix = cv2.getPerspectiveTransform(rect, dst)

    # преобразовать ROI
    scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))

    # давайте посмотрим на свёрнутый документ
    #cv2.imshow("Scaned",scan)

    # конвертация в серый
    scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)

    # показать финальное серое изображение
    #cv2.imshow("scanGray", scanGray)

    # конвертация в черно-белое с высоким контрастом для документов
    from skimage.filters import threshold_local
    # увеличить контраст в случае с документом
    T = threshold_local(scanGray, 29, offset=28, method="gaussian")
    scanBW = (scanGray > T).astype("uint8") * 255
    # показать финальное изображение с высоким контрастом
    #cv2.imshow("scanBW", scanBW)
    return scanBW

# Повернуть на 90 градусов по часовой стрелке
def RotateClockWise90(img):
    trans_img = cv2.transpose( img )
    new_img = cv2.flip(trans_img, 1)
    return new_img

# подсчет количества черных пикселей и процентного соотношения черных пикселей к белым, ориентация изображения
def pageOrientation(scanBW):
    for i in range(4):
        scanBW = RotateClockWise90(scanBW)
        height, width = scanBW.shape[:2]

        dx,dy = 5,5
        x,y = width, int(height * 0.031)
        x_crop = width - x,width
        y_crop = height - y,height
        s = (width-2*dx) * (y-dy)
        print(x_crop, y_crop)

        crop_img = scanBW[height - y:height-dy, dx:width-dx]
        count = np.count_nonzero(crop_img)
        dcount = s-count
        if count != 0:
            blackProcent = round(dcount/count*100,3)
        else:
            blackProcent = 0
        #cv2.imshow("Rotated image", scanBW)
        #cv2.imshow("cropped", crop_img)
        print(s, count, s-count, blackProcent)
        print(blackProcent)

        if 0 < blackProcent < 6:
            return scanBW
            break

        #cv2.waitKey(3000)
        #cv2.destroyWindow("cropped")

# параметр для сканируемого изображения
args_image = 'blanki/02.jpg'

# прочитать изображение
img_original = cv2.imread(args_image)
height, width = img_original.shape[:2]
new_size = width // 2, height // 2
image = cv2.resize(img_original,new_size)
orig = image.copy()

processing = imageProcessing(image)
orient = pageOrientation(processing)

#cv2.imshow("Original image", orig)
cv2.imshow("Rotated image", orient)
cv2.waitKey(0)
cv2.destroyAllWindows()