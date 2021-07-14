from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import cv2
import pickle
import joblib
import imutils

# 함수 정의
####### 필요한 함수 ########   
def RGB_extracter(img_bgr):
    img = img_bgr
    feature = []
    load_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(3):
        channel = load_img[:, :, i]
        indices_X, indices_Y = np.where(channel >= 5) # padding(검은색)부위에서 값이 튀는 경우가 있어서 5 이상으로 설정함
        mean_color = np.mean(channel[indices_X,indices_Y]) # 원하는 구간 선정
        feature.append(int(mean_color))
    return feature

def HSV_extracter(img_bgr):
    img = img_bgr
    feature = []
    load_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i in range(3):
        channel = load_img[:, :, i]
        indices_X, indices_Y = np.where(load_img[:,:,2] >= 5) # padding(검은색)부위에서 값이 튀는 경우가 있어서 5 이상으로 설정함
        mean_color = np.mean(channel[indices_X,indices_Y]) # 원하는 구간 선정
        feature.append(int(mean_color))
    return feature

def LAB_extracter(img_bgr):
    img = img_bgr
    feature = []
    load_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    for i in range(3):
        channel = load_img[:, :, i]
        # padding(검은색)부위에서 값이 튀는 경우가 있어서 5 이상으로 설정함   # padding(검은색)부위에서 값이 튀는 경우가 있어서 5 이상으로 설정함
        # 원하는 구간 선정
        indices_X, indices_Y, _ = np.where(load_img[:,:,:] != [0, 128, 128])
        color = np.mean(channel[indices_X,indices_Y])
        feature.append(int(color))
    return feature

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=10):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness,style)

def mouse_centercrop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, cropImg
    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True
    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y
    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished
        refPoint = [(x_start-np.abs(x_start-x_end), y_start-np.abs(y_start-y_end)), (x_end, y_end)]
        
        if len(refPoint) == 2: #when two points were found
            #print(refPoint)
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            #cv2.imshow("Cropped", roi)
            list_circles = detectcircles(roi)
            try:
                cropImg = circleExtract(roi,list_circles, cut_r)
            except:
                print("Cropping is not proper")

def detectcircles(img):
        # Convert to grayscale. 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.GaussianBlur(gray, (3, 3),0)

    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray, 
                    cv2.HOUGH_GRADIENT, 1.2, 20, param1 = 50, 
                param2 = 30, minRadius = 50, maxRadius = 60) 

    # Draw circles that are detected. 
    if detected_circles is not None: 
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles))
    return detected_circles

def circleExtract(img,list_circles,cut_r):
    x, y, r = list_circles[0][0].astype(np.int32)
    croped = img[y - r: y + r, x - r: x + r]
    roi = croped.copy()
    #cv2.imshow('roi',roi)
    width, height = roi.shape[:2]
    mask = np.zeros((width, height, 3), roi.dtype)
    cv2.circle(mask, (int(width / 2), int(height / 2)), r-cut_r, (255, 255, 255), -1)
    #cv2.imshow("circlemask",mask)
    dst = cv2.bitwise_and(roi, mask)
    croppedimage = imutils.resize(dst, height=112)
    show_image = imutils.resize(dst, height=500)
    cv2.imshow('show result',show_image)

    return croppedimage
    # 파일이름 가져오고, 현재 보고있는 원형이미지 저장하기

def circleExtract_Auto(img,list_circles, cut_r):
    x, y, r = list_circles[0][0].astype(np.int32)
    croped = img[y - r: y + r, x - r: x + r]
    roi = croped.copy()
    #cv2.imshow('roi',roi)
    width, height = roi.shape[:2]
    mask = np.zeros((width, height, 3), roi.dtype)
    cv2.circle(mask, (int(width / 2), int(height / 2)), r-cut_r, (255, 255, 255), -1)
    #cv2.imshow("circlemask",mask)
    dst = cv2.bitwise_and(roi, mask)
    croppedimage = imutils.resize(dst, height=112)
    show_image = imutils.resize(dst, height=500)
    #cv2.imshow('show result',show_image)

    return croppedimage
###############################
x_start, y_start, x_end, y_end = 0, 0, 0, 0
cut_r = 20
# ML,DL에 사용할 image 양식
# cropImg 값과 croppedimage 값을 항상 일치 시켜 놓기
cropImg = np.zeros((112, 112, 3), np.uint8)
# 이미지 불러오기

# 이미지 자르기
img_raw = cv2.imread('test.jpg')
# 회전
load_img = imutils.rotate(img_raw, -90)
# 원본 이미지가 image shape : (3024, 4032, 3)
image = imutils.resize(load_img, height=1400)
oriImage = image.copy()

# 자르기를 원하는 위치
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)
# 타겟 반지름
rad = 80
#
refPoint = [(cX-rad, cY-rad), (cX+rad, cY+rad)]
#refPoint = [refPoint_tuple]

roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
    #cv2.imshow("Cropped", roi)
list_circles = detectcircles(roi)

cropImg = circleExtract_Auto(roi,list_circles, cut_r)
cv2.imshow('test',cropImg)
# 이미지에서 데이터 추출
rgb_feature = RGB_extracter(cropImg)
        ### HSV ###
hsv_feature = HSV_extracter(cropImg)
### RAB ###
lab_feature = LAB_extracter(cropImg)

# 추출한 feature들 합치기
All_feature = np.concatenate((rgb_feature, hsv_feature, lab_feature), axis=None)
Data_X = np.empty((0,9), dtype=np.uint8)
Data_X = np.append(Data_X, [All_feature], axis=0)
# 머신러닝 모델 불러오기
clf_from_joblib = joblib.load('trainedmodel_gbc.pkl') 
# 머신러닝 모델 적용
predicted_result = clf_from_joblib.predict(Data_X)
list_concentration = ['Neg', '1 aM', '10 aM', '100 aM','1 fM','10 fM','100 fM', '1 pM', '10 pM','100 pM', '1 nM', '10 nM']
# 결과 출력
print(list_concentration[predicted_result[0]])