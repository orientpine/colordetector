import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False) # deprecation 표시 안함 
st.title("SARS-CoV-2 Detection using Machine Learning")
import numpy as np
import cv2
import joblib
import imutils
from PIL import Image


# 가장 중요한 설정 값들
x_start, y_start, x_end, y_end = 0, 0, 0, 0
cut_r = 20
# ML,DL에 사용할 image 양식
# cropImg 값과 croppedimage 값을 항상 일치 시켜 놓기
cropImg = np.zeros((112, 112, 3), np.uint8)

list_concentration = ['Neg', '1 aM', '10 aM', '100 aM','1 fM','10 fM','100 fM', '1 pM', '10 pM','100 pM', '1 nM', '10 nM']
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

# 이미지 불러오기
def Docrop(img):
    # 이미지 자르기
    img_pillow= img.convert('RGB') 
    open_cv_image = np.array(img_pillow)
    img_raw = open_cv_image[:, :, ::-1].copy()  
    # 회전
    load_img = imutils.rotate(img_raw, 0)
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
    
    return cropImg

def Drawarea(img):
    img_pillow= img.convert('RGB') 
    open_cv_image = np.array(img_pillow)
    img_raw = open_cv_image[:, :, ::-1].copy()  
    # 회전
    load_img = imutils.rotate(img_raw, 0)
    # 원본 이미지가 image shape : (3024, 4032, 3)
    image = imutils.resize(load_img, height=1400)
    oriImage = image.copy()

    # 자르기를 원하는 위치
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # 타겟 반지름
    rad = 90
    
    drawrect(oriImage,(cX-rad, cY-rad), (cX+rad, cY+rad), (50,0,255), 6, 'dotted')
    cv2.putText(oriImage, 'Target Well', (cX-rad-60, cY-rad-45), cv2.FONT_HERSHEY_COMPLEX, 2 ,(50,0,255),2,cv2.LINE_AA)
    return oriImage
def Dodetect(cropImg):
    # 이미지 자르기
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

    # 결과 출력
    return predicted_result
## 사이트 설정
options = ['SARS-CoV-2','SARS-CoV-2 variant']
selected_option = st.select_slider("Choose a type", options=options)
st.write("The selected type :mag: is",selected_option)
st.markdown("""---""")
st.header('Before guide RNA')
uploaded_file_before = st.file_uploader("Please upload your sample image before guide RNA.", type=['jpeg', 'png', 'jpg', 'webp'])
if uploaded_file_before is not None:
    with st.spinner('Now processing.....'):
        image_before = Image.open(uploaded_file_before)
        image_before_drawed = Drawarea(image_before)
        try:        
            cropped_before = Docrop(image_before)
        except:
            st.write("There is problem with processing...\nplease upload another image!")
        else:
            st.image([Image.fromarray(image_before_drawed[:, :, ::-1].copy()),Image.fromarray(cropped_before[:, :, ::-1].copy())], 
                    caption=['Uploaded sample image','Target well'], use_column_width ='auto')
            label_before = Dodetect(cropped_before)[0]
            st.write("")
            
    st.success('Done!')

st.markdown("""---""")
st.header('After guide RNA')
uploaded_file_after = st.file_uploader("Please upload your sample image after guide RNA.", type=['jpeg', 'png', 'jpg', 'webp'])
if uploaded_file_after is not None:
    with st.spinner('Now processing.....'):
        image_after = Image.open(uploaded_file_after)
        image_after_drawed = Drawarea(image_after)
        try:
            cropped_after = Docrop(image_after)
        except:
            st.write("There is problem with processing...\nplease upload another image!")
        else:
            st.image([Image.fromarray(image_after_drawed[:, :, ::-1].copy()),Image.fromarray(cropped_after[:, :, ::-1].copy())], 
                    caption=['Uploaded sample image','Target well'], use_column_width ='auto')
            label_after = Dodetect(cropped_after)[0]
    st.success('Done!')

st.markdown("""---""")
st.title('Detection Result')
st.write('Please do analyze!')
if st.button('Analyze'):
    with st.spinner('Now processing.....'):
        try:
            st.markdown('<p style="font-family:sans-serif; color:Black;font-size: 25px;">Sample Type:</p>', unsafe_allow_html=True)
            if label_before - label_after >= 3:
                if selected_option == 'SARS-CoV-2 variant':
                    st.markdown('<p style="font-family:sans-serif; color:Black;font-size: 32px;"><strong>SARS-CoV-2 variant</strong></p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="font-family:sans-serif; color:Black;font-size: 32px;"><strong>SARS-CoV-2</strong></p>', unsafe_allow_html=True)
                st.markdown('<p style="font-family:sans-serif; color:Black;font-size: 25x;">Result:</p>', unsafe_allow_html=True)
                st.markdown('<p style="font-family:sans-serif; color:Red; font-size: 32px;"><strong>POSITIVE</strong></p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-family:sans-serif; color:Black;font-size: 18px;"><strong>DNA Concentration $\approx$ {list_concentration[label_before]}</strong></p>', unsafe_allow_html=True)

            else:
                if selected_option == 'SARS-CoV-2 variant':
                   st.markdown('<p style="font-family:sans-serif; color:Black;font-size: 32px;"><strong>SARS-CoV-2 variant</strong></p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p style="font-family:sans-serif; color:Black;font-size: 32px;"><strong>SARS-CoV-2</strong></p>', unsafe_allow_html=True)
                st.markdown('<p style="font-family:sans-serif; color:Black;font-size: 25px;">Result:</p>', unsafe_allow_html=True)
                st.markdown('<p style="font-family:sans-serif; color:Blue; font-size: 32px;"><strong>NEGATIVE</strong></p>', unsafe_allow_html=True)
                st.markdown(f'<p style="font-family:sans-serif; color:Black;font-size: 18px;"><strong>DNA Concentration $\approx$ {list_concentration[label_before]}</strong></p>', unsafe_allow_html=True)

        except:
            st.markdown('Please __re-upload__ images')
    st.success('Done!')

