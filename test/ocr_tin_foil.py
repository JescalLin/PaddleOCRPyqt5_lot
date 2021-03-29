from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
import imutils

cls_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_mobile_v2.0_cls_infer'
det_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_det_infer' 
rec_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_rec_infer'
ocr = PaddleOCR(cls_model_dir=cls_model_dirs, det_model_dir=det_model_dirs ,rec_model_dir = rec_model_dirs)
mult_img = True

if  mult_img == True:
    path = './image/tin_foil/'
    allFileList = os.listdir(path)
    filename=[]
    for file in allFileList:
        if 'JPG' in file or 'jpg' in file:
            filename.append(file)
        if 'PNG' in file or 'png' in file:
            filename.append(file)
        if 'BMP' in file or 'bmp' in file:
            filename.append(file)
    for k in range(len(filename)):
        img = cv2.imread(path+filename[k])

        # 裁切區域的 x 與 y 座標（左上角）
        x = 0
        y = 500
        # 裁切區域的長度與寬度
        w = 540
        h = 220
        img = img[y:y+h, x:x+w]
        img  = imutils.resize(img,width=300)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        L3 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 3, 0.9), (w, h),borderValue=(128,128,128))
        R3 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, -3, 0.9), (w, h),borderValue=(128,128,128))
        L6 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 6, 0.9), (w, h),borderValue=(128,128,128))
        R6 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, -6, 0.9), (w, h),borderValue=(128,128,128))

        img=cv2.vconcat([R6,R3,img,L3,L6])

        result = ocr.ocr(img)
        for i in range(len(result)):
            pts = np.array([result[i][0]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(0,255,0))
            if (result[i][1][0].isalnum()==True and len(result[i][1][0])>=10 and 'EXP' not in result[i][1][0]):
                cv2.putText(img, result[i][1][0][-10:], (int(result[i][0][3][0]), int(result[i][0][3][1])-40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)



        print(result)
        cv2.imwrite("./image/output/"+filename[k],img)


else:
    name = 'CandyMagic-2' + '.bmp'
    img = cv2.imread("./image/tin_foil/"+name)

    # 裁切區域的 x 與 y 座標（左上角）
    x = 0
    y = 500
    # 裁切區域的長度與寬度
    w = 540
    h = 220
    img = img[y:y+h, x:x+w]
    img  = imutils.resize(img,width=300)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    L3 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 3, 0.9), (w, h),borderValue=(128,128,128))
    R3 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, -3, 0.9), (w, h),borderValue=(128,128,128))
    L6 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 6, 0.9), (w, h),borderValue=(128,128,128))
    R6 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, -6, 0.9), (w, h),borderValue=(128,128,128))

    img=cv2.vconcat([R6,R3,img,L3,L6])


    result = ocr.ocr(img)
    for i in range(len(result)):
        pts = np.array([result[i][0]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(0,255,0))
        if (result[i][1][0].isalnum()==True and len(result[i][1][0])>=10 and 'EXP' not in result[i][1][0]):
            cv2.putText(img, result[i][1][0][-10:], (int(result[i][0][3][0]), int(result[i][0][3][1])-40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    print(result)

    cv2.imshow("img",img)
    cv2.imwrite("./image/output/"+name,img)
    cv2.waitKey()