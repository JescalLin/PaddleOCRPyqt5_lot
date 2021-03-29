from paddleocr import PaddleOCR
import cv2
import numpy as np
import os

cls_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_mobile_v2.0_cls_infer'
det_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_det_infer' 
rec_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_rec_infer'
ocr = PaddleOCR(cls_model_dir=cls_model_dirs, det_model_dir=det_model_dirs ,rec_model_dir = rec_model_dirs)
mult_img = True

if  mult_img == True:
    path = './image/DMK33UX287+HS2514J/'
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
        result = ocr.ocr(img)
        for i in range(len(result)):
            pts = np.array([result[i][0]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(img,[pts],True,(0,255,0))
            cv2.putText(img, result[i][1][0], (int(result[i][0][3][0]), int(result[i][0][3][1])+40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        print(result)
        cv2.imwrite("./image/output/"+filename[k],img)


else:
    name = 'P070000762-3' + '.bmp'
    img = cv2.imread("./image/DMK33UX287+HS2514J/"+name)

    
    result = ocr.ocr(img)
    for i in range(len(result)):
        pts = np.array([result[i][0]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(0,255,0))
        cv2.putText(img, result[i][1][0], (int(result[i][0][3][0]), int(result[i][0][3][1])+40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    print(result)

    cv2.imshow("img",img)
    cv2.imwrite("./image/output/"+name,img)
    cv2.waitKey()