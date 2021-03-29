from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import re

cls_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_mobile_v2.0_cls_infer'
det_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_det_infer' 
rec_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_rec_infer'

ocr = PaddleOCR(
    cls_model_dir = cls_model_dirs, 
    det_model_dir = det_model_dirs,
    rec_model_dir = rec_model_dirs,
    det_db_thresh=0.3,
    det_db_box_thresh=0.1,
    det_db_unclip_ratio=1.6,
    drop_score=0.0,
    use_angle_cls=True,
    use_space_char=False,
    )

name = 'Miche Blooming-2' + '.bmp'
img = cv2.imread("./image/tin_foil/"+name)

# showCrosshair = False
# fromCenter = False
# r = cv2.selectROI(img, fromCenter, showCrosshair)
# img = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
# img = cv2.copyMakeBorder(img,20,20,20,20,cv2.BORDER_REPLICATE)

# 裁切區域的 x 與 y 座標（左上角）
x = 0
y = 500
# 裁切區域的長度與寬度
w = 540
h = 220
img = img[y:y+h, x:x+w]
img2 = img.copy()

kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
threshed = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,3)
dilate = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel1)
# cv2.imshow("dilate",dilate)
# Find contours and draw rectangle
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

k=0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    if(w>200):
        k = k +1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,100,0), 2)
        cropped_img = img2[y:(y+h), x:(w+x)]
        cropped_img = cv2.copyMakeBorder(cropped_img,250,250,100,100,cv2.BORDER_CONSTANT,value=(128,128,128))
        result = ocr.ocr(cropped_img)
        for i in range(len(result)):
            # if (result[i][1][0].isalnum()==True and len(result[i][1][0])>=10 and 'EXP' not in result[i][1][0]):
            pts = np.array([result[i][0]], np.int32)
            pts = pts.reshape((-1,1,2))
            cv2.polylines(cropped_img,[pts],True,(0,255,0))
            cv2.putText(cropped_img, result[i][1][0][-10:], (int(result[i][0][3][0]), int(result[i][0][3][1])+40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
            print(result[i][1][0][-10:])
            cv2.imshow("cropped_img"+str(k),cropped_img)
            cv2.imwrite("./image/output/tin_foil/"+name,cropped_img)

                


cv2.imshow("img",img)

cv2.waitKey()




# 左上
# result[0][0][0]
# 右上 
# result[0][0][1]
# 右下
# result[0][0][2]
# 左下
# result[0][0][3]

# 字
# print(result[0][1][0])