from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from collections import Counter
import imutils

cls_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_mobile_v2.0_cls_infer'
det_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_det_infer' 
rec_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_rec_infer'
# ocr = PaddleOCR(cls_model_dir=cls_model_dirs, det_model_dir=det_model_dirs ,rec_model_dir = rec_model_dirs,det_db_thresh=0.1,det_db_box_thresh=0.1,det_db_unclip_ratio=1.6)
ocr = PaddleOCR(
    cls_model_dir = cls_model_dirs, 
    det_model_dir = det_model_dirs,
    rec_model_dir = rec_model_dirs,
    )
            
# CandyMagic        HE2103B004
# LaPeche           FK21030000
# MicheBlooming     132009R001
# PureNatural       R52103B003
# Purity            FS2103N002
# Revia             HV2102P008
# SecretCandyMagic  A72103D000

name = 'MicheBlooming-2' + '.bmp'
img = cv2.imread("./image/tin_foil/"+name)

# showCrosshair = False
# fromCenter = False
# r = cv2.selectROI(img, fromCenter, showCrosshair)
# img = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
# img = cv2.copyMakeBorder(img,100,100,100,100,cv2.BORDER_REPLICATE)
# img = cv2.copyMakeBorder(img,50,50,100,100,cv2.BORDER_CONSTANT,value=(255,255,255))



# 裁切區域的 x 與 y 座標（左上角）
x = 0
y = 445
# 裁切區域的長度與寬度
w = 540
h = 720 -y
img = img[y:y+h, x:x+w]

h, w = img.shape[:2]
center = (w // 2, h // 2)

L3 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 3, 0.9), (w, h),borderValue=(128,128,128))
R3 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, -3, 0.9), (w, h),borderValue=(128,128,128))
L6 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, 6, 0.9), (w, h),borderValue=(128,128,128))
R6 = cv2.warpAffine(img, cv2.getRotationMatrix2D(center, -6, 0.9), (w, h),borderValue=(128,128,128))

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
d1 = cv2.dilate(img,kernel)
d2 = 255-cv2.dilate(255-img,kernel)

d1_L3 = cv2.warpAffine(d1, cv2.getRotationMatrix2D(center, 3, 0.9), (w, h),borderValue=(128,128,128))
d1_R3 = cv2.warpAffine(d1, cv2.getRotationMatrix2D(center, -3, 0.9), (w, h),borderValue=(128,128,128))
d1_L6 = cv2.warpAffine(d1, cv2.getRotationMatrix2D(center, 6, 0.9), (w, h),borderValue=(128,128,128))
d1_R6 = cv2.warpAffine(d1, cv2.getRotationMatrix2D(center, -6, 0.9), (w, h),borderValue=(128,128,128))


d2_L3 = cv2.warpAffine(d2, cv2.getRotationMatrix2D(center, 3, 0.9), (w, h),borderValue=(128,128,128))
d2_R3 = cv2.warpAffine(d2, cv2.getRotationMatrix2D(center, -3, 0.9), (w, h),borderValue=(128,128,128))
d2_L6 = cv2.warpAffine(d2, cv2.getRotationMatrix2D(center, 6, 0.9), (w, h),borderValue=(128,128,128))
d2_R6 = cv2.warpAffine(d2, cv2.getRotationMatrix2D(center, -6, 0.9), (w, h),borderValue=(128,128,128))


line1=cv2.hconcat([img,img,d1,d2])
line2=cv2.hconcat([L3,R3,L6,R6])
line3=cv2.hconcat([d1_L3,d1_R3,d1_L6,d1_R6])
line4=cv2.hconcat([d2_L3,d2_R3,d2_L6,d2_R6])


img=cv2.vconcat([line1,line2,line3,line4])



# img = cv2.copyMakeBorder(img,100,100,20,20,cv2.BORDER_REPLICATE)


result = ocr.ocr(img)
lot = []

for i in range(len(result)):
    pts = np.array([result[i][0]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(0,255,0))
    if (result[i][1][0].isalnum()==True and len(result[i][1][0])>=10 and 'EXP' not in result[i][1][0]):
        cv2.putText(img, result[i][1][0][-10:], (int(result[i][0][3][0]), int(result[i][0][3][1])+40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        lot.append(result[i][1][0][-10:])



cv2.imshow("img",imutils.resize(img,width=1000))


lot_counts = Counter(lot)
top = lot_counts.most_common(3)
print(top)
cv2.waitKey()

cv2.imwrite("./image/output/tin_foil/"+name,img)


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