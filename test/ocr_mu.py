from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

cls_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_mobile_v2.0_cls_infer'
det_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_det_infer' 
rec_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_rec_infer'

fontPath = "C:\\WINDOWS\\Fonts\\kaiu.TTF"
font = ImageFont.truetype(fontPath, 16)
ocr = PaddleOCR(cls_model_dir=cls_model_dirs, det_model_dir=det_model_dirs ,rec_model_dir = rec_model_dirs,enable_mkldnn=True)
for k in range(3):
    img = cv2.imread("./Image/66/R_"+str("%02d" % (k+1))+".bmp")

    # ocr = PaddleOCR(cls_model_dir=cls_model_dirs, det_model_dir=det_model_dirs ,rec_model_dir = rec_model_dirs)
    result = ocr.ocr(img)
    for i in range(len(result)):
        pts = np.array([result[i][0]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(0,255,0))

        x,y,w,h = cv2.boundingRect(pts)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.putText(img, result[i][1][0], (int(result[i][0][3][0]), int(result[i][0][3][1])+40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        imgPil = Image.fromarray(img)
        draw = ImageDraw.Draw(imgPil)
        draw.text((int(result[i][0][3][0]), int(result[i][0][3][1])), result[i][1][0], font=font, fill=(0, 0, 255))
        img = np.array(imgPil)
        cv2.imshow("img"+str(k),img)

    # print(result)


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