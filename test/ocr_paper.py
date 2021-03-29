from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

cls_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_mobile_v2.0_cls_infer'
det_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_det_infer' 
rec_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_rec_infer'

name = 'P070001146-150-3' + '.bmp'
img = cv2.imread("./image/paper/"+name)


fontPath = "C:\\WINDOWS\\Fonts\\kaiu.TTF"
font = ImageFont.truetype(fontPath, 16)

ocr = PaddleOCR(cls_model_dir=cls_model_dirs, det_model_dir=det_model_dirs ,rec_model_dir = rec_model_dirs)
result = ocr.ocr(img)
for i in range(len(result)):
    pts = np.array([result[i][0]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(img,[pts],True,(0,255,0))
    cv2.putText(img, result[i][1][0], (int(result[i][0][3][0]), int(result[i][0][3][1])+40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # imgPil = Image.fromarray(img)
    # draw = ImageDraw.Draw(imgPil)
    # draw.text((int(result[i][0][3][0]), int(result[i][0][3][1])), result[i][1][0], font=font, fill=(0, 0, 255))
    # img = np.array(imgPil)

print(result)




# img = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
cv2.imshow("img",img)
cv2.imwrite("./image/output/paper/"+name,img)
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