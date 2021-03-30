from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from collections import Counter
import imutils
from PIL import ImageFont, ImageDraw, Image
fontPath = "C:\\WINDOWS\\Fonts\\kaiu.TTF"
font = ImageFont.truetype(fontPath, 20)

cls_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_mobile_v2.0_cls_infer'
det_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_det_infer' 
rec_model_dirs = r'D:\PaddleOCR\inference\ch_ppocr_server_v2.0_rec_infer'
# ocr = PaddleOCR(cls_model_dir=cls_model_dirs, det_model_dir=det_model_dirs ,rec_model_dir = rec_model_dirs,det_db_thresh=0.1,det_db_box_thresh=0.1,det_db_unclip_ratio=1.6)
ocr = PaddleOCR(
    cls_model_dir = cls_model_dirs, 
    det_model_dir = det_model_dirs,
    rec_model_dir = rec_model_dirs,
    )

# 選擇第二隻攝影機
cap = cv2.VideoCapture(0)

while(True):
  # 從攝影機擷取一張影像
  ret, frame = cap.read()

  result = ocr.ocr(frame)
  for i in range(len(result)):
      pts = np.array([result[i][0]], np.int32)
      pts = pts.reshape((-1,1,2))
      cv2.polylines(frame,[pts],True,(0,255,0))
      #cv2.putText(frame, result[i][1][0][-10:], (int(result[i][0][3][0]), int(result[i][0][3][1])), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
      
      text = result[i][1][0]
      imgPil = Image.fromarray(frame)
      draw = ImageDraw.Draw(imgPil)
      draw.text((int(result[i][0][3][0]), int(result[i][0][3][1])), text, font=font, fill=(0, 0, 255))
      frame = np.array(imgPil)

  # 顯示圖片
  cv2.imshow('frame', frame)

  # 若按下 q 鍵則離開迴圈
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()