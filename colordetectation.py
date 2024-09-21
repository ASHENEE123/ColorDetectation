import cv2
import numpy as np
img=cv2.imread(r"D:\collage-image-same-happy-young-260nw-1416089468.webp")
resize=cv2.resize(img,(700,500))
cvt_color=cv2.cvtColor(resize,cv2.COLOR_BGR2HSV)
lower=np.array([0,0,0])
higher=np.array([225,255,225])
mask=cv2.inRange(cvt_color,lower,higher)
cv2.imshow("original img",img)
cv2.imshow("color detected image",mask)
cv2.waitKey(0)
cv2.destroyAllWindows()