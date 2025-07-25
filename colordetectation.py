import cv2
import numpy as np
img=cv2.VideoCapture(0)
while img.isOpened:
    _,res=img.read()
    color=cv2.cvtColor(res,cv2.COLOR_BGR2RGB)
    lower=np.array([0,0,0])
    higher=np.array([120,102,120])
    mask=cv2.inRange(color,lower,higher)
    cv2.imshow("original img",mask)
    if cv2.waitKey(1) & 0XFF==ord('q'):
         break
img.release()
cv2.destroyAllWindows()
