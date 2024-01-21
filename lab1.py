import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

fp = "C:/Users/admin/Desktop/ITMO/Image_Processing/"
img = cv.imread(fp+ "buscemi.png", cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
#img[100,100] = [255,255,255] 
cv.imshow("Steve Buscemi",img)
N = 256
histr = cv.calcHist([img],None,None,[N],[0,N])

plt.figure('hist')
plt.plot(histr)
plt.xlim([0,N])

equ = cv.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
hist_equ = cv.calcHist([equ],None,None,[N],[0,N])

plt.figure('hist equ')
plt.plot(hist_equ)
plt.xlim([0,N])


cv.imwrite(fp+'res.png',res)
cv.imshow("Steve Buscemi NEW",res)

cv.waitKey(0)
cv.destroyAllWindows()
plt.show()