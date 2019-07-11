import cv2
import math
import numpy as np

def medianBlur(img, kernel, padding_way):
    [row_i, col_i] = np.shape(img)
    [row_k, col_k] = np.shape(kernel)
    a = math.floor(row_k/2)
    b = math.floor(col_k/2)

    # outline extension
    if padding_way == 'ZERO':
        img_ex = np.zeros((row_i+2*a-1, col_i+2*b-1),int)
        img_ex[a-1:-a, b-1:-b] = img

    if padding_way == 'REPLICA':
        img_ex = img
        for i in range(a):
            img_ex = np.vstack((img[0, :], img_ex))
            img_ex = np.vstack((img_ex, img[-1, :]))

        for j in range(b):
            img_ex = np.column_stack((img_ex[:, 0], img_ex))
            img_ex = np.column_stack((img_ex, img_ex[:, -1]))


    # kernel convolution
    img_kc = np.zeros((row_i, col_i), int)
    for i in range(row_i):
        for j in range(col_i):
            print([i,j])
            print([i-a,i+a-1])
            print([j-b,j+b-1])
            img_kc[i, j] = round(np.median(img_ex[i-a+a:i+a-1+a, j-b+b:j+b-1+b]))

    return img_kc

img_gray = cv2.imread('/Users/weishao/PycharmProjects/cv/pic.jpg', 0)
cv2.imshow('img_gray', img_gray)
print(img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = [[0,1,2,3],[1,2,3,4],[2,3,4,5]]
img_blur = np.int8(medianBlur(img_gray,kernel,'REPLICA'))
cv2.imshow('img_blur', img_blur)

print(img_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
