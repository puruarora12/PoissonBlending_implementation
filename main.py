import cv2
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import linalg
import matplotlib.pyplot as plt

from align_target import align_target


def poisson_blend(source_image, target_image, target_mask):
    #source_image: image to be cloned
    #target_image: image to be cloned into
    #target_mask: mask of the target image
    h,w,c= target_image.shape
    print(target_image.shape)

    patch_pixels = np.argwhere(target_mask)
    pixels = len(patch_pixels)
    print(patch_pixels.shape)

    A= np.zeros((pixels, pixels))
    b = np.zeros((c, pixels))
    
    im2var = np.zeros((h,w), np.int32)

    for i in range(0,pixels):
        x = patch_pixels[i][0]
        y = patch_pixels[i][1]
        im2var[x,y]= i

    for i in range(h):
        for j in range(w):
            if (target_mask[i][j]==0):
                continue
            for k in range(c):
                A[im2var[i , j]  , im2var[i,j]] =4

                if( target_mask[i-1,j]== 1  ): 
                    A[im2var[i, j] , im2var[i-1 , j]] =-1
                    b[k , im2var[i , j]]+= int(source_image[i , j , k ]) - int(source_image[i-1 , j , k])
                else:
                    b[k , im2var[i , j]]+= target_image[i-1,j,k]


                if( target_mask[i,j-1]== 1  ):
                    A[im2var[i, j] , im2var[i , j-1]] =-1
                    b[k , im2var[i , j]]+= int(source_image[i , j , k ]) - int(source_image[i , j-1 , k])
                else:
                    b[k , im2var[i , j]]+= target_image[i-1,j,k] 


                if( target_mask[i+1,j]== 1  ):
                    A[im2var[i, j] , im2var[i+1 , j]] =-1
                    b[k , im2var[i, j]]+= int(source_image[i , j , k ]) - int(source_image[i+1 , j , k])
                else:
                    b[k , im2var[i , j]]+= target_image[i-1,j,k]

                if( target_mask[i,j+1]== 1 ):
                    A[im2var[i, j] , im2var[i , j+1]] =-1
                    b[k , im2var[i , j+1]]+= int(source_image[i , j , k ]) - int(source_image[i , j+1 , k])
                else:
                    b[k , im2var[i , j]]+= target_image[i-1,j,k]

    
    A= csr_matrix(A)
    b= csr_matrix(b)

    R = sparse.linalg.spsolve(A, b[2].T)
    G = sparse.linalg.spsolve(A, b[1].T)
    B = sparse.linalg.spsolve(A, b[0].T)
    

    red = np.linalg.norm(A * R - b[2])
    red2 = np.format_float_positional(red, trim='-')
    print(f"Red Error: {red} or {red2}")
    green = np.linalg.norm(A * G - b[1])
    green2 = np.format_float_positional(green, trim='-')
    print(f"Green Error: {green} or {green2}")
    blue = np.linalg.norm(A * B - b[0])
    blue2 = np.format_float_positional(blue, trim='-')
    print(f"Blue Error: {blue} or {blue2}")
    
    R= np.clip(R , 0 ,255)
    G= np.clip(G , 0 ,255)
    B= np.clip(B , 0 ,255)

    for c in range(pixels):
        i = patch_pixels[c][0]
        j = patch_pixels[c][1]
        target_image[i,j,2] = abs((R[c]))
        target_image[i,j,1] = abs((G[c]))
        target_image[i,j,0] = abs((B[c]))

    cv2.imshow("Poisson blend", target_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



                


        

if __name__ == '__main__':
    #read source and target images
    source_path = 'source1.jpg'
    target_path = 'target.jpg'
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)
    moon_image = cv2.imread("source2.jpg")
    #align target image
    im_source, mask = align_target(source_image, target_image)

    ##poisson blend
    blended_image = poisson_blend(im_source, target_image, mask)