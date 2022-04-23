# Jpeg encoding

import cv2
import numpy as np
import math

# import zigzag functions
from zigzag import *

# defining block size
block_size = 8

# reading image in grayscale
img = cv2.imread('image.jpeg', 0)
cv2.imshow('input image', img)

# get size of the image
[h , w] = img.shape

# convert h and w to float to get the right number
h = np.float32(h)
w = np.float32(w)

# to cover the whole image the number of blocks should be ceiling of the division of image size by block size
# at the end convert it to int

# number of blocks in height
nbh = math.ceil(h/block_size)
nbh = np.int32(nbh)

# number of blocks in width
nbw = math.ceil(w/block_size)##### your code #####
nbw = np.int32(nbw)

# height of padded image
H =  block_size * nbh

# width of padded image
W =  block_size * nbw##### your code #####

# create a numpy zero matrix with size of H,W
padded_img = np.zeros((H,W))

# copy the values of img  into padded_img[0:h,0:w]
for i in range(int(h)):
        for j in range(int(w)):
                pixel = img[i,j]
                padded_img[i,j] = pixel

cv2.imshow('input padded image', np.uint8(padded_img))

# start encoding:
# divide image into block size by block size (here: 8-by-8) blocks
# To each block apply 2D discrete cosine transform
# reorder DCT coefficients in zig-zag order
# reshaped it back to block size by block size (here: 8-by-8)


# iterate over blocks
for i in range(nbh):
    
        # Compute start row index of the block
        row_ind_1 = i*block_size
        
        # Compute end row index of the block
        row_ind_2 = row_ind_1+block_size
        
        for j in range(nbw):
            
            # Compute start column index of the block
            col_ind_1 = j*block_size
            
            # Compute end column index of the block
            col_ind_2 = col_ind_1+block_size
            
            # select the current block we want to process using calculated indices
            block = padded_img[ row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2 ]
            
            # apply 2D discrete cosine transform to the selected block
            # use opencv dct function
            DCT = cv2.dct(block)
            
            # reorder DCT coefficients in zig zag order by calling zigzag function
            # it will give you a one dimentional array
            reordered = zigzag(DCT)
            
            # reshape the reorderd array back to (block size by block size) (here: 8-by-8)
            reshaped= np.reshape(reordered, (block_size, block_size))
            
            # copy reshaped matrix into padded_img on current block corresponding indices
            padded_img[row_ind_1 : row_ind_2 , col_ind_1 : col_ind_2] = reshaped

cv2.imshow('encoded image', np.uint8(padded_img))

# write padded_img into 'encoded.txt' file. You can use np.savetxt function.
np.savetxt('encoded.txt',padded_img)

# write [h, w, block_size] into size.txt. You can use np.savetxt function.
np.savetxt('size.txt',[h, w, block_size])
cv2.waitKey(0)
cv2.destroyAllWindows()




