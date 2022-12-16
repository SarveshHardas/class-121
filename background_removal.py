# import cv2 to capture videofeed
import cv2

import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the mountain image
mountain = cv2.imread('mt_everest.jpg')

# resizing the mountain image as 640 X 480
mountain.resize(640,480)

while True:

    # read a frame from the attached camera
    status , frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = cv2.flip(frame , 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([0, 120, 50])
        upper_bound = np.array([10, 255, 255])

        lower_bound = np.array([170, 120, 70])
        upper_bound  = np.array([180, 255, 255])

        # thresholding image
        mask_1 = cv2.inRange(frame_rgb, lower_bound, upper_bound)
        mask_2 = cv2.inRange(frame_rgb, lower_bound, upper_bound)
        mask_1 = mask_1 + mask_2

        # inverting the mask
        mask_1 = cv2.morphologyEx(mask_1,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        mask_1 = cv2.morphologyEx(mask_1,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))

        # bitwise and operation to extract foreground / person
        mask_2 = cv2.bitwise_not(mask_1)
        mask1 = cv2.bitwise_and(frame,frame,mask = mask_2)
        mask2 = cv2.bitwise_and(frame,frame,mask = mask_1)

        # final image
        final_output = cv2.addWeighted(mask1,1,mask2,1,0)
        output_file.write(final_output)

        # show it
        cv2.imshow('frame' , final_output)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code  ==  32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
