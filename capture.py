# 1) ensure you have python installed on your machine
# 2) Follow the following guide to install open cv python on your windows machine
# https://pypi.org/project/opencv-python/ make sure to use "pip install opencv-contrib-python"
# else cv2.imshow() will cause errors.

# Credit: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

# import the open cv library
import cv2

# Use the computer's webcam as the video capture source
cap = cv2.VideoCapture(0)

# loop showing the video feel forever while the camera is operational
while cap.isOpened():

    ret, frame = cap.read()

    # if you've captured a new frame show it
    if ret == True:

        cv2.imshow('frame', frame)
        # frame is cv::Mat in python so convert it to a PIL imaage

        # dont quit unless the user presses 'q' on a 64 bit machine
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# clean up after the program is done
cap.release()
cv2.destroyAllWindows()
