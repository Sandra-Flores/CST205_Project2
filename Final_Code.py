import cv2

# load cascade file to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Load overlay image
img_emoji = cv2.imread('happy2.png',-1)
#creates the mask for the emoji
#aka picture withour background
orig_mask = img_emoji[:,:,3]
#creates the inverted mask for the emoji
#aka the background on the picture
orig_mask_inv = cv2.bitwise_not(orig_mask)
# Convert mustache image to BGR
# and save the original image size
img_emoji = img_emoji[:,:,0:3]
orig_emoji_height, orig_emoji_width = img_emoji.shape[:2]
# collect video input from first webcam on system
video_capture = cv2.VideoCapture(0)
while True:
    # Capture video feed
    ret, frame = video_capture.read()
    # Create greyscale image from the video feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in input video stream
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
   # Iterate over each face found
    for (x, y, w, h) in faces:
        #face = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #manipulate the width and height of the picture to
        #fit that of the rectangle on the face
        orig_emoji_width = w
        orig_emoji_height = h
        #resize the emoji with the resize function
        #parameters (emoji, width, height, cv2 function)
        emoji = cv2.resize(img_emoji, (orig_emoji_width,orig_emoji_height),interpolation = cv2.INTER_AREA)
        mask = cv2.resize(orig_mask, (orig_emoji_width,orig_emoji_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(orig_mask,(orig_emoji_width,orig_emoji_height), interpolation = cv2.INTER_AREA)
        roi = roi_color[0:h,0:w]
         # roi_bg contains the original image only where the emoji is not
         # in the region that is the size of the emoji (aka blank space)
        roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
        # roi_fg contains the image of the emoji only where the emoji is
        #aka the emoji face
        roi_fg = cv2.bitwise_and(emoji,emoji,mask = mask)
        #join the roi_bg and roi_fg
        dst = cv2.add(roi_bg,roi_fg)
        roi_color[0:h,0:w] = dst

    #diplay the resulting frame
    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
