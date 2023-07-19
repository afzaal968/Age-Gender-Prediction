import cv2
#The code imports the OpenCV library, which is used for computer vision tasks
#we used openCV in this project (open source computer vision)

def faceBox(faceNet,frame):
    #faceNet is face detection model & frame takes input image or video frame

    frameHeight=frame.shape[0] #retrieves height of frame
    frameWidth=frame.shape[1] #retrieves width of frame

    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False) #deep neural network
    #generates blob(binary representationn of image using pixels) from frame,
    #which performs preprocessing and prepares the image for input to the neural network.

    faceNet.setInput(blob) #Sets the blob as the input to the faceNet model.
    detection=faceNet.forward() #to detect faces in the frame
    bboxs=[] #draws a rectangular box on the frame
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]

        if confidence>0.7:
        #if visibility of face > 70%, it calculates the bounding box coordinates based on the frame dimensions
        #and the detected face's relative coordinates.

            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2]) #storing the coordinates of frame in bboxs
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs

#setting up path to the required model file
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"
#These variables store the paths to the model configuration files (*.pbtxt)
#and the corresponding model weights files (*.caffemodel). 
#These files are necessary for face detection, age estimation, and gender prediction.

#using deep learning model we read the face, age, gender using below syntax 
faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

#prediction list to process the image
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
#The "model mean value" refers to the pixel-wise mean values used to preprocess the input image
#or face before passing it through the neural network model.
#It is a normalization technique commonly used in computer vision tasks.

ageList = ['(0-2)', '(4-6)', '(8-12)', '(16-20)', '(19-22)', '(33-38)', '(48-53)', '(70-80)']
genderList = ['Male', 'Female']

video=cv2.VideoCapture(0)
#captures video realtime aswell as loaded.
#0 defines index of our webcam, we can play videos by changing the index and provide path of the video.

padding=20
#processing video frames

while True:
    ret,frame=video.read()
    #Reads each frame from the video capture using video.read(). The resulting frame is stored in frame,
    # and the return value (ret) indicates whether the frame was successfully read.

    frame,bboxs=faceBox(faceNet,frame)
    #Calls the faceBox function to detect faces and obtain the bounding boxes for each face in the frame. 
    #The modified frame and the list of bounding boxes are returned.
    
    for bbox in bboxs:

        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        #Extracts the face region from the frame based on the bounding box coordinates.
        #Preprocesses the face image by generating a blob using cv2.dnn.blobFromImage.

        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]
        #Sets the blob as the input to the gender prediction model and performs forward pass inference to obtain gender predictions.

        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]
        #Sets the blob as the input to the age estimation model and performs forward pass inference to obtain age predictions.

        label="{},{}".format(gender,age)
        #Combines the gender and age predictions into a label.

        cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0),-1) 
        #Draws a rectangle around the face region on the frame.

        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
        #Displays the label on the frame.

    cv2.imshow("Age-Gender",frame)
    #Shows the frame with annotations in a window named "Age-Gender" using cv2.imshow.

    k=cv2.waitKey(1)
    if k==ord('q'):
        break
    #Waits for a key press using cv2.waitKey(1). If the key 'q' is pressed, the loop is exited.

video.release()
cv2.destroyAllWindows()
#After the loop ends, the code releases the video capture resources using video.release() 
#and closes all the windows created using cv2.destroyAllWindows().

#The above code captures video frames, detects faces, and predicts the gender and age for each detected face, annotating the frames accordingly.