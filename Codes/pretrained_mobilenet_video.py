import numpy as  np
import cv2

# Load the image to detect, get width, height 
# Resize to match input size, convert to blob to pass into model
vidStream = cv2.VideoCapture('20161227_104808.mp4')
# vidStream = cv2.VideoCapture(0)

while True:
    ret, current_frame = vidStream.read()

    # Mirror the input video
    current_frame = cv2.flip(current_frame, 1)

    img_height = current_frame.shape[0]
    img_width = current_frame.shape[1]
    resized_img_to_detect = cv2.resize(current_frame,(300,300))
    img_blob = cv2.dnn.blobFromImage(resized_img_to_detect,0.007843,(300,300),127.5)
    # print(img_blob)
    # Recommended scale factor is 0.007843, width,height of blob is 300,300, mean of 255 is 127.5, 

    # set of 21 class labels in alphabetical order (background + rest of 20 classes)
    class_labels = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]

    # Loading pretrained model from prototext and caffemodel files
    # Input preprocessed blob into model and pass through the model
    # Obtain the detection predictions by the model using forward() method
    mobilenetssd = cv2.dnn.readNetFromCaffe('mobilenetssd.prototext', 'mobilenetssd.caffemodel')
    mobilenetssd.setInput(img_blob)
    obj_detections = mobilenetssd.forward()
    # Returned obj_detections[0, 0, index, 1] , 1 => will have the prediction class index
    # 2 => will have confidence, 3 to 7 => will have the bounding box co-ordinates
    no_of_detections = obj_detections.shape[2]

    # Loop over the detections
    for index in np.arange(0, no_of_detections):
        prediction_confidence = obj_detections[0, 0, index, 2]
        # Take only predictions with confidence more than 20%
        if prediction_confidence > 0.20:
            
            # Get the predicted label
            predicted_class_index = int(obj_detections[0, 0, index, 1])
            predicted_class_label = class_labels[predicted_class_index]
            
            # Obtain the bounding box co-oridnates for actual image from resized image size
            bounding_box = obj_detections[0, 0, index, 3:7] * np.array([img_width, img_height, img_width, img_height])
            (start_x_pt, start_y_pt, end_x_pt, end_y_pt) = bounding_box.astype("int")
            
            # Print the prediction in console
            predicted_class_label = "{}: {:.2f}%".format(class_labels[predicted_class_index], prediction_confidence * 100)
            print("predicted object {}: {}".format(index+1, predicted_class_label))
            
            # Draw rectangle and text in the image
            cv2.rectangle(current_frame, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), (0,255,0), 2)
            cv2.putText(current_frame, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


    cv2.imshow("Detection Output", current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidStream.release()
cv2.destroyAllWindows()
