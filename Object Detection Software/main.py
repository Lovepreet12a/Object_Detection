import cv2 
# import numpy as np
from gui_buttons import Buttons

# initialize buttons
button = Buttons()
button.add_button("person", 20, 20)
button.add_button("cell phone", 20, 100)
button.add_button("remote", 20, 180)
button.add_button("cup", 20, 260)
button.add_button("keyboard", 20, 340)

   


# Opencv DNN
net = cv2.dnn.readNet("dnn_model\yolov4-tiny.weights", "dnn_model\yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size = (320,320), scale= 1/255)


# Loading class list
classes = []
with open("dnn_model\classes.txt") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# print(classes)

# opening the web camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Full HD 1920 x 1080

button_person = False

def click_button(event, x,y,flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x,y)
        # polygon = np.array([[(20,20), (150,20), (150,70), (20,70)]])

        # is_inside = cv2.pointPolygonTest(polygon, (x,y), False)
        # if is_inside > 0:
        #     # print("We are clicking inside the button")
        #     if button_person is False:
        #         button_person = True
        #     else:
        #         button_person = False

        #     print("Now button person is: ", button_person)

        
        

# Creating window 
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)



while True:
    # Get frames
    ret, frame = cap.read()


    # Get active buttons list
    active_buttons = button.active_buttons_list()
    # print("Active Buttons: ", active_buttons)

    
    # Object detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x,y,w,h) = bbox
        class_name = classes[class_id]

        if class_name in active_buttons:
            cv2.putText(frame, class_name,(x,y - 5),cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)


    # Display buttons
    button.display_buttons(frame)




    # print("class_ids: ", class_ids)
    # print("score: ", scores)
    # print("bboxes: ", bboxes)

    # Creating buttons
    # cv2.rectangle(frame, (20,20), (150,70), (0,0,200), -1)
    # polygon = np.array([[(20,20), (150,20), (150,70), (20,70)]])
    # cv2.fillPoly(frame, polygon, (0,0,200))
    # cv2.putText(frame, "Person", (21,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)


    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

