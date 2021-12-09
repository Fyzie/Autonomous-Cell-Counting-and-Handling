import os
import cv2
import imutils
from pushbullet import Pushbullet
import time
import copy
from tkinter import *
# from robotic import *

# init()  # initialize robotic arm position

# register email into pushbullet browsert
# gmail = curiosity.iium@gmail.com
# psk = procuriosity21

pyr = Tk()


def replace(title, message, color):
    k = 0
    pyr.geometry('300x60')
    pyr.title(title)
    pyr.configure(background=color)
    lbl = Label(pyr, text=message, bg=color, font=('Aerial', 17))
    lbl.pack()
    while True:
        pyr.update_idletasks()
        pyr.update()
        # control()
        pyr.destroy()
        break


# get access code
pb = Pushbullet("o.OIb3JHHXpqOwXJIaW7TLaM0Ok1qZlViH")
print(pb.devices)  # print all registered devices

# load the model
net = cv2.dnn_DetectionModel("yolov4-tiny-custom.cfg", "yolov4-tiny-custom_best_aug.weights")
net.setInputSize(416, 416)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

# always check if new image exists

while True:
    path = "new/"
    # path to USB port: PC
    # path = "H:/"
    # path to USB port: Raspberry Pi
    # path = "/media/pi/4201-21A6/"
    if os.path.exists(path):
        dirs = os.listdir(path)
        if len(dirs) != 0:
            for item in dirs:
                full_path = os.path.join(path, item)
                if os.path.isfile(full_path):
                    select_img = full_path
                    chosen = os.path.basename(select_img)
                    frame = cv2.imread(full_path)
                    frame = imutils.resize(frame, width=1000)
                    h, w, _ = frame.shape
                    print(frame.shape)
                    frame_size = h * w

                    # read available class name in the obj.names file
                    with open('obj.names', 'r') as f:
                        names = f.read().split('\n')

                    # label all detected object using trained YOLO model
                    confluency = 0
                    cell = 0
                    classes, confidence, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
                    for classId, confidence, box in zip(classes.flatten(), confidence.flatten(), boxes):
                        label = '%.2f' % confidence  # get confidence score of each object detection
                        label = '%s:%s' % (
                            names[classId], label)  # concatenate class name and respective confidence score
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4, 1)
                        left, top, width, height = box
                        top = max(top, labelSize[1])
                        cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top),
                                      (255, 255, 255), cv2.FILLED)
                        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
                        c_frame = copy.deepcopy(frame)

                        cell += 1
                        area = width * height
                        confluency += area
                        percentage = confluency / frame_size * 100
                        percentage = round(percentage, 2)
                        details = 'NOC: {}  POC: {:.2f}%'.format(cell, percentage)
                        indexSize, _ = cv2.getTextSize(details, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, 1)
                        text_pos = (int((w / 2) - (indexSize[0] / 2)), 50)
                        cv2.putText(c_frame, details, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0),
                                    3)

                        cv2.imshow('Result: ' + chosen, c_frame)
                        if cv2.waitKey(1) == 27:
                            break  # esc to quit
                        time.sleep(0.1)

                    print("Image: ", chosen)
                    num_cell = len(classes.flatten())
                    print('Number of cells: ', num_cell)
                    print('Confluency: ', percentage, '%')
                    chosen_name = chosen.split(".")
                    file_name = chosen_name[0] + " N" + str(num_cell) + " P" + str(percentage) + ".jpg"
                    cv2.imwrite("result3/" + file_name, c_frame)

                    # notify the users through Pushbullet network
                    # for all registered devices
                    # push1 = pb.push_note("Alert!!",
                    #                      "Image: " + chosen + "\n" +
                    #                      "Number of cells: " + str(num_cell) + "\n" +
                    #                      "Confluency: " + str(percentage) + "%")
                    #
                    # with open("result3/" + file_name, "rb") as pic:
                    #     file_data = pb.upload_file(pic, chosen)
                    # push2 = pb.push_file(**file_data)

                    # robotic arm: replace the dish if limit reached
                    # the limit can be adjusted
                    if percentage > 4:
                        replace("Alert!", "Cell confluency reached limit\nReplacing...", "#fff")

                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    os.remove(full_path)
    time.sleep(1)
