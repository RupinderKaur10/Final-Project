import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime


path = r"C:\Users\lenovo\PycharmProjects\pythonProject1\imageattendence"                 # giving the path where database of the images is present
images = []                                          # forming an empty list for the images to be append with their respective paths
className = []                                   # forming an empty list so that the name of the images could be append along with their path
mylist = os.listdir(path)                     # listing all the directories present in the given path
print(mylist)                                       #printing the name of the directories


def markAttendance(name):            #user define to program to mark attendence
    with open('Attendance.csv','r+') as f:  # opening CSV file with read and write
        f = open('Attendance.csv', 'r+')# opening CSV file
        myDataList = f.readlines()        # reading the file line by line and returning first value
        nameList = []                             # empty list to store the names for the attendance
    for line in myDataList:                  # creating a loop to split value from ","
        entry = line.split(',')                  # function for spliting
        nameList.append(entry[0])     #entry of names in the empty name list with the first value of the string name written under the photo of the person who is recognized
    if name not in nameList:             # entering the name in the csv file only if there is only one entry
        now = datetime.now()            # function for date and time
        dtString = now.strftime('%H:%M:%S')   # giving a format for the time to be noted
        f.writelines(f'\n{name},{dtString}')
        #writing all the information all at once in the csv file


def findEncdoing(images):                           # user define function for encoding the images in the real mean time
    encodelist = []                                           # empty list for storing the encoded list
    for img in images:                                     # looping
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceloc = face_recognition.face_locations(img)[0]
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


#loading images with the help of loop so that we don't ahve to call the single images again and again
for cls in mylist:                            # forming the loop so that we don't need to write the same code again and again for different images present in the path
    curimg = cv2.imread(f'{path}/{cls}')          # reading all the files in the directories one by one
    images.append(curimg)          # appending all the images in the empty list we form.
    className.append(os.path.splitext(cls)[0]) # appending all the names of the images in the empty list with only their first name.
print(className)


# calling findencoding function for execution
knownEncodelist = findEncdoing(images)
print('Encoding Complete')


cap = cv2.VideoCapture(0)  # starting the video capturing

#creating a while loop to get each frame one by one
while True:
    success, img = cap.read()# this will provide our image
    imgS = cv2.resize(img,(0,0),None,0.25,0.25) # reducing the size of the real time image to 1/4th, so that we capture in order to increase the performance
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # converting from bgr to rgb
    facesCurrFrame = face_recognition.face_locations(imgS) # to find the location of face in the live cam
    encodesCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)  # changing the image into encoding
    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(knownEncodelist, encodeFace)
        faceDis = face_recognition.face_distance(knownEncodelist, encodeFace)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = className[matchIndex].upper()
            print(name)
            startX, startY, endX, endY = faceLoc
            startX, startY, endX, endY = startX * 4, startY * 4, endX * 4, endY * 4
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255,0), 2)
            cv2.rectangle(img, (startX, endY - 35), (endX, endY), (0, 255,0), cv2.FILLED)
            cv2.putText(img, name, (startX + 6, endY - 6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            markAttendance(name)
        else:
            name1= "Unauthorized"
            startX, startY,endX,endY = faceLoc
            startX, startY,endX,endY = startX*4, startY*4,endX*4,endY*4

            cv2.rectangle(img, (startX,startY), (endX,endY), (0, 0, 255), 2)
            cv2.rectangle(img, (startX, endY - 35), (endX, endY), (0, 0, 255), cv2.FILLED)
            cv2.putText(img,name1 , (startX + 6, endY - 6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)

    print("starting video capturing")
    cv2.imshow("Webcam",img)
    cv2.waitKey(1)
