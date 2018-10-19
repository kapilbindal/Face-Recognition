import cv2
import numpy as np

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade.xml")

skip = 0
face_data = []
dataset_path = './data/'
file_name = input("Enter the name of the person : ")
while True:
	ret,frame = cam.read()
	if ret == False:
		continue
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if len(faces)==0:
		continue
	faces = sorted(faces,key=lambda f:f[2]*f[3])
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		skip += 1
		if(skip % 2 == 0):
			face_data.append(face_section)
			face_data_np = np.array(face_data)
			face_data_np = face_data_np.reshape((face_data_np.shape[0],-1))
			np.save(dataset_path+file_name+'.npy',face_data_np)
			print(len(face_data))

	cv2.imshow("Frame",frame)
	cv2.imshow("Face Section",face_section)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		print("Data Successfully save at "+dataset_path+file_name+'.npy')
		break

cam.release()
cv2.destroyAllWindows()