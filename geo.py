import pygame
import serial
#ser = serial.Serial('COM14', baudrate = 9600, timeout = 1)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import math
from queue import PriorityQueue
import time
import argparse
import cv2
import numpy as np
from yolo import YOLO
import wave
import contextlib
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import speech_recognition as sr
import pyttsx3
import nltk
import warnings
import datetime
import webbrowser
import time
from time import ctime
import os
from os import path
import socket
import wikipedia
import subprocess
import wolframalpha
import json
import requests
import pyrealsense2 as rs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)
r = sr.Recognizer()
browserExe = "chrome.exe"

HOST = '127.0.0.1'  # Symbolic name meaning all available interfaces
PORT = 61915  # Arbitrary non-privileged port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
s.listen(1)
conn, addr = s.accept()

PORT1= 12345
s1=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s1.bind((HOST, PORT1))
s1.listen(1)
conn1, addr = s1.accept()


print('Connected by', addr)
y = 'La connexion est établie'
conn.sendall(y.encode('utf-8'))

c=0
list_max = []
Noms = []
s=0 #the number that helps to know when excuting the voice function what i should excute the main boucle or a boucle inside it

precision=1
len_commande=0
RR=0
DD=0
UU=0
LL=0
RD=0
RU=0
LD=0
LU=0
n=0
xs=1*precision
ys=1*precision
ii=xs
jj=ys
Commande=""

start_time=0

pixels=[]
TempBarrier = []
vitesse = 60 #km per h
grid_width=0.002*3600 #grille en km


WIDTH = 1200*precision
WIDTH2= 450*precision
pix = 375*precision # nbr de grille

pixel_distance = 200/(pix*100)
#WIN = pygame.display.set_mode((WIDTH, WIDTH2))
#pygame.display.set_caption("bienvenidos tu R2D2")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)


list_max = []
Noms = []
s=0 #the number that helps to know when excuting the voice function what i should excute the main boucle or a boucle inside it



'''


ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
	print("loading yolo...")
	yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
	print("loading yolo-tiny-prn...")
	yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
else:
	print("loading yolo-tiny...")
	yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])



# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

#pipeline.start()
yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("starting webcam...")
def distance(frames):
    # Wait for a coherent pair of frames: depth and color
    global c
    depth_frame = frames.get_depth_frame()
    frame = frames.get_color_frame()
    frame = np.asanyarray(frame.get_data())

    frame_dpt = depth_frame.as_depth_frame()
    #frame_dpt = pipeline.wait_for_frames().get_depth_frame().as_depth_frame()
    width, height, inference_time, results = yolo.inference(frame)
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = int(x + (w / 2))
        cy = int(y + (h / 2))
        print((cx,cy))
        distance_to_hand = frame_dpt.get_distance(cx, cy)
    try:
        c=c+1
        return distance_to_hand
    except:
        return "ba9i"
        # draw a bounding box rectangle and label on the image
        #color = (224, 41, 9)
        #cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        #text = "%s (%s)" % (name, round(distance_to_hand,2))
        #cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

while True:
		frames = pipeline.wait_for_frames()
		DISTANCE = distance(frames)
		print(DISTANCE)
		if DISTANCE != "ba9i" and DISTANCE > 0 and DISTANCE <6 :
			print("hani")
			break


def ttdroit(distance):
	global pipeline
	distance=distance-0.5
	marwan="2;RR;"+str(distance)+";"
	print(marwan)
	#ser.write(marwan.encode())
ttdroit(DISTANCE)

arduinoData = ""
while arduinoData == "":
    #global arduinoData
    arduinoData = str(ser.readline().decode('ascii'))
    print(arduinoData)
    #ser.write("0".encode())
print("mok")


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.8:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) == 1:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
def sliman():
    global prototxtPath,weightsPath,faceNet,maskNet
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
	# load the face mask detector model from disk
    maskNet = load_model("mask_detector.model")
sliman()
def run():
	# initialize the video stream
	print("[INFO] starting video stream...")

	# loop over the frames from the video stream
	t = "5"
	#vs = VideoStream(src=0).start()

	while True:
		# grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
		frame = pipeline.wait_for_frames()
		frame = frame.get_color_frame()
		frame = np.asanyarray(frame.get_data())
		frame = imutils.resize(frame, width=400)

		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

		# loop over the detected face locations and their corresponding
		# locations
		for (box, pred) in zip(locs, preds):
			# unpack the bounding box and predictions
				(startX, startY, endX, endY) = box
				(mask, withoutMask) = pred
			# determine the class label and color we'll use to draw
			# the bounding box and text
				if (withoutMask > 0.80):
					label = "No Mask"
					color = (0, 0, 255)
					cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
					cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
					print(withoutMask)
					t = "5"
					conn.sendall(t.encode('utf-8'))
				elif (mask > 0.80):
					print("+", mask)
					label = "Mask"
					color = (0, 255, 0)
					cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
					cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
					print(mask)
					t = "6"
					break
		if t == "6":
			print("good")
			break

		# show the output frames
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF


		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	# do a bit of cleanup
	cv2.destroyAllWindows()
	#vs.stop()
	conn.sendall(t.encode('utf-8'))
	#conn.close()

run()
'''
class geolocalisation:
	def vers(grid):
		global xf,yf
		destination =grid[xf][yf]
		destination.make_start()
		return destination
	def main_zakaria(win, width):
		global Commande,start_time
		global TempBarrier
		ROWS = pix
		grid = geolocalisation.make_grid(ROWS, width)
		start = None
		end = None
		geolocalisation.obstacles(grid)
		end=geolocalisation.FROM(grid)
		start=geolocalisation.vers(grid)
		start_time=time.time()
		for row in grid:
			for spot in row:
				spot.update_neighbors(grid, 0)

		geolocalisation.algorithm(lambda: geolocalisation.draw(pygame.display.set_mode((WIDTH, WIDTH2)), grid, ROWS, width), grid, start, end)
		pygame.quit()

	def reconstruct_path(came_from, current, draw,grid):

		global pix, start_time,pixels,xf,yf,xs,ys,WIDTH,Commande,len_commande
		count=0

		while current in came_from  :

			pixels.append([came_from[current].get_row(),current.get_row(),came_from[current].get_col(),current.get_col()])
			current = came_from[current]
			current.make_path()
			xs=current.get_row()
			ys=current.get_col()
			geolocalisation.instruction1(count)

			count=count+1
		b=time.time()-start_time
		print(b)
		draw()
		time.sleep(10)
		geolocalisation.instructionf()

		b=str(len_commande)+";"+Commande
		#ser.write(b.encode())
		print(b)
	def FROM(grid):
		global xs,ys,ii,jj
		destination = grid[xs][ys]
		ii=xs
		jj=ys

		destination.make_end()
		return destination
	def h(p1, p2, c=0):
		x1, y1 = p1
		x2, y2 = p2
		if c == 0:
			return (abs(x1-x2) + abs(y1-y2)) * pixel_distance
		return math.sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1-y2)) * pixel_distance

	def algorithm(draw, grid, start, end):
		count = 0
		open_set = PriorityQueue()
		open_set.put((0, count, start))
		came_from = {}
		g_score = {spot: float("inf") for row in grid for spot in row}
		g_score[start] = 0
		f_score = {spot: float("inf") for row in grid for spot in row}
		f_score[start] = geolocalisation.h(start.get_pos(), end.get_pos())

		open_set_hash = {start}
		
		while not open_set.empty():
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit()

			current = open_set.get()[2]
			open_set_hash.remove(current)

			if current == end:
				geolocalisation.reconstruct_path(came_from, end, draw,grid) 
				end.make_end()
				return True

			for neighbor in current.neighbors:
				
				xn, yn = neighbor.get_pos()
				xc, yc = current.get_pos()
				
				if xn == xc:
					temp_g_score = g_score[current] + pixel_distance
				elif yn == yc:
					temp_g_score = g_score[current] + pixel_distance
				else:
					temp_g_score = g_score[current] + pixel_distance * math.sqrt(2)

				if temp_g_score < g_score[neighbor]:
					came_from[neighbor] = current
					g_score[neighbor] = temp_g_score
					f_score[neighbor] = temp_g_score + geolocalisation.h(neighbor.get_pos(), end.get_pos())
					if neighbor not in open_set_hash:
						count += 1
						open_set.put((f_score[neighbor], count, neighbor))
						open_set_hash.add(neighbor)
						neighbor.make_open()


			WIDTH

			if current != start:
				current.make_closed()

		return False

	def make_grid(rows, width):
		grid = []
		gap = width // rows
		for i in range(rows):
			grid.append([])
		
			for j in range(rows):
				
				spot = Spot(i, j, gap, rows)
				grid[i].append(spot)

		return grid

	def draw_grid(win, rows, width):
		gap = width // rows

		for i in range(rows):
			pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
			for j in range(rows):
				
					
				pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

	def draw(win, grid, rows, width):
		win.fill(WHITE)

		for row in grid:
			
			for spot in row:
				spot.draw(win)

		geolocalisation.draw_grid(win, rows, width)
		pygame.display.update()


	def get_clicked_pos(pos, rows, width):
		gap = width // rows
		y, x = pos

		row = y // gap
		col = x // gap

		return row, col

	def instructionf():
		global RR,DD,UU,LL,RD,RU,LD,LU,Commande,len_commande
		if LD!=0:
			#Commande.append(["LD",LD])
			Commande=Commande+"LD"+";"+str(LD)+";"
			LD=0
			len_commande+=2
		if RU!=0:
			#Commande.append(["RU",RU])
			Commande=Commande+"RU"+";"+str(RU)+";"
			RU=0
			len_commande+=2
		if RD!=0:
			#Commande.append(["RD",RD])
			Commande=Commande+"RD"+";"+str(RD)+";"
			RD=0
			len_commande+=2
		if UU!=0:
			#Commande.append(["UU",UU])
			Commande=Commande+"UU"+";"+str(UU)+";"
			UU=0
			len_commande+=2
		if DD!=0:
			#Commande.append(["DD",DD])
			Commande=Commande+"DD"+";"+str(DD)+";"
			DD=0
			len_commande+=2
		if LL!=0:
			#Commande.append(["LL",LL])
			Commande=Commande+"LL"+";"+str(LL)+";"
			LL=0
			len_commande+=2

		if LU!=0:
			#Commande.append(["LU",LU])
			Commande=Commande+"LD"+";"+str(LD)+";"
			LU=0
			len_commande+=2
		if RR!=0:
			#Commande.append(["RR",RR])
			Commande=Commande+"RR"+";"+str(RR)+";"
			RR=0
			len_commande+=2
	def instruction1(i):
		global  pixels,Commande,RR,DD,UU,LL,RD,RU,LD,LU,pixel_distance,len_commande


		if  pixels[i][0]==pixels[i][1]+1 and pixels[i][2]==pixels[i][3]:
			if LD!=0:
				#Commande.append(["LD",LD])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LD=0
				len_commande+=2
			if RU!=0:
				#Commande.append(["RU",RU])
				Commande=Commande+"RU"+";"+str(RU)+";"
				RU=0
				len_commande+=2
			if RD!=0:
				#Commande.append(["RD",RD])
				Commande=Commande+"RD"+";"+str(RD)+";"
				RD=0
				len_commande+=2
			if UU!=0:
				#Commande.append(["UU",UU])
				Commande=Commande+"UU"+";"+str(UU)+";"
				UU=0
				len_commande+=2
			if DD!=0:
				#Commande.append(["DD",DD])
				Commande=Commande+"DD"+";"+str(DD)+";"
				DD=0
				len_commande+=2
			if LL!=0:
				#Commande.append(["LL",LL])
				Commande=Commande+"LL"+";"+str(LL)+";"
				LL=0
				len_commande+=2

			if LU!=0:
				#Commande.append(["LU",LU])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LU=0
				len_commande+=2
			
			
			#print ("RR")
			
			RR=RR+1*pixel_distance
		
		elif  pixels[i][0]==pixels[i][1]-1 and pixels[i][2]==pixels[i][3]:
			if LD!=0:
				#Commande.append(["LD",LD])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LD=0
				len_commande+=2
			if RU!=0:
				#Commande.append(["RU",RU])
				Commande=Commande+"RU"+";"+str(RU)+";"
				RU=0
				len_commande+=2
			if RD!=0:
				#Commande.append(["RD",RD])
				Commande=Commande+"RD"+";"+str(RD)+";"
				RD=0
				len_commande+=2
			if UU!=0:
				#Commande.append(["UU",UU])
				Commande=Commande+"UU"+";"+str(UU)+";"
				UU=0
				len_commande+=2
			if DD!=0:
				#Commande.append(["DD",DD])
				Commande=Commande+"DD"+";"+str(DD)+";"
				DD=0
				len_commande+=2

			if LU!=0:
				#Commande.append(["LU",LU])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LU=0
				len_commande+=2
			if RR!=0:
				#Commande.append(["RR",RR])
				Commande=Commande+"RR"+";"+str(RR)+";"
				RR=0
				len_commande+=2
			#print ("LL")
			LL=LL+1*pixel_distance
		
		
		elif  pixels[i][0]==pixels[i][1] and pixels[i][2]==pixels[i][3]+1:
			if LD!=0:
				#Commande.append(["LD",LD])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LD=0
				len_commande+=2
			if RU!=0:
				#Commande.append(["RU",RU])
				Commande=Commande+"RU"+";"+str(RU)+";"
				RU=0
				len_commande+=2
			if RD!=0:
				#Commande.append(["RD",RD])
				Commande=Commande+"RD"+";"+str(RD)+";"
				RD=0
				len_commande+=2
			if UU!=0:
				#Commande.append(["UU",UU])
				Commande=Commande+"UU"+";"+str(UU)+";"
				UU=0
				len_commande+=2
			if LL!=0:
				#Commande.append(["LL",LL])
				Commande=Commande+"LL"+";"+str(LL)+";"
				LL=0
				len_commande+=2

			if LU!=0:
				#Commande.append(["LU",LU])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LU=0
				len_commande+=2
			if RR!=0:
				#Commande.append(["RR",RR])
				Commande=Commande+"RR"+";"+str(RR)+";"
				RR=0
				len_commande+=2
			#print ("DD")
			DD=DD+1*pixel_distance
		

		elif pixels[i][0]==pixels[i][1] and pixels[i][2]==pixels[i][3]-1:
			if LD!=0:
				#Commande.append(["LD",LD])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LD=0
				len_commande+=2
			if RU!=0:
				#Commande.append(["RU",RU])
				Commande=Commande+"RU"+";"+str(RU)+";"
				RU=0
				len_commande+=2
			if RD!=0:
				#Commande.append(["RD",RD])
				Commande=Commande+"RD"+";"+str(RD)+";"
				RD=0
				len_commande+=2
			if DD!=0:
				#Commande.append(["DD",DD])
				Commande=Commande+"DD"+";"+str(DD)+";"
				DD=0
				len_commande+=2
			if LL!=0:
				#Commande.append(["LL",LL])
				Commande=Commande+"LL"+";"+str(LL)+";"
				LL=0
				len_commande+=2

			if LU!=0:
				#Commande.append(["LU",LU])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LU=0
				len_commande+=2
			if RR!=0:
				#Commande.append(["RR",RR])
				Commande=Commande+"RR"+";"+str(RR)+";"
				RR=0
				len_commande+=2
			#print ("UU")
			UU=UU+1*pixel_distance


		elif  pixels[i][0]==pixels[i][1]+1 and pixels[i][2]==pixels[i][3]+1:
			if LD!=0:
				#Commande.append(["LD",LD])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LD=0
				len_commande+=2
			if RU!=0:
				#Commande.append(["RU",RU])
				Commande=Commande+"RU"+";"+str(RU)+";"
				RU=0
				len_commande+=2
			if UU!=0:
				#Commande.append(["UU",UU])
				Commande=Commande+"UU"+";"+str(UU)+";"
				UU=0
				len_commande+=2
			if DD!=0:
				#Commande.append(["DD",DD])
				Commande=Commande+"DD"+";"+str(DD)+";"
				DD=0
				len_commande+=2
			if LL!=0:
				#Commande.append(["LL",LL])
				Commande=Commande+"LL"+";"+str(LL)+";"
				LL=0
				len_commande+=2

			if LU!=0:
				#Commande.append(["LU",LU])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LU=0
				len_commande+=2
			if RR!=0:
				#Commande.append(["RR",RR])
				Commande=Commande+"RR"+";"+str(RR)+";"
				RR=0
				len_commande+=2
			#print ("RD")
			RD=RD+1.4*pixel_distance

		elif pixels[i][0]==pixels[i][1]+1 and pixels[i][2]==pixels[i][3]-1:
			if LD!=0:
				#Commande.append(["LD",LD])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LD=0
				len_commande+=2
			if RD!=0:
				#Commande.append(["RD",RD])
				Commande=Commande+"RD"+";"+str(RD)+";"
				RD=0
				len_commande+=2
			if UU!=0:
				#Commande.append(["UU",UU])
				Commande=Commande+"UU"+";"+str(UU)+";"
				UU=0
				len_commande+=2
			if DD!=0:
				#Commande.append(["DD",DD])
				Commande=Commande+"DD"+";"+str(DD)+";"
				DD=0
				len_commande+=2
			if LL!=0:
				#Commande.append(["LL",LL])
				Commande=Commande+"LL"+";"+str(LL)+";"
				LL=0
				len_commande+=2

			if LU!=0:
				#Commande.append(["LU",LU])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LU=0
				len_commande+=2
			if RR!=0:
				#Commande.append(["RR",RR])
				Commande=Commande+"RR"+";"+str(RR)+";"
				RR=0
				len_commande+=2
			#print ("RU")		
			RU=RU+1.4*pixel_distance
		

		elif  pixels[i][0]==pixels[i][1]-1 and pixels[i][2]==pixels[i][3]+1:
		
			if RU!=0:
				#Commande.append(["RU",RU])
				Commande=Commande+"RU"+";"+str(RU)+";"
				RU=0
				len_commande+=2
			if RD!=0:
				#Commande.append(["RD",RD])
				Commande=Commande+"RD"+";"+str(RD)+";"
				RD=0
				len_commande+=2
			if UU!=0:
				#Commande.append(["UU",UU])
				Commande=Commande+"UU"+";"+str(UU)+";"
				UU=0
				len_commande+=2
			if DD!=0:
				#Commande.append(["DD",DD])
				Commande=Commande+"DD"+";"+str(DD)+";"
				DD=0
				len_commande+=2
			if LL!=0:
				#Commande.append(["LL",LL])
				Commande=Commande+"LL"+";"+str(LL)+";"
				LL=0
				len_commande+=2

			if LU!=0:
				#Commande.append(["LU",LU])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LU=0
				len_commande+=2
			if RR!=0:
				#Commande.append(["RR",RR])
				Commande=Commande+"RR"+";"+str(RR)+";"
				RR=0
				len_commande+=2
			#print ("LD")		
			LD=LD+1.4*pixel_distance

		elif  pixels[i][0]==pixels[i][1]-1 and pixels[i][2]==pixels[i][3]-1:
			if LD!=0:
				#Commande.append(["LD",LD])
				Commande=Commande+"LD"+";"+str(LD)+";"
				LD=0
				len_commande+=2
			if RU!=0:
				#Commande.append(["RU",RU])
				Commande=Commande+"RU"+";"+str(RU)+";"
				RU=0
				len_commande+=2
			if RD!=0:
				#Commande.append(["RD",RD])
				Commande=Commande+"RD"+";"+str(RD)+";"
				RD=0
				len_commande+=2
			if UU!=0:
				#Commande.append(["UU",UU])
				Commande=Commande+"UU"+";"+str(UU)+";"
				UU=0
				len_commande+=2
			if DD!=0:
				#Commande.append(["DD",DD])
				Commande=Commande+"DD"+";"+str(DD)+";"
				DD=0
				len_commande+=2
			if LL!=0:
				#Commande.append(["LL",LL])
				Commande=Commande+"LL"+";"+str(LL)+";"
				LL=0
				len_commande+=2
			if RR!=0:
				#Commande.append(["RR",RR])
				Commande=Commande+"RR"+";"+str(RR)+";"
				RR=0
				len_commande+=2
			#print ("LU")		
			LU=LU+1.4*pixel_distance
	def obstacles(grid):
		global pix,precision
		A=pix-4*precision
		for i in range (A):
			bar = grid[i][0]
			bar.make_barrier()
		for i in range (int(A*14.6/37.1)):
			bar = grid[0][i]
			bar.make_barrier()
		for i in range (int(A*0.7/37.1)+1): ##porte acceuil 1
			bar = grid[int(A*28.7/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*1.3/37.1),int(A*2/37.1)):
			bar = grid[int(A*28.7/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*14.6/37.1)):
			bar = grid[A][i]
			bar.make_barrier()
		for i in range (int(A*1.9/37.1)):
			bar = grid[i][int(A*2/37.1)]
			bar.make_barrier()
		for i in range (int(A*2/37.1),int(A*9.2/37.1)):
			bar = grid[int(A*1.9/37.1)-1][i]
			bar.make_barrier()
		for i in range (int(A*1.9/37.1)):
			bar = grid[i][int(A*9.2/37.1)-1]
			bar.make_barrier()
		for i in range (int(A*2.3/37.1)+1,int(A*5.4/37.1)):
			bar = grid[i][int(A*2/37.1)]
			bar.make_barrier()
		for i in range (int(A*2/37.1),int(A*6.5/37.1)):
			bar = grid[int(A*2.3/37.1)+1][i]
			bar.make_barrier()
		for i in range (int(A*6.7/37.1),int(A*8.5/37.1)):
			bar = grid[int(A*2.3/37.1)+1][i]
			bar.make_barrier()
		for i in range (int(A*8.7/37.1)+1,int(A*9.2/37.1)):
			bar = grid[int(A*2.3/37.1)+1][i]
			bar.make_barrier()
		for i in range (int(A*2.3/37.1)+1,int(A*5.4/37.1)):
			bar = grid[i][int(A*6.1/37.1)]
			bar.make_barrier()
		for i in range (int(A*2/37.1),int(A*9.2/37.1)):
			bar = grid[int(A*5.4/37.1)-1][i]
			bar.make_barrier()
		for i in range (A):
			bar = grid[i][int(A*14.6/37.1)]
			bar.make_barrier()
		### escalier 
		for i in range (int(A*0.6/37.1),int(A*1.3/37.1)):
			bar = grid[int(A*3/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*0.6/37.1),int(A*1.3/37.1)):
			bar = grid[int(A*5.5/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*3/37.1),int(A*5.5/37.1)):
			bar = grid[i][int(A*0.6/37.1)]
			bar.make_barrier()
		for i in range (int(A*3/37.1),int(A*5.5/37.1)):
			bar = grid[i][int(A*1.3/37.1)-1]
			bar.make_barrier()
		for i in range (int(A*0.6/37.1),int(A*1.3/37.1)):
			bar = grid[int(A*18.6/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*0.6/37.1),int(A*1.3/37.1)):
			bar = grid[int(A*22/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*18.6/37.1),int(A*22/37.1)):
			bar = grid[i][int(A*0.6/37.1)]
			bar.make_barrier()
		for i in range (int(A*18.6/37.1),int(A*22/37.1)):
			bar = grid[i][int(A*1.3/37.1)-1]
			bar.make_barrier()

		########## bloc C now 	
		for i in range (int(A*10/37.1),int(A*14.6/37.1)):
			bar = grid[int(A*16.2/37.1)-1][i]
			bar.make_barrier()
		for i in range (int(A*16.2/37.1),int(A*17.6/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.7/37.1)):
			bar = grid[int(A*17.6/37.1)-1][i]
			bar.make_barrier()
		for i in range (int(A*17.6/37.1),int(A*19.4/37.1)+1):
			bar = grid[i][int(A*12.7/37.1)-1]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.7/37.1)):
			bar = grid[int(A*19.4/37.1)+1][i]
			bar.make_barrier()
		for i in range (int(A*19.4/37.1)+1,int(A*21/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*14.6/37.1)):
			bar = grid[int(A*21/37.1)-1][i]
			bar.make_barrier()
		for i in range (int(A*18/37.1),int(A*19.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*18/37.1),int(A*19.1/37.1)):
			bar = grid[i][int(A*12.2/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
			bar = grid[int(A*18/37.1)-1][i]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
			bar = grid[int(A*19.1/37.1)-1][i]
			bar.make_barrier()
			##
		for i in range (int(A*21/37.1),int(A*24.2/37.1)+1):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*21/37.1)-int(A*8.1/37.1),int(A*24.2/37.1)+1-int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*21/37.1)+int(A*8.1/37.1),int(A*24.2/37.1)+1+int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*21/37.1)-2*int(A*8.1/37.1),int(A*24.2/37.1)+1-2*int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		########## bloc B now 	
		for i in range (int(A*10/37.1),int(A*14.6/37.1)):
			bar = grid[int(A*16.2/37.1)+int(A*8.1/37.1)-1][i]
			bar.make_barrier()
		for i in range (int(A*16.2/37.1)+int(A*8.1/37.1),int(A*17.6/37.1)+int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.7/37.1)):
			bar = grid[int(A*17.6/37.1)-1+int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*17.6/37.1)+int(A*8.1/37.1),int(A*19.4/37.1)+1+int(A*8.1/37.1)):
			bar = grid[i][int(A*12.7/37.1)-1]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.7/37.1)):
			bar = grid[int(A*19.4/37.1)+1+int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*19.4/37.1)+1+int(A*8.1/37.1),int(A*21/37.1)+int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*14.6/37.1)):
			bar = grid[int(A*21/37.1)-1+int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*18/37.1)+int(A*8.1/37.1),int(A*19.1/37.1)+int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*18/37.1)+int(A*8.1/37.1),int(A*19.1/37.1)+int(A*8.1/37.1)):
			bar = grid[i][int(A*12.2/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
			bar = grid[int(A*18/37.1)-1+int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
			bar = grid[int(A*19.1/37.1)-1+int(A*8.1/37.1)][i]
			bar.make_barrier()
			
		########## bloc D now 	
		for i in range (int(A*10/37.1),int(A*14.6/37.1)):
			bar = grid[int(A*16.2/37.1)-int(A*8.1/37.1)-1][i]
			bar.make_barrier()
		for i in range (int(A*16.2/37.1)-int(A*8.1/37.1),int(A*17.6/37.1)-int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.7/37.1)):
			bar = grid[int(A*17.6/37.1)-1-int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*17.6/37.1)-int(A*8.1/37.1),int(A*19.4/37.1)+1-int(A*8.1/37.1)):
			bar = grid[i][int(A*12.7/37.1)-1]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.7/37.1)):
			bar = grid[int(A*19.4/37.1)+1-int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*19.4/37.1)+1-int(A*8.1/37.1),int(A*21/37.1)-int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*14.6/37.1)):
			bar = grid[int(A*21/37.1)-1-int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*18/37.1)-int(A*8.1/37.1),int(A*19.1/37.1)-int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*18/37.1)-int(A*8.1/37.1),int(A*19.1/37.1)-int(A*8.1/37.1)):
			bar = grid[i][int(A*12.2/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
			bar = grid[int(A*18/37.1)-1-int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
			bar = grid[int(A*19.1/37.1)-1-int(A*8.1/37.1)][i]
			bar.make_barrier()
		########## bloc E now 	
		for i in range (int(A*10/37.1),int(A*14.6/37.1)):
			bar = grid[int(A*16.2/37.1)-2*int(A*8.1/37.1)-1][i]
			bar.make_barrier()
		for i in range (int(A*16.2/37.1)-2*int(A*8.1/37.1),int(A*17.6/37.1)-2*int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.7/37.1)):
			bar = grid[int(A*17.6/37.1)-1-2*int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*17.6/37.1)-2*int(A*8.1/37.1),int(A*19.4/37.1)+1-2*int(A*8.1/37.1)):
			bar = grid[i][int(A*12.7/37.1)-1]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.7/37.1)):
			bar = grid[int(A*19.4/37.1)+1-2*int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*19.4/37.1)+1-2*int(A*8.1/37.1),int(A*21/37.1)-2*int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*14.6/37.1)):
			bar = grid[int(A*21/37.1)-1-2*int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*18/37.1)-2*int(A*8.1/37.1),int(A*19.1/37.1)-2*int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*18/37.1)-2*int(A*8.1/37.1),int(A*19.1/37.1)-2*int(A*8.1/37.1)):
			bar = grid[i][int(A*12.2/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
			bar = grid[int(A*18/37.1)-1-2*int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
			bar = grid[int(A*19.1/37.1)-1-2*int(A*8.1/37.1)][i]
			bar.make_barrier()
			#####
		for i in range (int(A*2.3/37.1)+1,int(A*3.4/37.1)):
			bar = grid[i][int(A*9.2/37.1)-1]
			bar.make_barrier()	
		for i in range (int(A*4.3/37.1),int(A*18/37.1)):
			bar = grid[i][int(A*9.2/37.1)-1]
			bar.make_barrier()	
		for i in range (int(A*5.3/37.1)+1,int(A*18/37.1)):
			bar = grid[i][int(A*2/37.1)]
			bar.make_barrier()	
		for i in range (int(A*19.1/37.1),int(A*21.5/37.1)):
			bar = grid[i][int(A*2/37.1)]
			bar.make_barrier()	
		for i in range (int(A*19.1/37.1),int(A*(21.5+11.8)/37.1)):
			bar = grid[i][int(A*9.2/37.1)-1]
			bar.make_barrier()	
		for i in range (int(A*2/37.1),int(A*9.1/37.1)):
			bar = grid[int(A*18/37.1)-1][i]
			bar.make_barrier()	
		for i in range (int(A*2/37.1),int(A*9.1/37.1)):
			bar = grid[int(A*19.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*2/37.1),int(A*9.1/37.1)):
			bar = grid[int(A*18.4/37.1)+1][i]
			bar.make_barrier()
		for i in range (int(A*2/37.1),int(A*9.1/37.1)):
			bar = grid[int(A*18.6/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*18.4/37.1)+1,int(A*18.6/37.1)):
			bar = grid[i][int(A*2/37.1)]
			bar.make_barrier()
		for i in range (int(A*18.4/37.1)+1,int(A*18.6/37.1)):
			bar = grid[i][int(A*9.2/37.1)-1]
			bar.make_barrier()
		for i in range (int(A*9.1/37.1)-1,int(A*9.4/37.1)): ##porte acceuil 
			bar = grid[int(A*(21.5+11)/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*9.8/37.1)-1,int(A*10/37.1)): ##porte acceuil 
			bar = grid[int(A*(21.5+11)/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*19.1/37.1),int(A*(21.5+11.8)/37.1)):
			bar = grid[i][int(A*2/37.1)]
			bar.make_barrier()
		for i in range (int(A*2/37.1),int(A*9.1/37.1)): ##porte acceuil 
			bar = grid[int(A*(21.5+11.8)/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*(21.5+11.8+1.3)/37.1),int(A*(21.5+11.8+1.3+2.7)/37.1)):
			bar = grid[i][int(A*2/37.1)]
			bar.make_barrier()
		for i in range (int(A*2/37.1),int(A*4.3/37.1)): ##porte acceuil 
			bar = grid[int(A*(21.5+11.8+1.3)/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*(21.5+11.8+1.3)/37.1),int(A*(21.5+11.8+1.3+2.7)/37.1)):
			bar = grid[i][int(A*4.3/37.1)]
			bar.make_barrier()
		########## bloc A now 	
		for i in range (int(A*10/37.1),int(A*14.6/37.1)):
			bar = grid[int(A*16.2/37.1)+2*int(A*8.1/37.1)-1][i]
			bar.make_barrier()
		for i in range (int(A*16.2/37.1)+2*int(A*8.1/37.1),int(A*17.6/37.1)+2*int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.7/37.1)):
			bar = grid[int(A*17.6/37.1)-1+2*int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*17.6/37.1)+2*int(A*8.1/37.1),int(A*19.4/37.1)+1+2*int(A*8.1/37.1)):
			bar = grid[i][int(A*12.7/37.1)-1]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.7/37.1)):
			bar = grid[int(A*19.4/37.1)+1+2*int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*19.4/37.1)+1+2*int(A*8.1/37.1),int(A*21/37.1)+2*int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*14.6/37.1)):
			bar = grid[int(A*21/37.1)-1+2*int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*18/37.1)+2*int(A*8.1/37.1),int(A*19.1/37.1)+2*int(A*8.1/37.1)):
			bar = grid[i][int(A*10/37.1)]
			bar.make_barrier()
		for i in range (int(A*18/37.1)+2*int(A*8.1/37.1),int(A*19.1/37.1)+2*int(A*8.1/37.1)):
			bar = grid[i][int(A*12.2/37.1)]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
			bar = grid[int(A*18/37.1)-1+2*int(A*8.1/37.1)][i]
			bar.make_barrier()
		for i in range (int(A*10/37.1),int(A*12.2/37.1)+1):
			bar = grid[int(A*19.1/37.1)-1+2*int(A*8.1/37.1)][i]
			bar.make_barrier()

class Spot:
	def __init__(self, row, col, width, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.color = WHITE
		self.neighbors = []
		self.width = width
		self.total_rows = total_rows

	def get_pos(self):
		return self.row, self.col
	def get_row(self):
		return self.row
	def get_col(self):
		return self.col

	def is_closed(self):
		return self.color == RED

	def is_open(self):
		return self.color == GREEN

	def is_barrier(self):
		return self.color == BLACK

	def is_start(self):
		return self.color == ORANGE

	def is_end(self):
		return self.color == TURQUOISE

	def reset(self):
		self.color = WHITE

	def make_start(self):
		self.color = ORANGE

	def make_closed(self):
		self.color = RED

	def make_open(self):
		self.color = GREEN

	def make_barrier(self):
		self.color = BLACK

	def make_end(self):
		self.color = TURQUOISE

	def make_path(self):
		self.color = PURPLE
		

	def draw(self, win):
		pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

	def update_neighbors(self, grid, open=1):
		self.neighbors = []
		if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
			self.neighbors.append(grid[self.row + 1][self.col])
		if self.row < self.total_rows - 1 and self.col < self.total_rows - 1 and not grid[self.row + 1][self.col + 1].is_barrier(): #DOWNRIGHT
			self.neighbors.append(grid[self.row + 1][self.col + 1])
		if self.row < self.total_rows - 1 and self.col > 0 and not grid[self.row + 1][self.col - 1].is_barrier(): #DOWNLEFT
			self.neighbors.append(grid[self.row + 1][self.col - 1])
		if self.row > 0 and self.col < self.total_rows - 1 and not grid[self.row - 1][self.col + 1].is_barrier(): #UPRIGHT
			self.neighbors.append(grid[self.row - 1][self.col + 1])
		if self.row > 0 and self.col > 0 and not grid[self.row - 1][self.col - 1].is_barrier(): #UPLEFT
			self.neighbors.append(grid[self.row - 1][self.col - 1])
		if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
			self.neighbors.append(grid[self.row - 1][self.col])
		if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
			self.neighbors.append(grid[self.row][self.col + 1])
		if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
			self.neighbors.append(grid[self.row][self.col - 1])

	def __lt__(self, other):
		return False
voice_data="0"
class francais:
    engine = pyttsx3.init()
    fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0"
    engine.setProperty('voice', fr_voice_id)
    def there_exists(terms):
        for term in terms:
            if term in voice_data:
                return True

    def speak(text):
        engine = pyttsx3.init()
        fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0"
        engine.setProperty('voice', fr_voice_id)
        sum = "0"
        conn.sendall(sum.encode('utf-8'))
        engine.say(text)
        engine.runAndWait()
        sum = "1"
        conn.sendall(sum.encode('utf-8'))
    def record_audio(ask=""):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            if ask:
                francais.speak(ask)
            audio = r.listen(source)
            try:
                voice_data = r.recognize_google(audio, language='fr')
                print(voice_data)

            except:
                voice_data = francais.record_audio("Pardon, pouvez-vous répéter, je vous écoute")
                return voice_data
            return voice_data

    def respond(voice_data):
        # quit
        if francais.there_exists(["au revoir", "ok bye", "stop", "bye"]):
            francais.speak("Votre robot d'assistance s'arrête, au revoir")
            exit()

        # Elearning
        if francais.there_exists(['elearning', 'ouvrir elearning']):
            webbrowser.open_new_tab("https://elearning.emines.um6p.ma/")
            francais.speak("Vous avez l'accés à Elearning")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # Oasis
        if francais.there_exists(['oasis', 'ouvrir oasis']):
            a = "https://emines.oasis.aouka.org/"
            francais.speak("Vous avez l'accés à Oasis")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # Outlook
        if francais.there_exists(['Outlook','outlook', 'ouvrir Outlook']):
            webbrowser.open_new_tab("https://www.office.com/")
            francais.speak("Vous avez l'accés à Outlook")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # Emines website
        if francais.there_exists(['site emines', 'émine']):
            webbrowser.open_new_tab("https://www.emines-ingenieur.org/")
            francais.speak("Vous avez l'accés au site d'EMINES")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # visite UM6P
        if francais.there_exists(["je veux visiter virtuellement l'université", "visite virtuelle"]):
            webbrowser.open_new_tab("https://alpharepgroup.com/um6p_univer/models.php")
            francais.speak("Bienvenue à la visite virtuelle de um6p ")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # time
        if francais.there_exists(['quelle heure est-il', 'heure']):
            strTime = datetime.datetime.now().strftime("%H""heures""%M")
            francais.speak(f"Il est {strTime}")

        # presentation
        if francais.there_exists(['présente-toi', 'qui est tu']):
            francais.speak("Je suis votre robot d'assistance" " Réalisé par les étudiants du quatrième année d'EMINE"
                  "Je suis un projet mécatronique pour rendre service aux visiteurs de l'université mohamed 6 polytechnique")
            time.sleep(3)

        # google
        if francais.there_exists(['google', 'ouvrir google']):
            webbrowser.open_new_tab("https://www.google.com")
            francais.speak("Vous avez l'accés à Google")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # localisation google maps
        if francais.there_exists(["où suis-je exactement", "où suis-je", "où je suis ", "où je suis exactement", "localisation"]):
            webbrowser.open_new_tab("https://www.google.com/maps/search/Where+am+I+?/")
            francais.speak("Selon Google maps, vous devez être quelque part près d'ici")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # météo
        if francais.there_exists(["météo", "combien fait-il de degrés maintenant", "degré"]):
            search_term = voice_data.split("for")[-1]
            webbrowser.open_new_tab(
                "https://www.google.com/search?sxsrf=ACYBGNSQwMLDByBwdVFIUCbQqya-ET7AAA%3A1578847393212&ei=oUwbXtbXDN-C4-EP-5u82AE&q=weather&oq=weather&gs_l=psy-ab.3..35i39i285i70i256j0i67l4j0i131i67j0i131j0i67l2j0.1630.4591..5475...1.2..2.322.1659.9j5j0j1......0....1..gws-wiz.....10..0i71j35i39j35i362i39._5eSPD47bv8&ved=0ahUKEwiWrJvwwP7mAhVfwTgGHfsNDxsQ4dUDCAs&uact=5")
            francais.speak("Voici ce que j'ai trouvé pour la météo sur Google")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # BDD
        Noms =['Nicolas', 'Cheimanoff', 'Khadija', 'AitHadouch', 'Frédéric',
            'Fontane', 'Bouchikhi', 'Reda', 'Fatiha','Elabdellaoui']
        jobs = ['directeur','logistique','responsable','recherche']
        l1 = ['Cheimanoff', 'Nicolas','Directeur','Nicolas.CHEIMANOFF@emines.um6p.ma','B']
        l2 = ['Fontane', 'Frederic',"Directeur de l'enseignement",'Frederic.FONTANE@emines.um6p.ma','C']
        l3 = ['Elabdellaoui', 'Fatiha','Responsable de scolarite','Fatiha.ELABDELLAOUI@emines.um6p.ma','C']
        l4 = ['AitHadouch', 'Khadija','Assistante de direction','Khadija.AITHADOUCH@emines.um6p.ma','B']
        l5 = ['Bouchikhi', 'Reda','Responsable logistique','Reda.Bouchikhi@emines.um6p.ma','D']
        if francais.there_exists(Noms):
            list = voice_data.split()
            term = set(list) & set(Noms)
            term0 = " ".join(term)
            if term0 in Noms:
                answer = francais.record_audio("voulez vous savoir qui est " + term0)
                francais.respond(answer)
                if 'oui' in answer:
                    if term0 in ['Nicolas','Cheimanoff']:
                        l = l1
                    elif term0 in ['Frédéric', 'Fontane']:
                        l = l2
                    elif term0 in ['Fatiha', 'Elabdellaoui']:
                        l = l3
                    elif term0 in ['Khadija', 'AitHadouch']:
                        l = l4
                    elif term0 in ['Reda', 'Bouchikhi']:
                        l = l5
                    y = json.dumps(l)
                    k = str(y)[1:-1]
                    conn1.send(bytes(k,"utf-8"))
                    engine = pyttsx3.init()
                    sum = "0"
                    conn.sendall(sum.encode('utf-8'))
                    engine.say(l[1] + l[0])
                    engine.say(l[2] + "de l'EMINES  ")
                    engine.say(" son bureau se  trouve dans le bloc   " + l[4])
                    engine.say(" et voici son email qui s'affiche sur l'ecran si vous voulez le contacter ")
                    engine.runAndWait()
                    sum = "1"
                    conn.sendall(sum.encode('utf-8'))
        elif francais.there_exists(jobs):
            list1 = voice_data.split()
            term1 = set(list1) & set(jobs)
            job = " ".join(term1)
            list2 = job.split()
            list2.reverse()
            print(list2)
            if any(item in jobs for item in list2):
                answer = francais.record_audio("voulez vous savoir qui est " + job + "de l'EMINES")
                francais.respond(answer)
                if 'oui' in answer:
                    if any(item in ['directeur',"l'enseignement"] for item in list2):
                        l = l1
                    if any(item in ['logistique','responsable'] for item in list2):
                        l = l5
                engine = pyttsx3.init()
                sum = "0"
                conn.sendall(sum.encode('utf-8'))
                engine.say(l[1] + l[0])
                engine.say(l[2] + "de l'EMINES  ")
                engine.say(" son bureau se  trouve dans le bloc   " + l[4])
                engine.say(" et voici son email qui s'affiche sur l'ecran si vous voulez le contacter ")
                engine.runAndWait()
                sum = "0"
                conn.sendall(sum.encode('utf-8'))
        #else:
            #voice = francais.record_audio("Je vous ai pas compris, pouvez vous repeter?")
            #francais.respond(voice)

    #francais.speak("Bienvenus à l'université Mohammed 6 polytechnique, je suis votre robot d'assistance. Quelle est votre question ?")
    def start():
        while True:
            voice_data = francais.record_audio("je vous écoute")
            francais.respond(voice_data)
            time.sleep(10)
            if francais.there_exists(["j'ai besoin de votre aide"]):
                voice_data = francais.record_audio("je vous écoute")
                print(voice_data, ...)
                francais.respond(voice_data)
            else:
                time.sleep(5)
                voice_data = francais.record_audio("Avez-vous besoin de mon aide")
				#voice_data = francais.record_audio("je vous écoute")
                print(voice_data, ...)
                if francais.there_exists(['oui']):
                    voice_data = francais.record_audio("je vous écoute")
                    print(voice_data, ...)
                    francais.respond(voice_data)
                    continue
                if francais.there_exists(['non','merci']):
                    francais.speak("Votre estes sur le point de quitter les questions generales")
                    print("Votre estes sur le point de quitter les questions generales")
                    break




class anglais:
    global voice_data
    engine = pyttsx3.init()
    fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
    engine.setProperty('voice', fr_voice_id)
    def there_exists(terms):
        for term in terms:
            if term in voice_data:
                return True

    def speak(text):
        engine = pyttsx3.init()
        fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
        engine.setProperty('voice', fr_voice_id)
        sum = "0"
        conn.sendall(sum.encode('utf-8'))
        engine.say(text)
        engine.runAndWait()
        sum = "1"
        conn.sendall(sum.encode('utf-8'))
    def record_audio(ask=""):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            if ask:
                anglais.speak(ask)
            audio = r.listen(source)
            try:
                voice_data = r.recognize_google(audio, language='en-US')
                print(voice_data)

            except:
                voice_data = anglais.record_audio("sorry, I couldn't understand you,please repeat")
                return voice_data
            return voice_data

    def respond(voice_data):
        # quit
        if anglais.there_exists(["good by", "ok bye", "stop", "bye", "see you"]):
            anglais.speak("you are now leaving questions loop")
            exit()

        # Elearning
        if anglais.there_exists(['elearning', 'open elearning']):
            webbrowser.open_new_tab("https://elearning.emines.um6p.ma/")
            anglais.speak("you have acces to elearning")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # Oasis
        if anglais.there_exists(['oasis', 'open oasis']):
            a = "https://emines.oasis.aouka.org/"
            anglais.speak("you have acces to Oasis")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # Outlook
        if anglais.there_exists(['Outlook','outlook', 'open Outlook']):
            webbrowser.open_new_tab("https://www.office.com/")
            anglais.speak("you have acces to Outlook")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # Emines website
        if anglais.there_exists(['site emines', 'émine']):
            webbrowser.open_new_tab("https://www.emines-ingenieur.org/")
            anglais.speak("you have acces to EMINES")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # visite UM6P
        if anglais.there_exists(["i want a virtual visit to this university", "virtual visit"]):
            webbrowser.open_new_tab("https://alpharepgroup.com/um6p_univer/models.php")
            anglais.speak("you're welcome to the virtual visit to Emines")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # time
        if anglais.there_exists(['what time is it', 'time']):
            strTime = datetime.datetime.now().strftime("%H""heures""%M")
            anglais.speak(f"it is {strTime}")

        # presentation
        if anglais.there_exists(['who are you', 'present yourself']):
            anglais.speak("I'am your assitante robot" " Realized by the students of the fourth year of EMINE"
                  "I am a mechatronics project to serve visitors to the mohamed 6 polytechnic university")
            time.sleep(3)

        # google
        if anglais.there_exists(['google', 'open google']):
            webbrowser.open_new_tab("https://www.google.com")
            anglais.speak("Vous avez l'accés à Google")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # localisation google maps
        if anglais.there_exists(["where am I exactly", "where am i", "where am i exactly ", "where am I", "location"]):
            webbrowser.open_new_tab("https://www.google.com/maps/search/Where+am+I+?/")
            anglais.speak("According to Google maps, you must be somewhere near here")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # météo
        if anglais.there_exists(["weather forecast", "how many degrees now", "weather"]):
            search_term = voice_data.split("for")[-1]
            webbrowser.open_new_tab(
                "https://www.google.com/search?sxsrf=ACYBGNSQwMLDByBwdVFIUCbQqya-ET7AAA%3A1578847393212&ei=oUwbXtbXDN-C4-EP-5u82AE&q=weather&oq=weather&gs_l=psy-ab.3..35i39i285i70i256j0i67l4j0i131i67j0i131j0i67l2j0.1630.4591..5475...1.2..2.322.1659.9j5j0j1......0....1..gws-wiz.....10..0i71j35i39j35i362i39._5eSPD47bv8&ved=0ahUKEwiWrJvwwP7mAhVfwTgGHfsNDxsQ4dUDCAs&uact=5")
            anglais.speak("Voici ce que j'ai trouvé pour la météo sur Google")
            time.sleep(3)
            os.system("taskkill /f /im " + browserExe)
        # BDD
        Noms =['Nicolas', 'Cheimanoff', 'Khadija', 'AitHadouch', 'Frédéric',
            'Fontane', 'Bouchikhi', 'Reda', 'Fatiha','Elabdellaoui']
        jobs = ['director','logistics','head','manager','research']
        l1 = ['Cheimanoff', 'Nicolas','director','Nicolas.CHEIMANOFF@emines.um6p.ma','B']
        l2 = ['Fontane', 'Frederic',"Director of Education",'Frederic.FONTANE@emines.um6p.ma','C']
        l3 = ['Elabdellaoui', 'Fatiha','Head of education','Fatiha.ELABDELLAOUI@emines.um6p.ma','C']
        l4 = ['AitHadouch', 'Khadija','Executive assistant','Khadija.AITHADOUCH@emines.um6p.ma','B']
        l5 = ['Bouchikhi', 'Reda','Logistics manager','Reda.Bouchikhi@emines.um6p.ma','D']
        if anglais.there_exists(Noms):
            list = voice_data.split()
            term = set(list) & set(Noms)
            term0 = " ".join(term)
            if term0 in Noms:
                answer = anglais.record_audio("do you want to know who is " + term0)
                anglais.respond(answer)
                if 'yes' in answer:
                    if term0 in ['Nicolas','Cheimanoff']:
                        l = l1
                    elif term0 in ['Frédéric', 'Fontane']:
                        l = l2
                    elif term0 in ['Fatiha', 'Elabdellaoui']:
                        l = l3
                    elif term0 in ['Khadija', 'AitHadouch']:
                        l = l4
                    elif term0 in ['Reda', 'Bouchikhi']:
                        l = l5
                    y = json.dumps(l)
                    k = str(y)[1:-1]
                    conn1.send(bytes(k,"utf-8"))
                    engine = pyttsx3.init()
                    sum = "0"
                    conn.sendall(sum.encode('utf-8'))
                    engine.say(l[1] + l[0])
                    engine.say(l[2] + "de l'EMINES  ")
                    engine.say(" his office is in the block   " + l[4])
                    engine.say(" and here is his email which is displayed on the screen if you want to contact him ")
                    engine.runAndWait()
                    sum = "1"
                    conn.sendall(sum.encode('utf-8'))
        elif anglais.there_exists(jobs):
            list1 = voice_data.split()
            term1 = set(list1) & set(jobs)
            job = " ".join(term1)
            list2 = job.split()
            list2.reverse()
            print(list2)
            if any(item in jobs for item in list2):
                answer = anglais.record_audio("do you want to know who is " + job + "of EMINES")
                anglais.respond(answer)
                if 'yes' in answer:
                    if any(item in ['director',"education"] for item in list2):
                        l = l1
                    if any(item in ['logistics','manager'] for item in list2):
                        l = l5
                engine = pyttsx3.init()
                sum = "0"
                conn.sendall(sum.encode('utf-8'))
                engine.say(l[1] + l[0])
                engine.say(l[2] + "de l'EMINES  ")
                engine.say(" his office is in the block   " + l[4])
                engine.say(" and here is his email which is displayed on the screen if you want to contact him ")
                engine.runAndWait()
                sum = "0"
                conn.sendall(sum.encode('utf-8'))
        #else:
            #voice = anglais.record_audio("Je vous ai pas compris, pouvez vous repeter?")
            #anglais.respond(voice)

    #anglais.speak("Bienvenus à l'université Mohammed 6 polytechnique, je suis votre robot d'assistance. Quelle est votre question ?")
    def start():
        while True:
            voice_data = anglais.record_audio("I'm listening to you")
            anglais.respond(voice_data)
            time.sleep(10)
            if anglais.there_exists(["I need your help"]):
                voice_data = anglais.record_audio("I'm listening to you")
                print(voice_data, ...)
                anglais.respond(voice_data)
            else:
                time.sleep(5)
                voice_data = anglais.record_audio("Do you need my help")
				#voice_data = anglais.record_audio("je vous écoute")
                print(voice_data, ...)
                if anglais.there_exists(['yes']):
                    voice_data = anglais.record_audio("I'm listening to you")
                    print(voice_data, ...)
                    anglais.respond(voice_data)
                    continue
                if anglais.there_exists(['no','thank you']):
                    anglais.speak("Your are about to leave general questions")
                    print("Your are about to leave general questions")
                    break

def geo(destination_1):
    global xf,yf
    xf=320*precision
    yf=95*precision
    geolocalisation.main_zakaria(pygame.display.set_mode((WIDTH, WIDTH2)), WIDTH)

class hind_class:
	def boucle(conn):
		#text1 is the one we get from the voice function
		words1 = text1.split() #the sentence to a list of words
		converted_words1 = [x.upper() for x in words1] #the words' list en majuscule
		words = converted_words1 #new list of words with all the words majuscule
		print(words)
		exit_list = ['REVOIR', 'BYE', 'PROCHAINE']
		text2 = set(words) & set(exit_list) #setting the intersection between the two lists words and exit_list
		str_val = " ".join(text2) #making the intersection words a string to use it
		text02 = str_val #the new text variable we going to work with
		#two conditions whether the visitor wanna leave or complete
		if text02 in exit_list:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			text_say = "au revoir, et bienvenue à emines"
			engine = pyttsx3.init()
			engine.say(text_say)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
		else:
			#define the list we using to match the departement
			list1 = ['RÉCEPTION', 'ENTRÉE', 'BLOC', 'A']
			list2 = ['DÉPARTEMENT', 'BLOC', 'NICOLAS', 'CHEIMANOFF', 'KHADIJA', 'AITHADOUCH', 'FRÉDÉRIC', 'DIRECTION',
					'DIRECTION EMINES', 'NICO', 'SAAD', 'KHATAB', 'B']
			list3 = ['DÉPARTEMENT', 'FATIHA', 'ABDELAOUI', 'C', 'BLOC', 'FOYER', 'SCOLARITÉ','ZINEB']
			list4 = ['DÉPARTEMENT', 'BLOC', 'D', 'REDA', 'BOUCHIKHI', 'HAJAR', 'KHOUKH']
			list5 = ['POLE', ' SANTÉ', 'BLOC', 'E', 'MÉDECIN', 'SANITAIRE', 'INFIRMERIE']
			list6 = ['ETECH', 'LABO', 'LABORATOIRE', 'BLOC']
			a = []
			hind = ['Réception', 'département direction', 'département de scolarité',
					'département logistique', 'département santé', 'E-tech'] #list of the departement names
			#setting the matching words between the original words list we got from the visitor text said and the lists of departements
			list7 = sorted(set(list1) & set(words), key=lambda k: list1.index(k))
			a.append(list7)
			list8 = sorted(set(list2) & set(words), key=lambda k: list2.index(k))
			a.append(list8)
			list9 = sorted(set(list3) & set(words), key=lambda k: list3.index(k))
			a.append(list9)
			list10 = sorted(set(list4) & set(words), key=lambda k: list4.index(k))
			a.append(list10)
			list11 = sorted(set(list5) & set(words), key=lambda k: list5.index(k))
			a.append(list11)
			list12 = sorted(set(list6) & set(words), key=lambda k: list6.index(k))
			a.append(list12)
			# print(a)
			max(a) #the list with a max of matching words
			print(max(a))
			for i in range(6):
				if max(a) == a[i]: #for i bind with max list matched get it and then from the department list names take the name corresponding to it
					print(hind[i])
					global département_déplacement
					département_déplacement = hind[i]
					Noms = ['NICOLAS', 'KHADIJA', 'FRÉDÉRIC', 'BOUCHIKHI', 'INFIRMIÈRE', 'MÉDECIN', 'FATIHA', 'KAWTAR',
							'HAJAR', 'KHATAB',
							'REDA', 'HAJAR', 'ZINEB', 'KHOUKH', 'ABDELAOUI', 'CHEIMANOFF', 'AITHADOUCH', 'FRÉDÉRIC',
							'NICO', 'SAAD','ORCHI'] #list of all the  emines' staff names
					Noms1 = ['FRÉDÉRIC','NICOLAS', 'KHADIJA', 'KHATAB', 'SAAD', 'CHEIMANOFF', 'AITHADOUCH', 'NICO']  # direction
					Noms2 = ['ZINEB', 'KHOUKH', 'ABDELAOUI', 'FATIHA', 'KAWTAR']  # scolarité
					Noms3 = ['REDA', 'HAJAR', 'BOUCHIKHI', 'ORCHI']  # logistique
					Noms4 = ['INFIRMIÈRE', 'MÉDECIN']  # health center
					global NOMS
					if département_déplacement == 'département direction':
						NOMS = Noms1
					elif département_déplacement == 'département de scolarité':
						NOMS = Noms2
					elif département_déplacement == 'département logistique':
						NOMS = Noms3
					elif département_déplacement == 'département santé':
						NOMS = Noms4
					term0 = set(words) & set(Noms)
					str_val = " ".join(term0)
					global term1
					term1 = str_val
					print(term1)
					if term1 in words:  # see if one of the words in the sentence is the word we want
						sum = "0"
						conn.sendall(sum.encode('utf-8'))
						text_term1_say = "voulez vous allez au " + département_déplacement + " chez " + term1 #term1 is the name of the person we want
						engine = pyttsx3.init()
						engine.say(text_term1_say)
						engine.runAndWait()
						sum = "1"
						conn.sendall(sum.encode('utf-8'))
						global s
						s = 2
						print(s)
						hind_class.register(conn)
					else:
						sum = "0"
						conn.sendall(sum.encode('utf-8'))
						text_term_say = "voulez vous allez au" + département_déplacement + "chez une personne précise?"
						engine = pyttsx3.init()
						engine.say(text_term_say)
						engine.runAndWait()
						sum = "1"
						conn.sendall(sum.encode('utf-8'))
						s = 3
						hind_class.register(conn)
	def boucle1(conn):
		words2 = text22.split()
		print(words2)
		words22 = [x.upper() for x in words2]  # new list of words with all the words majuscule
		print(words22)
		rsp = " ".join(words22)  # making the intersection words a string to use it
		if 'OUI' in rsp :
			print('ok')
			print(NOMS)
			if term1 in NOMS:
				sum = "0"
				conn.sendall(sum.encode('utf-8'))
				text__say = "d'accord, je vais vous guidez, s'il vous plait suivez moi"
				engine = pyttsx3.init()
				engine.say(text__say)
				engine.runAndWait()
				sum = "1"
				conn.sendall(sum.encode('utf-8'))
				geo(term1)
			else:
				sum = "0"
				conn.sendall(sum.encode('utf-8'))
				text__say_else = "le nom et le département que vous voulez ne sont pas dépendants "
				engine = pyttsx3.init()
				engine.say(text__say_else)
				engine.runAndWait()
				sum = "1"
				conn.sendall(sum.encode('utf-8'))
				hind_class.main_hind(conn)
		else:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			text__say1 = "je vous est pas bien compris, pouvez redire "
			engine = pyttsx3.init()
			engine.say(text__say1)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			global s
			s = 2
			hind_class.register(conn)
	def boucle2(conn):
		Text = text3.split()
		print(Text)
		words3 = [x.upper() for x in Text]  # new list of words with all the words majuscule
		print(words3)
		text33 = " ".join(words3)
		if 'OUI' in text33:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			text0 = "chez qui?"
			engine = pyttsx3.init()
			engine.say(text0)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			global s
			s=4
			hind_class.register(conn)
		else:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			engine = pyttsx3.init()
			engine.say("D'accord, suivez moi je vais vous emmenez au " + département_déplacement)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			geo(département_déplacement)
	def boucle3(conn):
		words4 = text4.split()
		print(words4)
		z = [x.upper() for x in words4]
		Noms = ['NICOLAS', 'KHADIJA', 'FRÉDÉRIC', 'BOUCHIKHI', 'INFIRMIÈRE', 'MÉDECIN', 'FATIHA', 'KAWTAR',
				'HAJAR', 'KHATAB',
				'REDA', 'HAJAR', 'ZINEB', 'KHOUKH', 'ABDELAOUI', 'CHEIMANOFF', 'AITHADOUCH', 'FRÉDÉRIC',
				'NICO', 'SAAD']  # list of all the  emines' staff names
		term4 = set(z) & set(Noms)
		str_val = " ".join(term4)
		global term44
		term44 = str_val
		print(term44)
		if term44 in Noms:
			term41 = term44.lower()
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			text_say3 = "voulez vous allez au" + département_déplacement + "chez " + term41
			engine = pyttsx3.init()
			engine.say(text_say3)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			global s
			s=5
			hind_class.register(conn)
		else:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			engine = pyttsx3.init()
			engine.say("je trouve pas la personne que vous voulez")
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			hind_class.main_hind(conn)
	def boucle4(conn):
		word5 = text5.split()
		print(word5)
		word5 = [x.upper() for x in word5]  # new list of words with all the words majuscule
		print(word5)
		text55 = " ".join(word5)
		if 'OUI' in text55:
			print(NOMS)
			if term44 in NOMS:
				sum = "0"
				conn.sendall(sum.encode('utf-8'))
				text_say_term44 = "d'accord, je vais vous guidez, s'il vous plait suivez moi"
				engine = pyttsx3.init()
				engine.say(text_say_term44)
				engine.runAndWait()
				sum = "1"
				conn.sendall(sum.encode('utf-8'))
				geo(term44)
			else:
				sum = "0"
				conn.sendall(sum.encode('utf-8'))
				text__say_else0 = "le nom et le département que vous voulez ne sont pas dans le même emplacement "
				engine = pyttsx3.init()
				engine.say(text__say_else0)
				engine.runAndWait()
				sum = "1"
				conn.sendall(sum.encode('utf-8'))
				hind_class.main_hind(conn)
		else:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			text0 = "je trouve pas la personne que vous voulez"
			engine = pyttsx3.init()
			engine.say(text0)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			hind_class.main_hind(conn)
	#the function that changes the audio file contain to a text and calls the function
	#with all the possibilities to answer called boucle
	#def function voice; audio to text
	def voice(conn):
		print('voicetotext')
		AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "C:\\filtered1.wav")
		# use the audio file as the audio source
		r = sr.Recognizer()
		with sr.AudioFile(AUDIO_FILE) as source:
			audio = r.record(source)  # read the entire audio file
		# recognize speech using Google Speech Recognition
		
		try:
			TEXT = r.recognize_google(audio, language="fr-FR")
			print(TEXT)
			global s
			if s==0:
				text0 = TEXT
				print(text0)
				#taking the response from the visitor, the robot even start explaining and then runs the main function or just run it is
				#the visitor wanna pass the explanation
				Text0 = text0.split()
				list_continuer = ['continue','continuer','terminer','entendre']
				text001 = set(Text0) & set(list_continuer)  # setting the intersection between the two lists words and exit_list
				str_val = " ".join(text001)  # making the intersection words a string to use it
				text001 = str_val  # the new text variable we going to work with
				# two conditions whether the visitor wanna leave or complete
				try:
					if text001 in list_continuer:
						sum = "0"
						conn.sendall(sum.encode('utf-8'))
						text01 = "d'accord, je commencerais :Au niveau du rez de chaussée se trouve cinq département; premièrement la réception qui se trouve au bloc A: " \
								"si vous voulez mieux connaitre université deuxièment la direction qui se trouve au bloc B: se trouve le bureau " \
								"du directeur Nicolas Cheimanoff , son assistante Khadiija Aitahadouch et le bureau de Saad Aitkhatab le " \
								"responsable de communication de EMINES troixièment la scolarité qui se trouve au bloc C: elle se trouve en face" \
								" du foyer des élèves, se trouve le bureau de Fatiha Alabdelaoui responsable de scolarité de EMINES, Zineb " \
								"Elkhoukh assistante du directeur d'éducation et de la recherche, et messieur Orchi responsable d'impression " \
								"quatrièment la logistique qui se trouve au bloc D: il se trouve le bureau de Reda Elbouchikhi responsable " \
								"hébergement son assistante hajar Azerkouk cinquièment le pôle santé: ou se trouve le médecin et les infirmières " \
								"et dernièrement E-tech: le club de robotique Emines si vous voulez consulter les projets effectués par nos " \
								"étudiants."
						engine = pyttsx3.init()
						engine.setProperty("rate", 250)
						fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0"
						# Use female french voice
						engine.setProperty('voice', fr_voice_id)
						engine.say(text01)
						engine.runAndWait()
						sum = "1"
						conn.sendall(sum.encode('utf-8'))
						hind_class.main_hind(conn)
					else:
						hind_class.main_hind(conn)
					
				except sr.UnknownValueError:
					sum = "0"
					conn.sendall(sum.encode('utf-8'))
					text_say__0_try = "je vous ai pas bien entendu, pouvez vous répètez, merci"
					engine = pyttsx3.init()
					engine.say(text_say__0_try)
					engine.runAndWait()
					sum = "1"
					conn.sendall(sum.encode('utf-8'))
					hind_class.register(conn)

			elif s == 1:
				global text1
				text1 = TEXT
				print(text1)
				hind_class.boucle(conn)
				
			elif s == 2:
				global text22
				text22 = TEXT
				print(text22)
				hind_class.boucle1(conn)
			
			elif s==3:
				global text3
				text3 = TEXT
				print(text3)
				hind_class.boucle2(conn)
				
			elif s==4:
				global text4
				text4 = TEXT
				print(text4)
				hind_class.boucle3(conn)
				
			elif s==5:
				global text5
				text5 = TEXT
				print(text5)
				hind_class.boucle4(conn)
				
		except sr.UnknownValueError:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			text_say__0 = "je vous ai pas bien entendu, pouvez vous répètez, merci"
			engine = pyttsx3.init()
			engine.say(text_say__0)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			hind_class.register(conn)
		#the function that takes the voices from the microphone register it in a wav file and then filter it and call the function
		#that changes the audio file contain into a text called voice
	#def register; register voice and filter it
	def register(conn):
		print('register')
		freq = 41000
		#i should calculate the duration using the volume
		duration = 6
		recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
		sd.wait()
		write("C:\\test1.wav", freq, recording)
		fname = 'C:\\test1.wav'
		outname = 'C:\\filtered1.wav'
		cutOffFrequency = 200.0
		def running_mean(x, windowSize):
			cumsum = np.cumsum(np.insert(x, 0, 0))
			return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize
		def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved=True):
			if sample_width == 1:
				dtype = np.uint8  # unsigned char
			elif sample_width == 2:
				dtype = np.int16  # signed 2-byte short
			else:
				raise ValueError("Only supports 8 and 16 bit audio formats.")
			channels = np.frombuffer(raw_bytes, dtype=dtype)
			if interleaved:
				# channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
				channels.shape = (n_frames, n_channels)
				channels = channels.T
			else:
				# channels are not interleaved. All samples from channel M occur before all samples from channel M-1
				channels.shape = (n_channels, n_frames)
			return channels
		data, samplerate = sf.read(fname)
		sf.write(fname, data, samplerate, subtype='PCM_16')
		with contextlib.closing(wave.open(fname, 'rb')) as spf:
			sampleRate = spf.getframerate()
			ampWidth = spf.getsampwidth()
			nChannels = spf.getnchannels()
			nFrames = spf.getnframes()
			# Extract Raw Audio from multi-channel Wav File
			signal = spf.readframes(nFrames * nChannels)
			spf.close()
			channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)
			# get window size
			freqRatio = (cutOffFrequency / sampleRate)
			N = int(math.sqrt(0.196196 + freqRatio ** 2) / freqRatio)
			# Use moviung average (only on first channel)
			filtered = running_mean(channels[0], N).astype(channels.dtype)
			wav_file = wave.open(outname, "w")
			wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
			wav_file.writeframes(filtered.tobytes('C'))
			wav_file.close()
			hind_class.voice(conn)
	#the main function here is the first one calles in the programm and it counts n the number of no responses once n is higher or
	#equal to 2 due to the bad quality of voice...etc, the visitor gonna be directed to tape his direction on an interface 
	#the main function; count n and execute the register function for n<=1 / n>=2 --> interface

	def main_hind(conn):
		global n
		n += 1
		print(n)
		print(n)
		if n <= 2:# n is the nombre of tries and said no as response
			text_main0 = "qu'elle est votre question?"
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			engine = pyttsx3.init()
			engine.say(text_main0)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			global s
			s=1
			print(s)
			hind_class.register(conn)
		else:
			text_main1 = "je préfére que vous tapez votre question, afin que je puisse vous comprendre"
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			engine = pyttsx3.init()
			engine.say(text_main1)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			#ajouter une interface

	def start(conn):
		#the starting text where the robot represente himself and offer to explain more about the school
		sum = "0"
		conn.sendall(sum.encode('utf-8'))
		text00 = "Bonjour chers visiteurs je suis votre robot d'assistance, je suis destiné à vous aider à se déplacer au sein de " \
				"l'EMINES et vous diriger vers votre destination et aussi répondre à vos questions."
		text000="je commencera par vous décrire le plan de l'école pour que je puisse vous aider efficacement à se déplacer,      mais  vous pouvez dépasser " \
				"cette partie descriptive durant cette étape en disons je passe,   sinon et si vous voulez l'entendre disez simplement je continue, "
		#text0000="je vous renseigne aussi qu'une fois mon micro est ouvert pour vous entendre mon interface devienne verte."
		engine = pyttsx3.init()
		engine.setProperty("rate", 250)
		fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_FR-FR_HORTENSE_11.0"
		# Use female french voice
		engine.setProperty('voice', fr_voice_id)
		engine.say(text00)
		engine.say(text000)
		#engine.say(text0000)
		engine.runAndWait()
		sum = "1"
		conn.sendall(sum.encode('utf-8'))
		hind_class.register(conn)


class hind_class_en:
	def boucle():
		#text1 is the one we get from the voice function
		words1 = text1.split() #the sentence to a list of words
		converted_words1 = [x.upper() for x in words1] #the words' list en majuscule
		words = converted_words1 #new list of words with all the words majuscule
		print(words)
		exit_list = ['goodbye', 'good bye', 'GOODBYE']
		text2 = set(words) & set(exit_list) #setting the intersection between the two lists words and exit_list
		str_val = " ".join(text2) #making the intersection words a string to use it
		text02 = str_val #the new text variable we going to work with
		#two conditions whether the visitor wanna leave or complete
		if text02 in exit_list:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			text_say = "goodbye, and welcome emines"
			engine = pyttsx3.init()
			engine.say(text_say)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
		else:
			#define the list we using to match the departement
			list1 = ['RECEPTION', 'ENTRANCE', 'BLOC', 'A']
			list2 = ['DEPARTEMENT', 'BLOC', 'NICOLAS', 'CHEIMANOFF', 'KHADIJA', 'AITHADOUCH', 'FREDERIC', 'DIRECTION',
					'DIRECTION EMINES', 'NICO', 'SAAD', 'KHATAB', 'B']
			list3 = ['DEPARTEMENT', 'FATIHA', 'ABDELAOUI', 'C', 'BLOC', 'FOYER', 'SCHOOLING','ZINEB']
			list4 = ['DEPARTEMENT', 'BLOC', 'D', 'REDA', 'BOUCHIKHI', 'HAJAR', 'KHOUKH']
			list5 = ['POLE', ' HEALTH', 'BLOC', 'E', 'DOCTOR', 'SANITARY', 'INFIRMARY']
			list6 = ['ETECH', 'LABO', 'LABORATORY', 'BLOC']
			a = []
			hind = ['Reception', 'management department', 'education department',
					'logistics Department', 'health department', 'E-tech'] #list of the departement names
			#setting the matching words between the original words list we got from the visitor text said and the lists of departements
			list7 = sorted(set(list1) & set(words), key=lambda k: list1.index(k))
			a.append(list7)
			list8 = sorted(set(list2) & set(words), key=lambda k: list2.index(k))
			a.append(list8)
			list9 = sorted(set(list3) & set(words), key=lambda k: list3.index(k))
			a.append(list9)
			list10 = sorted(set(list4) & set(words), key=lambda k: list4.index(k))
			a.append(list10)
			list11 = sorted(set(list5) & set(words), key=lambda k: list5.index(k))
			a.append(list11)
			list12 = sorted(set(list6) & set(words), key=lambda k: list6.index(k))
			a.append(list12)
			# print(a)
			max(a) #the list with a max of matching words
			print(max(a))
			for i in range(6):
				if max(a) == a[i]: #for i bind with max list matched get it and then from the department list names take the name corresponding to it
					print(hind[i])
					global département_déplacement
					département_déplacement = hind[i]
					Noms = ['NICOLAS', 'KHADIJA', 'FREDERIC', 'BOUCHIKHI', 'NURSE', 'DOCTOR', 'FATIHA', 'KAWTAR',
							'HAJAR', 'KHATAB',
							'REDA', 'HAJAR', 'ZINEB', 'KHOUKH', 'ABDELAOUI', 'CHEIMANOFF', 'AITHADOUCH', 'FREDERIC',
							'NICO', 'SAAD','ORCHI'] #list of all the  emines' staff names
					Noms1 = ['FREDERIC','NICOLAS', 'KHADIJA', 'KHATAB', 'SAAD', 'CHEIMANOFF', 'AITHADOUCH', 'NICO']  # direction
					Noms2 = ['ZINEB', 'KHOUKH', 'ABDELAOUI', 'FATIHA', 'KAWTAR']  # scolarité
					Noms3 = ['REDA', 'HAJAR', 'BOUCHIKHI', 'ORCHI']  # logistique
					Noms4 = ['NURSE', 'DOCTOR']  # health center
					global NOMS
					if département_déplacement == 'management department':
						NOMS = Noms1
					elif département_déplacement == 'education department':
						NOMS = Noms2
					elif département_déplacement == 'logistics Department':
						NOMS = Noms3
					elif département_déplacement == 'health department':
						NOMS = Noms4
					term0 = set(words) & set(Noms)
					str_val = " ".join(term0)
					global term1
					term1 = str_val
					print(term1)
					if term1 in words:  # see if one of the words in the sentence is the word we want
						sum = "0"
						conn.sendall(sum.encode('utf-8'))
						text_term1_say = "do you want to go to " + département_déplacement + " to " + term1 #term1 is the name of the person we want
						engine = pyttsx3.init()
						engine.say(text_term1_say)
						engine.runAndWait()
						sum = "1"
						conn.sendall(sum.encode('utf-8'))
						global s
						s = 2
						print(s)
						hind_class_en.register()
					else:
						sum = "0"
						conn.sendall(sum.encode('utf-8'))
						text_term_say = "do you want to go to" + département_déplacement + "to a specific person?"
						engine = pyttsx3.init()
						engine.say(text_term_say)
						engine.runAndWait()
						sum = "1"
						conn.sendall(sum.encode('utf-8'))
						s = 3
						hind_class_en.register()
	def boucle1():
		words2 = text22.split()
		print(words2)
		words22 = [x.upper() for x in words2]  # new list of words with all the words majuscule
		print(words22)
		rsp = " ".join(words22)  # making the intersection words a string to use it
		if 'YES' in rsp :
			print('ok')
			print(NOMS)
			if term1 in NOMS:
				sum = "0"
				conn.sendall(sum.encode('utf-8'))
				text__say = "ok i will guide you, please follow me"
				engine = pyttsx3.init()
				engine.say(text__say)
				engine.runAndWait()
				sum = "1"
				conn.sendall(sum.encode('utf-8'))
				geo(term1)
			else:
				sum = "0"
				conn.sendall(sum.encode('utf-8'))
				text__say_else = "the name and the department you want are not dependent "
				engine = pyttsx3.init()
				engine.say(text__say_else)
				engine.runAndWait()
				sum = "1"
				conn.sendall(sum.encode('utf-8'))
				hind_class_en.main_hind()
		else:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			text__say1 = "I don't understand you well, can you repeat it? "
			engine = pyttsx3.init()
			engine.say(text__say1)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			global s
			s = 2
			hind_class_en.register()
	def boucle2():
		Text = text3.split()
		print(Text)
		words3 = [x.upper() for x in Text]  # new list of words with all the words majuscule
		print(words3)
		text33 = " ".join(words3)
		if 'YES' in text33:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			text0 = "where?"
			engine = pyttsx3.init()
			engine.say(text0)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			global s
			s=4
			hind_class_en.register()
		else:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			engine = pyttsx3.init()
			engine.say("Alright, follow me I'll take you to" + département_déplacement)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			geo(département_déplacement)
	def boucle3():
		words4 = text4.split()
		print(words4)
		z = [x.upper() for x in words4]
		Noms = ['NICOLAS', 'KHADIJA', 'FREDÉRIC', 'BOUCHIKHI', 'INFIRMIÈRE', 'MEDECIN', 'FATIHA', 'KAWTAR',
				'HAJAR', 'KHATAB',
				'REDA', 'HAJAR', 'ZINEB', 'KHOUKH', 'ABDELAOUI', 'CHEIMANOFF', 'AITHADOUCH', 'FREDERIC',
				'NICO', 'SAAD']  # list of all the  emines' staff names
		term4 = set(z) & set(Noms)
		str_val = " ".join(term4)
		global term44
		term44 = str_val
		print(term44)
		if term44 in Noms:
			term41 = term44.lower()
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			text_say3 = "do you want to go to" + département_déplacement + "to " + term41
			engine = pyttsx3.init()
			engine.say(text_say3)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			global s
			s=5
			hind_class_en.register()
		else:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			engine = pyttsx3.init()
			engine.say("I can't find the person you want")
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			hind_class_en.main_hind()
	def boucle4():
		word5 = text5.split()
		print(word5)
		word5 = [x.upper() for x in word5]  # new list of words with all the words majuscule
		print(word5)
		text55 = " ".join(word5)
		if 'YES' in text55:
			print(NOMS)
			if term44 in NOMS:
				sum = "0"
				conn.sendall(sum.encode('utf-8'))
				text_say_term44 = "ok i will guide you, please follow me"
				engine = pyttsx3.init()
				engine.say(text_say_term44)
				engine.runAndWait()
				sum = "1"
				conn.sendall(sum.encode('utf-8'))
				geo(term44)
			else:
				sum = "0"
				conn.sendall(sum.encode('utf-8'))
				text__say_else0 = "the name and the department you want are not in the same location "
				engine = pyttsx3.init()
				engine.say(text__say_else0)
				engine.runAndWait()
				sum = "1"
				conn.sendall(sum.encode('utf-8'))
				hind_class_en.main_hind()
		else:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			text0 = "I can't find the person you want"
			engine = pyttsx3.init()
			engine.say(text0)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			hind_class_en.main_hind()
	#the function that changes the audio file contain to a text and calls the function
	#with all the possibilities to answer called boucle
	#def function voice; audio to text
	def voice():
		print('voicetotext')
		AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), "C:\\filtered1.wav")
		# use the audio file as the audio source
		r = sr.Recognizer()
		with sr.AudioFile(AUDIO_FILE) as source:
			audio = r.record(source)  # read the entire audio file
		# recognize speech using Google Speech Recognition
		
		try:
			TEXT = r.recognize_google(audio, language="en_US")
			print(TEXT)
			global s
			if s==0:
				text0 = TEXT
				print(text0)
				#taking the response from the visitor, the robot even start explaining and then runs the main function or just run it is
				#the visitor wanna pass the explanation
				Text0 = text0.split()
				list_continuer = ['carry on','continue','keep going','keep on']
				text001 = set(Text0) & set(list_continuer)  # setting the intersection between the two lists words and exit_list
				str_val = " ".join(text001)  # making the intersection words a string to use it
				text001 = str_val  # the new text variable we going to work with
				# two conditions whether the visitor wanna leave or complete
				try:
					if text001 in list_continuer:
						sum = "0"
						conn.sendall(sum.encode('utf-8'))
						text01 = "Okay, I'll start: On the ground floor level are five departments; firstly the reception which is in block A: " \
								"if you want to get to know the university better, the direction which is located in block B: is the office " \
								"of director Nicolas Cheimanoff, his assistant Khadiija Aitahadouch and the office of Saad Aitkhatab the " \
								"communication manager of EMINES thirdly the schooling which is in block C: it is opposite" \
								" of the students' home, is the office of Fatiha Alabdelaoui responsible for education of EMINES, Zineb" \
								"Elkhoukh assistant to the director of education and research, and Mr. Orchi responsible for printing " \
								"fourth is the logistics which is in block D: there is the office of Reda Elbouchikhi responsible " \
								"hébergement son assistante hajar Azerkouk cinquièment le pôle santé: ou se trouve le médecin et les infirmières " \
								"and most recently E-tech: the robotics club Emines if you want to consult the projects carried out by our " \
								"students."
						engine = pyttsx3.init()
						engine.setProperty("rate", 250)
						fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
						# Use female french voice
						engine.setProperty('voice', fr_voice_id)
						engine.say(text01)
						engine.runAndWait()
						sum = "1"
						conn.sendall(sum.encode('utf-8'))
						hind_class_en.main_hind()
					else:
						hind_class_en.main_hind()

				except sr.UnknownValueError:
					sum = "0"
					conn.sendall(sum.encode('utf-8'))
					text_say__0_try = "I didn't hear you right, can you repeat yourself, thank you"
					engine = pyttsx3.init()
					engine.say(text_say__0_try)
					engine.runAndWait()
					sum = "1"
					conn.sendall(sum.encode('utf-8'))
					hind_class_en.register()

			elif s == 1:
				global text1
				text1 = TEXT
				print(text1)
				hind_class_en.boucle()

			elif s == 2:
				global text22
				text22 = TEXT
				print(text22)
				hind_class_en.boucle1()

			elif s==3:
				global text3
				text3 = TEXT
				print(text3)
				hind_class_en.boucle2()

			elif s==4:
				global text4
				text4 = TEXT
				print(text4)
				hind_class_en.boucle3()

			elif s==5:
				global text5
				text5 = TEXT
				print(text5)
				hind_class_en.boucle4()

		except sr.UnknownValueError:
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			text_say__0 = "I didn't hear you right, can you repeat yourself, thank you"
			engine = pyttsx3.init()
			engine.say(text_say__0)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			hind_class_en.register()
		#the function that takes the voices from the microphone register it in a wav file and then filter it and call the function
		#that changes the audio file contain into a text called voice
	#def register; register voice and filter it
	def register():
		print('register')
		freq = 41000
		#i should calculate the duration using the volume
		duration = 6
		recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
		sd.wait()
		write("C:\\test1.wav", freq, recording)
		fname = 'C:\\test1.wav'
		outname = 'C:\\filtered1.wav'
		cutOffFrequency = 200.0
		def running_mean(x, windowSize):
			cumsum = np.cumsum(np.insert(x, 0, 0))
			return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize
		def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved=True):
			if sample_width == 1:
				dtype = np.uint8  # unsigned char
			elif sample_width == 2:
				dtype = np.int16  # signed 2-byte short
			else:
				raise ValueError("Only supports 8 and 16 bit audio formats.")
			channels = np.frombuffer(raw_bytes, dtype=dtype)
			if interleaved:
				# channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
				channels.shape = (n_frames, n_channels)
				channels = channels.T
			else:
				# channels are not interleaved. All samples from channel M occur before all samples from channel M-1
				channels.shape = (n_channels, n_frames)
			return channels
		data, samplerate = sf.read(fname)
		sf.write(fname, data, samplerate, subtype='PCM_16')
		with contextlib.closing(wave.open(fname, 'rb')) as spf:
			sampleRate = spf.getframerate()
			ampWidth = spf.getsampwidth()
			nChannels = spf.getnchannels()
			nFrames = spf.getnframes()
			# Extract Raw Audio from multi-channel Wav File
			signal = spf.readframes(nFrames * nChannels)
			spf.close()
			channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)
			# get window size
			freqRatio = (cutOffFrequency / sampleRate)
			N = int(math.sqrt(0.196196 + freqRatio ** 2) / freqRatio)
			# Use moviung average (only on first channel)
			filtered = running_mean(channels[0], N).astype(channels.dtype)
			wav_file = wave.open(outname, "w")
			wav_file.setparams((1, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
			wav_file.writeframes(filtered.tobytes('C'))
			wav_file.close()
			hind_class_en.voice()
	#the main function here is the first one calles in the programm and it counts n the number of no responses once n is higher or
	#equal to 2 due to the bad quality of voice...etc, the visitor gonna be directed to tape his direction on an interface 
	#the main function; count n and execute the register function for n<=1 / n>=2 --> interface

	def main_hind():
		global n
		n += 1
		print(n)
		print(n)
		if n <= 2:# n is the nombre of tries and said no as response
			text_main0 = "qu'elle est votre question?"
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			engine = pyttsx3.init()
			engine.say(text_main0)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			global s
			s=1
			print(s)
			hind_class_en.register()
		else:
			text_main1 = "I prefer that you type your question, so that I can understand you"
			sum = "0"
			conn.sendall(sum.encode('utf-8'))
			engine = pyttsx3.init()
			engine.say(text_main1)
			engine.runAndWait()
			sum = "1"
			conn.sendall(sum.encode('utf-8'))
			#ajouter une interface

	def start():
		#the starting text where the robot represente himself and offer to explain more about the school
		sum = "0"
		conn.sendall(sum.encode('utf-8'))
		text00 = "Hello dear visitors I am your assistance robot, I am intended to help you move within " \
				"EMINES and direct you to your destination and also answer your questions."
		text000="I will start by describing the school plan to you so that I can effectively help you get around, but you can exceed " \
				"this descriptive part during this step let's say I pass, if not and if you want to hear it just say I continue, "
		#text0000="je vous renseigne aussi qu'une fois mon micro est ouvert pour vous entendre mon interface devienne verte."
		engine = pyttsx3.init()
		engine.setProperty("rate", 182)
		fr_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
		# Use female french voice
		engine.setProperty('voice', fr_voice_id)
		engine.say(text00)
		engine.say(text000)
		#engine.say(text0000)
		engine.runAndWait()
		sum = "1"
		conn.sendall(sum.encode('utf-8'))
		hind_class_en.register()


francais.speak("Bienvenus à l'université Mohammed 6 polytechnique, je suis votre robot d'assistance.veuillez choisir la langue que vous préfèrez")
anglais.speak("welcome to R2D2 robot, please choose your language")
data = conn.recv(1024)
data=str(data.decode("utf-8"))
print(data)

if data=="1":
	while True:
		francais.speak("est-ce que vous vouler le deplacement ou les questions ou bien vous voulez quitter?")
		data1 = conn1.recv(1024)
		data1=str(data1.decode("utf-8"))
		print(data1)
		b="9"
		if data1=="5" :
			break
		elif data1=="4":
			francais.start()
			conn.sendall(b.encode('utf-8'))
		elif data1=="3":
			hind_class.start(conn)
			conn.sendall(b.encode('utf-8'))
		time.sleep(1)
if data=="1":
	francais.speak("au revoir")

if data=="2":
	while True:
		anglais.speak("do you want to ask question or to go to an office or to quit")
		data1 = conn1.recv(1024)
		data1=str(data1.decode("utf-8"))
		print(data1)
		b="9"
		if data1=="5" :
			break
		elif data1=="4":
			anglais.start()
			conn.sendall(b.encode('utf-8'))
		elif data1=="3":
			hind_class_en.start()
			conn.sendall(b.encode('utf-8'))
		time.sleep(1)


if data=="2":
	anglais.speak("good by")