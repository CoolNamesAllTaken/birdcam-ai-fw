import cv2
import time
import os
import re
import threading
import numpy as np
from flask import Response, Flask

import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter

# Image frame sent to the Flask object
global video_frame
video_frame = None

# Use locks for thread-safe viewing of frames in multiple browsers
global video_frame_lock 
video_frame_lock = threading.Lock()

global annotated_frame
annotated_frame = None

global annotated_frame_lock
annotated_frame_lock = threading.Lock()

MODEL_NAME = "ssd_mobilenet_v3_small_coco_2020_01_14"
MODEL_PATH = os.path.join("..", "models", MODEL_NAME, "model.tflite")
LABELS_PATH = os.path.join("..", "models", MODEL_NAME, "labels.txt")
MODEL_FLOATING = False

MIN_CONF_THRESHOLD = 0.5
SHOW_ANNOTATIONS = False

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FRAME_RATE = 10
USB_CAMERA = True

# Create the Flask object for the application
app = Flask(__name__)

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

def inference_service():
	interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
	interpreter.allocate_tensors()

	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	input_shape = input_details[0]['shape']
	width = input_shape[1]
	height = input_shape[2]

	labels = load_labels(LABELS_PATH)

	while True:
		with video_frame_lock:
			global video_frame
			if video_frame is None:
				continue
				print("skipping frame")
			else:
				frame = video_frame.copy()

		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame_resized = cv2.resize(frame_rgb, (width, height))
		input_data = np.expand_dims(frame_resized, axis=0)

		# Normalize pixel values if using a floating model (i.e. if model is non-quantized)
		if MODEL_FLOATING:
			input_data = (np.float32(input_data) - input_mean) / input_std

		# Perform the actual detection by running the model with the image as input
		# print("width={} height={}".format(width, height))
		# print("input_data.shape={} frame_resized.shape={} frame_rgb.shape={}".format(input_data.shape, frame_resized.shape, frame_rgb.shape))
		interpreter.set_tensor(input_details[0]['index'],input_data)
		interpreter.invoke()

		# Retrieve detection results
		boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
		classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
		scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
		#num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

		# Loop over all detections and draw detection box if confidence is above minimum threshold
		for i in range(len(scores)):
			if ((scores[i] > MIN_CONF_THRESHOLD) and (scores[i] <= 1.0)):

				# Get bounding box coordinates and draw box
				# Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
				ymin = int(max(1,(boxes[i][0] * IMAGE_HEIGHT)))
				xmin = int(max(1,(boxes[i][1] * IMAGE_WIDTH)))
				ymax = int(min(IMAGE_HEIGHT,(boxes[i][2] * IMAGE_HEIGHT)))
				xmax = int(min(IMAGE_WIDTH,(boxes[i][3] * IMAGE_WIDTH)))
				
				cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

				# Draw label
				object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
				label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
				labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
				label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
				cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
				cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

		# Draw framerate in corner of frame
		# cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

		global annotated_frame, annotated_frame_lock
		with annotated_frame_lock:
			annotated_frame = frame.copy()

		time.sleep(float(1)/FRAME_RATE)

		# All the results have been drawn on the frame, so it's time to display it.
		# cv2.imshow('Object detector', frame)

class ProtectedVideoCapture:
	def __enter__(self):
		if USB_CAMERA:
			gst_str = (
				'v4l2src device=/dev/video{} ! '
				'video/x-raw, width=(int){}, height=(int){} ! '
				'videoconvert ! appsink').format(
				1, 
				IMAGE_WIDTH, 
				IMAGE_HEIGHT)
			self.stream = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
		else:
			gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction){}/1 ! '
                   'nvvidconv flip-method={} ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(
                   FRAME_RATE, 
                   2, 
                   IMAGE_WIDTH, 
                   IMAGE_HEIGHT)
			self.stream = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
		return self.stream
	def __exit__(self, *args):
		self.stream.release()
		print("bye")

def capture_frame_service():
	global video_frame, video_frame_lock

	# Video capturing from OpenCV
	with ProtectedVideoCapture() as video_capture:
	# video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

		while True and video_capture.isOpened():
			return_key, frame = video_capture.read()
			if not return_key:
				break

			# Create a copy of the frame and store it in the global variable,
			# with thread safe access
			with video_frame_lock:
				video_frame = frame.copy()
			
			time.sleep(float(1) / FRAME_RATE)
			# key = cv2.waitKey(30) & 0xff
			# if key == 27:
			#     break

	# video_capture.release()
		
def encode_frame():
	global video_frame_lock, video_frame, annotated_frame_lock, annotated_frame
		
	while True:
		# Acquire video_frame_lock to access the global video_frame object
		if SHOW_ANNOTATIONS:
			with annotated_frame_lock:
				if annotated_frame is None:
				    continue
				return_key, encoded_image = cv2.imencode(".jpg", annotated_frame)
				# if not return_key:
				#     continue 
		else:
			with video_frame_lock:
				if video_frame is None:
				    continue
				return_key, encoded_image = cv2.imencode(".jpg", video_frame)
				# if not return_key:
				#     continue 
	   

		# Output image as a byte array
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encoded_image) + b'\r\n')
		time.sleep(float(1) / FRAME_RATE)

@app.route("/")
def stream_frames():
	return Response(encode_frame(), mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':

	# Create a thread and attach the method that captures the image frames, to it
	process_thread = threading.Thread(target=capture_frame_service)
	# process_thread.daemon = True

	# Start the thread
	process_thread.start()

	inference_thread = threading.Thread(target=inference_service)
	# inference_thread.daemon = True

	inference_thread.start()

	# start the Flask Web Application
	# While it can be run on any feasible IP, IP = 0.0.0.0 renders the web app on
	# the host machine's localhost and is discoverable by other machines on the same network 
	app.run("0.0.0.0", port="8000")