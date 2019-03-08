import os
import cv2
import time
import argparse
import multiprocessing
import numpy as np
import tensorflow as tf
#from gtts import gTTS

from utils.app_utils import FPS, WebcamVideoStream
from multiprocessing import Queue, Pool, Process
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from utils.defaults import *

import smtplib
import time
import imaplib
import email

FROM_EMAIL  = "sydeitemidentifer@gmail.com"
FROM_PWD    = "SydeFr334all"
SMTP_SERVER = "imap.gmail.com"
SMTP_PORT   = 993

mail = imaplib.IMAP4_SSL(SMTP_SERVER, SMTP_PORT)
mail.login(FROM_EMAIL, FROM_PWD)

# load label map
label_map =  label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
															use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# text to speech
#engine = pyttsx.init()

def detect_objects(image_np, sess, detection_graph, displaced_obj):
	# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
	image_np_expanded = np.expand_dims(image_np, axis=0)
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

	# Each box represents a part of the image where a particular object was detected.
	boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

	# Each score represent how level of confidence for each of the objects.
	# Score is shown on the result image, together with the class label.
	scores = detection_graph.get_tensor_by_name('detection_scores:0')
	classes = detection_graph.get_tensor_by_name('detection_classes:0')
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')

	# Actual detection.
	(boxes, scores, classes, num_detections) = sess.run(
		[boxes, scores, classes, num_detections],
		feed_dict={image_tensor: image_np_expanded})

	class_box = {}
	ref_box = {}

	for i in range(len(scores[0])):
		if category_index[classes[0][i]]['name'] not in ALLOWED_CLASSES:
			scores[0][i] = 0
		else:
			if scores[0][i] >= DETECT_THRESHOLD:
				obj = category_index[classes[0][i]]['name']
				if obj in REFERENCE_OBJECTS:
					ref_box[obj] = np.squeeze(boxes[0][i])
				else:
					class_box[obj] = np.squeeze(boxes[0][i])

	if not ref_box:
		print('No reference object detected.')
		#engine.say('No reference objects detected.')
		#engine.runAndWait()
	elif not class_box:
		print('No objects were detected.')
		#engine.say('No objects were detected.')
		#engine.runAndWait()
	else:
		if displaced_obj not in class_box:
			print('Unable to located %s.' % displaced_obj)
			#engine.say('Unable to locate %s.' % displaced_obj)
		else:
			# find obj midpoint
			obj_midpoint = class_box[displaced_obj][3] - class_box[displaced_obj][1]
			# find closest reference object
			min_dist = None
			closest_ref = None
			for ref in ref_box:
				ref_midpoint = ref_box[ref][3] - ref_box[ref][1]
				if not min_dist or abs(obj_midpoint - ref_midpoint) < min_dist:
					min_dist = abs(obj_midpoint - ref_midpoint)
					closest_ref = ref

			# find position relative to reference object
			ref_xmin = ref_box[closest_ref][1]
			ref_xmax = ref_box[closest_ref][3]
			obj_xmin = class_box[displaced_obj][1]
			obj_xmax = class_box[displaced_obj][3]
			if (ref_xmin < obj_xmin and ref_xmax > obj_xmax) or (obj_xmin < ref_xmin and obj_xmax > ref_xmax): # check if in front
				position = 'front'
			elif (ref_xmax + ref_xmin)/2 > (obj_xmax + obj_xmin)/2: # check if left
				position = 'left'
			else: # right
				position = 'right'

			# say to user
			if position == 'front':
				print('%s is in front of the %s' % (displaced_obj, closest_ref))
			else:
				print('%s is to the %s of the %s' % (displaced_obj, position, closest_ref))

	# Visualization of the results of a detection.
	vis_util.visualize_boxes_and_labels_on_image_array(
		image_np,
		np.squeeze(boxes),
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=8,
		min_score_thresh=DETECT_THRESHOLD)
	return image_np


def worker(input_q, output_q, request_q, displaced_obj):
	# Load a (frozen) Tensorflow model into memory.
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

		sess = tf.Session(graph=detection_graph)

	fps = FPS().start()
	while True:
		# get next frame
		fps.update()
		frame = input_q.get()
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = detect_objects(frame_rgb, sess, detection_graph, displaced_obj)

		# get requested object

		# if there is an object, build output message from objects

		# print and say message

		# put image in output_q
		output_q.put(image)

	fps.stop()
	sess.close()

def checkMail():
    try:
        mail.select('inbox')
        (retcode, messages) = mail.search(None, '(UNSEEN)')

        if (retcode == 'OK'):
            for num in messages[0].split():
                typ, data = mail.fetch(num, '(RFC822)' )
                for response_part in data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_string(response_part[1].decode('utf-8'))
                        email_subject = msg['subject']
                        email_from = msg['from']
                        if (email_subject == "ItemIdentifier."):
                            print(msg.get_payload().strip())
                            typ, data = mail.store(num,'+FLAGS','\\Seen')
    except Exception as e:
        print(str(e))

def request_worker(request_q):
	# wait for object and add to request_q when arrives
	while 1:
	    print('checking mail')
	    checkMail()
	    time.sleep(1)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-src', '--source', dest='video_source', type=int,
						default=-1, help='Device index of the camera.')
	parser.add_argument('-wd', '--width', dest='width', type=int,
						default=1600, help='Width of the frames in the video stream.')
	parser.add_argument('-ht', '--height', dest='height', type=int,
						default=900, help='Height of the frames in the video stream.')
	parser.add_argument('-num-w', '--num-workers', dest='num_workers', type=int,
						default=2, help='Number of workers.')
	parser.add_argument('-q-size', '--queue-size', dest='queue_size', type=int,
						default=5, help='Size of the queue.')
	parser.add_argument('-obj', '--object', dest='displaced_obj', type=str, default='bottle', help='Object to locate.')
	args = parser.parse_args()

	logger = multiprocessing.log_to_stderr()
	logger.setLevel(multiprocessing.SUBDEBUG)

	input_q = Queue(maxsize=args.queue_size)
	output_q = Queue(maxsize=args.queue_size)
	request_q = Queue(maxsize=args.queue_size)
	pool = Pool(args.num_workers, worker, (input_q, output_q, request_q, args.displaced_obj))
	request_p = Process(target=request_worker, args=(request_q,))
	request_p.start()

	video_capture = WebcamVideoStream(src=args.video_source, width=args.width, height=args.height).start()
	#fps = FPS().start()
	#video_capture = cv2.VideoCapture(-1)

	while True:  # fps._numFrames < 120
		frame = video_capture.read()
		input_q.put(frame)

		#t = time.time()

		output_rgb = cv2.cvtColor(output_q.get(), cv2.COLOR_RGB2BGR)
		cv2.imshow('Video', output_rgb)
		#fps.update()

		#print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	#fps.stop()
	#print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
	#print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
	request_p.join()
	pool.terminate()
	video_capture.stop()
	cv2.destroyAllWindows()
