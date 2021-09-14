import argparse
import logging
import time
import os
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import scripts.label_image as label_img
import scripts.label_image_scene as label_img_scene


logger = logging.getLogger('Pose_Action_and_Scene_Understanding')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
address = os.getcwd()
if __name__ == '__main__':
	'''
	1. 필요한 Parameter 받는다.
	parser을 통해 args에 각종 argument들을 받는다.
	'''
	parser = argparse.ArgumentParser(description='tf-human-action-classification')
	parser.add_argument('--image', type=str, required=True)
	parser.add_argument('--show-process', type=bool, default=False,
						help='for debug purpose, if enabled, speed for inference is dropped.')
	parser.add_argument('--model', type=str, default='mobilenet_thin')
	args = parser.parse_args()

	print(args.model)
	'''
	2. model parameter에서 estimator를 로딩한다.
	'''
	# logger.debug('initialization %s : %s' % ('mobilenet_thin', get_graph_path('mobilenet_thin')))
	# e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
	logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
	e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
	print(os.getcwd())
	'''
	3. 이미지를 cv2로 읽는다.
	'''
	frame = cv2.imread(args.image)
	logger.info('cam image=%dx%d' % (frame.shape[1], frame.shape[0]))

	# count = 0
	
	logger.debug('+frame processing+')
	logger.debug('+postprocessing+')
	start_time = time.time()
	cv2.imshow("1", frame)
	input()
	humans = e.inference(frame, upsample_size=4.0)
	output_image = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
	
	logger.debug('+classification+')
	# Getting only the skeletal structure (with white background) of the actual image
	skeleton_image = np.zeros(frame.shape,dtype=np.uint8)
	skeleton_image.fill(255) 
	skeleton_image = TfPoseEstimator.draw_humans(skeleton_image, humans, imgcopy=False)
	
	# Classification
	pose_class = label_img.classify(frame)
	scene_class = label_img_scene.classify(frame)
	end_time = time.time()
	logger.debug('+displaying+')
	cv2.putText(output_image,
				"Predicted Pose: %s" %(pose_class),
				(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				(0, 0, 255), 2)
	cv2.putText(output_image,
				"Predicted Scene: %s" %(scene_class),
				(10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				(0, 0, 255), 2)
	print('\n Overall Evaluation time (1-image): {:.3f}s\n'.format(end_time-start_time))
	cv2.imwrite('show1.png',output_image)
	cv2.imshow('tf-human-action-classification result', output_image)
	cv2.waitKey(0)
	logger.debug('+finished+')
	cv2.destroyAllWindows()

# =============================================================================
# For running the script simply run the following in the cmd prompt/terminal :
# python run_image.py --image=2.jpg
# python run_image.py --image=2.jpg --model=retrained
# =============================================================================
