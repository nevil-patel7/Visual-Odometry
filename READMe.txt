# Visual Odometry
## ENPM673 Spring2020- Assignment5

##Directory Structure-

	        Assignment5-
		|--Code-
		|  |-Oxford_dataset
		|  |-	|-model
		|  | 	|-stereo		#dataset
		|  |-opencv_LibFunc.py		#Using inbuilt functions
		|  |-VisualOdometry.py		#Using custom functions
		|  |-functions.py		#essential functions definitions
		|  |-ReadCameraModel.py
		|  |-UndistortImage.py
		|--Report.pdf
		|--README.md

##System Requirements-
	Python (v3.0.x or later)

##Libraries needed-
	1) OpenCV (opencv-python)
	2) Numpy
	3) Scipy 
	4) matplotlib
	5) random
	6) glob

##Run instructions-
	1) For each file the dataset directories must be updated. 
	2) The submission does not included dataset (The File structure can be updated as directory stucture above)
	3) Run 'VisualOdometry.py' to perform visual odometry using userdefined functions.
	4) Run 'opencv_LibFUnc.py' to perform visual odometry using opencv functions

## Overview

Visual Odometry is a crucial concept in Robotics Perception for estimating the trajectory of the robot (the camera on the robot to be precise). The concepts involved in Visual Odometry are quite the same for SLAM which needless to say is an integral part of Perception.
In this project you are given frames of a driving sequence taken by a camera in a car, and the scripts to extract the intrinsic parameters. Your will implement the different steps to estimate the 3D motion of the camera, and provide as output a plot of the trajectory of the camera. The dataset can be found [here](https://drive.google.com/drive/folders/1f2xHP_l8croofUL_G5RZKmJo2YE9spx9).

