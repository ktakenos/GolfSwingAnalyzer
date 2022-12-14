# GolfSwingAnalyzer
**Demonstration of this code**
https://youtu.be/QRXIYl-tVjI
![Title Slide](https://github.com/ktakenos/GolfSwingAnalyzer/blob/main/images/GolfSwingAnalyzerDemo.jpg)
![Slide #1](https://github.com/ktakenos/GolfSwingAnalyzer/blob/main/images/GolfSwingAnalyzerDemo(1).jpg)
![Slide #2](https://github.com/ktakenos/GolfSwingAnalyzer/blob/main/images/GolfSwingAnalyzerDemo(2).jpg)
![Slide #3](https://github.com/ktakenos/GolfSwingAnalyzer/blob/main/images/GolfSwingAnalyzerDemo(3).jpg)
![Slide #4](https://github.com/ktakenos/GolfSwingAnalyzer/blob/main/images/GolfSwingAnalyzerDemo(4).jpg)


## For Golf Swing Analysis using 3D plotting of body joints converted from privately recorded videos
This code runs an Qt application, where you can analyze your gold swing in terms of your skeleton motion in 3D coordinates.
In order to determine your joint coordinates, you need two videos recorded to your swing of your interest. You can synchronize the two videos in this application so they don't have to be started recording at the same time.
The two cameras are idealy 90 degrees apart with respect to your body. You will need to tell the application that the target direction in the two video frames. In author's case, one camera is just in front of him facing. It is 360 degree camera, Gopro max. The other camera is behind the ball on the target line.<br>

The main window consits of the tabs,  Video, Camera calibration, 3D plot, and 3D data table. <br>
**In the video tab window**, you can seek a specific video frame and make the two videos in synch. The left pane is for the main video and the right is for the sub video. The main video determins the start frame and the end frame. The joint position data is analyzed mainly from the main video, more specifically, x-axis and z-axis, where x-axis is the direction to the target and z-axis points to the sky. The sub video provides the depth information, in other words, positions in y-axis pointing the direction away from the ball. You can seek the video frame using the slidebar under the main video, and differrentiate the frame shift in the sub video. 
For the body estimation, I desinged the application to crop the video frames so that only your body will be passed to the estimation otherwise too many joints will be detected.
The body estimation starts a specific seconds before the selected video frame. It will be useful to limit the actual swing motion only for the body estimation. In auther's case, 1.2 seconds is goood number for the swing motion starts before the impact. The finishing frame also needs to be determined after the selected video frame.<br>

**In the camera calibration tab**, you can adjust the camera facing angles so that the generated 3D plot to be reasonable. I will make a seperate instruction for this tab.<br>

**In the 3D plolt tab**, you will be seeing your skeleton model in 3D space, which utilizes opengl on pyqt using pyopengl.<br>

**In the 3D Data tab**, you see the actual numbers of your joints. The data is stored in a csv file. The column 1 is for x, 2 for y, and 3 for z based on the angles from the main camera. The column 4 is for another z numbers based on the angles from the sub camera. The two cameras need to be calibrated ideally to match the two z numbers. The row 1 to 13 is for one frame. I will explain lator which row number is which joint of you body.<br>


## Things required to make it work:
Packages: python, pyqt5, opencv (incl. numpy), opengl, openpose-pytorch (https://github.com/Hzzone/pytorch-openpose)<br>
Cameras: Any types but two needed to get 3D analysis<br>
Tripods: Optional but recommended.<br>


## The condition that I have been working on:
Windows 10 64bit running on the hardware - Core i5-8400 @2.8GHz, 16GB memory, Nvidia GeForce GTX 1050 Ti<br>
Built in Spyder in Anaconda, python 3.9.7, pyqt 5.12.3, opencv 4.5.3, pyopengl 3.1.6<br>
openpose-pytorch package is copied to the same directory of the .py file. It means that the src, model directories are copied.<br>
Gopro max and sony AS200 are used for the video recording.<br>


## Instruction
Install required packages. Copy the openpose-pytorch to your directory, or modify the source code so that it works.

Go to the driving range or golf studio.<br>
Set up two cameras, one in front, another in right hand side. Level the camras as precisely as possible.<br>
Start recording two cameras.<br>
Hit the ball, maybe a few times.<br>

Store the two video files. MP4 is the default but most video format should work.<br>
Open main video first from the file menu.<br>
Open sub video next from the file menu.<br>
Seek a frame for the impact in the main video.<br>
Seek the same impact frame in the sub video.<br>
Go to the camera calibration tab.<br>
Determin the target direction by moving the slidebar in both main and sub videos.<br>
Input the camera location with respect to the ball for both main and sub cameras then clike "Update" buttons.<br>
Retern to the video tab, check the crop, and click left mouse button where the left upper corner of the crop area and drag to the right bottom corner and then release the button.<br>
Repeat the crop process until you body fits in the the selected area.<br>
Repeat it for sub video.<br>
Chceck body extimation for both main video and sub video.<br>
For modeling 3D for this single frame, click the 3D model button.<br>
Go to 3D plot tab, you should see your skeleton plot.<br>
For modeling 3D plot of all frames during the swing, check 3D Data Save.<br>
Befor export, save the project from the file menu. The processed file will be stored into the same directory of the project file.<br>
From the file menu, select export, then process starts. After finishing the export, files are in the directroy of the project.<br>


## Known issues and things needs to be improved:
Joint positions will be ackward when they are invisible.<br>
Camera calibration is not acurate.<br>
Resized image, cropped image may be too small.<br>

## Things to implement later:
Exporting opengl pixel data convering to video file.<br>
Video Audio to be converted to pose video.<br>
Trace the club head position and plot them as well.<br>
