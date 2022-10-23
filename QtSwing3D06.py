# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 15:40:37 2022

@author: ktakenos@gmail.com

MIT License

Copyright (c) 2022 ktakenos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


For menu bar
https://realpython.com/python-menus-toolbars/
https://www.pythonguis.com/tutorials/pyqt-actions-toolbars-menus/

File dialog
https://qiita.com/Nobu12/items/d5f6cc57274a64170734

For layout
https://zetcode.com/gui/pyqt5/layout/

QTab
https://pythonspot.com/pyqt5-tabs/

QGridLayout
https://zetcode.com/gui/pyqt5/layout/

QTable
https://doc.qt.io/qtforpython-5/PySide2/QtWidgets/QTableWidget.html
https://qiita.com/hoshianaaa/items/8b66ef7f2dcb46dad312

Opencv Image to label
https://symfoware.blog.fc2.com/blog-entry-2302.html

Opengl Widget
https://www.gamedev.net/blogs/entry/2270745-minimal-opengl-example-in-c-qt5-pyqt5-and-typescript-webgl-10/
https://nrotella.github.io/journal/first-steps-python-qt-openGL.html
https://stackoverflow.com/questions/33201384/pyqt-opengl-drawing-simple-scenes

QTimer
https://qiita.com/tak0/items/6dcbfc70e9d583d12911

QSlider
https://pythonbasics.org/qslider/

Mouse events
https://stackoverflow.com/questions/3504522/pyqt-get-pixel-position-and-value-when-mouse-click-on-the-image

Message box
https://pythonbasics.org/pyqt-qmessagebox/

QThread
https://qiita.com/softbase/items/0b3e7d2de006efc7b16d

"""

import os
import os.path
import shutil
import sys
from PyQt5.QtCore import Qt, QTimer, QThread
from PyQt5.QtWidgets import (QMainWindow, QAction, QFileDialog, QApplication, \
                             QMenu, QTabWidget, QWidget, QVBoxLayout, QGridLayout, \
                                 QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,\
                                     QOpenGLWidget, QPushButton, QLineEdit, QSlider,\
                                         QCheckBox, QMessageBox)
from PyQt5.QtGui import (QImage, QPixmap) 
#from PyQt5.QtGui import QIcon
import csv
import cv2
import numpy as np
from OpenGL import GL as GL
import subprocess

from src.body import Body
body_estimation = Body('model/body_pose_model.pth')

class ProjectData():
    #Member Variables
    ProjectFileName = ""
    MainVideoFileName = ""
    SubVideoFileName = ""
    DataFolderName = ""
    MainCameraFovW=360
    MainCameraFovH=180
    MainCameraDist=0.1
    (MainCameraX, MainCameraY, MainCameraZ) = (-0.8, -0.2, 0.6)
    MainCameraCenterAngle = 90
    MainVideoFramePosition = 0
    SubCameraFovW=120
    SubCameraFovH=90
    SubCameraDist=2.5
    (SubCameraX, SubCameraY, SubCameraZ) = (-2.5, 0.3, 0.5)
    SubCameraCenterAngle = 0
    SubVideoFrameDelta = 0

class SkeletonData():
    #Member Variables
    fCSVLoaded = 0
    n3dElements = 4
    nJoints3D = 14
    nFrames = 1
    AnglesMain = []
    AnglesSub = []
    Joints3D = []
    Joints3DSingle = []
    PCenter3D = np.zeros(1, dtype=np.float32)
    ShoulderExtent = 1.6
    ShoulderLineL = np.zeros(1, dtype=np.float32)
    ShoulderLineR = np.zeros(1, dtype=np.float32)
    PelvisExtent = 2.0
    PelvisLineL = np.zeros(1, dtype=np.float32)
    PelvisLineR = np.zeros(1, dtype=np.float32)
    SpineExtent = 0.8
    SpineHigh = np.zeros(1, dtype=np.float32)
    SpineLow = np.zeros(1, dtype=np.float32)
    KneeExtent = 1.2
    KneeLineL = np.zeros(1, dtype=np.float32)
    KneeLineR = np.zeros(1, dtype=np.float32)
    
    def pixel2deg(self, x, xo, w, FA):
        return float(-(x -xo)/w*FA)
    def deg2pixel(self, deg, xo, w, FA):
        return int(xo - w/FA*deg)
    def cameraPosition(self, DistH, degX, degY):
        radX = np.radians(degX)
        radY = np.radians(degY)
        x = -DistH * np.cos(radX)
        y = -DistH * np.sin(radX)
        z = -DistH * np.cos(radX) * np.sin(radY)
        return (x, y, z)
    def JointPosition(self, x1, y1, z1, x2, y2, z2, DegH1, DegV1, DegH2, DegV2):
        radH1 = np.radians(DegH1)
        radV1 = np.radians(DegV1)
        radH2 = np.radians(DegH2)
        radV2 = np.radians(DegV2)
        x = (np.tan(radH1) * x1 - np.tan(radH2) * x2 - (y1-y2))/(np.tan(radH1) - np.tan(radH2))
        y = np.tan(radH2) * (x - x2) + y2
        Dist1 = np.sqrt((x - x1)**2+(y - y1)**2)
        z_1 = z1 + Dist1 * np.tan(radV1)
        Dist2 = np.sqrt((x - x2)**2+(y - y2)**2)
        z_2 = z2 + Dist2 * np.tan(radV2)
        return(x, y, z_1, z_2)
    def Ang2Pos(self):
        # self.Joints3DSingle=np.zeros((self.n3dElements*self.nJoints3D), dtype=np.float32)
        self.Joints3DSingle= [0]  * (self.n3dElements*self.nJoints3D)
        (c1x, c1y, c1z) = (PData.MainCameraX , PData.MainCameraY, PData.MainCameraZ)
        (c2x, c2y, c2z) = (PData.SubCameraX , PData.SubCameraY, PData.SubCameraZ)
        for joint in range(self.nJoints3D):
            (JointX, JointY,JointZ_1, JointZ_2) = \
                SData.JointPosition(c1x, c1y, c1z, c2x, c2y, c2z, \
                                    self.AnglesMain[joint*2], self.AnglesMain[joint*2+1], \
                                    self.AnglesSub[joint*2], self.AnglesSub[joint*2+1])
            self.Joints3DSingle[joint*self.n3dElements+0] = JointX
            self.Joints3DSingle[joint*self.n3dElements+1] = JointY
            self.Joints3DSingle[joint*self.n3dElements+2] = JointZ_1
            self.Joints3DSingle[joint*self.n3dElements+3] = JointZ_2
    def AlterZData(self):
        BackupJoint3D = self.Joints3D.copy()
        for joint in range(self.nJoints3D):
            self.Joints3D[joint*self.n3dElements+2] = BackupJoint3D[joint*self.n3dElements+3]
            self.Joints3D[joint*self.n3dElements+3] = BackupJoint3D[joint*self.n3dElements+2]
    def CalculateExtents(self):
        for idxFrame in range(self.nFrames):
            position = idxFrame * self.nJoints3D*self.n3dElements 
            if(idxFrame==0):
                self.PCenter3D =                           self.Joints3D[position+self.n3dElements*7]
                self.PCenter3D = np.append(self.PCenter3D, self.Joints3D[position+self.n3dElements*7 + 1])
                self.PCenter3D = np.append(self.PCenter3D, self.Joints3D[position+self.n3dElements*7 + 2])
                self.PCenter3D = np.append(self.PCenter3D, self.Joints3D[position+self.n3dElements*7 + 3])
                dx = self.Joints3D[position+self.n3dElements*4]   - self.Joints3D[position+self.n3dElements*1]
                dy = self.Joints3D[position+self.n3dElements*4+1] - self.Joints3D[position+self.n3dElements*1+1]
                dz = self.Joints3D[position+self.n3dElements*4+2] - self.Joints3D[position+self.n3dElements*1+2]
                self.ShoulderLineL =                               self.Joints3D[position+self.n3dElements*4]   + dx * self.ShoulderExtent
                self.ShoulderLineL = np.append(self.ShoulderLineL, self.Joints3D[position+self.n3dElements*4+1] + dy * self.ShoulderExtent)
                self.ShoulderLineL = np.append(self.ShoulderLineL, self.Joints3D[position+self.n3dElements*4+2] + dz * self.ShoulderExtent)
                self.ShoulderLineR =                               self.Joints3D[position+self.n3dElements*1]   - dx * self.ShoulderExtent
                self.ShoulderLineR = np.append(self.ShoulderLineR, self.Joints3D[position+self.n3dElements*1+1] - dy * self.ShoulderExtent)
                self.ShoulderLineR = np.append(self.ShoulderLineR, self.Joints3D[position+self.n3dElements*1+2] - dz * self.ShoulderExtent)
                dx = self.Joints3D[position+self.n3dElements*11]   - self.Joints3D[position+self.n3dElements*8]
                dy = self.Joints3D[position+self.n3dElements*11+1] - self.Joints3D[position+self.n3dElements*8+1]
                dz = self.Joints3D[position+self.n3dElements*11+2] - self.Joints3D[position+self.n3dElements*8+2]
                self.PelvisLineL =                             self.Joints3D[position+self.n3dElements*11]   + dx * self.PelvisExtent
                self.PelvisLineL = np.append(self.PelvisLineL, self.Joints3D[position+self.n3dElements*11+1] + dy * self.PelvisExtent)
                self.PelvisLineL = np.append(self.PelvisLineL, self.Joints3D[position+self.n3dElements*11+2] + dz * self.PelvisExtent)
                self.PelvisLineR =                             self.Joints3D[position+self.n3dElements*8]    - dx * self.PelvisExtent
                self.PelvisLineR = np.append(self.PelvisLineR, self.Joints3D[position+self.n3dElements*8+1]  - dy * self.PelvisExtent)
                self.PelvisLineR = np.append(self.PelvisLineR, self.Joints3D[position+self.n3dElements*8+2]  - dz * self.PelvisExtent)
                dx = self.Joints3D[position+self.n3dElements*0]   - self.Joints3D[position+self.n3dElements*7]
                dy = self.Joints3D[position+self.n3dElements*0+1] - self.Joints3D[position+self.n3dElements*7+1]
                dz = self.Joints3D[position+self.n3dElements*0+2] - self.Joints3D[position+self.n3dElements*7+2]
                self.SpineHigh =                           self.Joints3D[position+self.n3dElements*0]   + dx * self.SpineExtent
                self.SpineHigh = np.append(self.SpineHigh, self.Joints3D[position+self.n3dElements*0+1] + dy * self.SpineExtent)
                self.SpineHigh = np.append(self.SpineHigh, self.Joints3D[position+self.n3dElements*0+2] + dz * self.SpineExtent)
                self.SpineLow =                            self.Joints3D[position+self.n3dElements*7]   - dx * self.SpineExtent
                self.SpineLow = np.append(self.SpineLow,   self.Joints3D[position+self.n3dElements*7+1] - dy * self.SpineExtent)
                self.SpineLow = np.append(self.SpineLow,   self.Joints3D[position+self.n3dElements*7+2] - dz * self.SpineExtent)
                dx = self.Joints3D[position+self.n3dElements*12]   - self.Joints3D[position+self.n3dElements*9]
                dy = self.Joints3D[position+self.n3dElements*12+1] - self.Joints3D[position+self.n3dElements*9+1]
                dz = self.Joints3D[position+self.n3dElements*12+2] - self.Joints3D[position+self.n3dElements*9+2]
                self.KneeLineL =                             self.Joints3D[position+self.n3dElements*12]   + dx * self.KneeExtent
                self.KneeLineL = np.append(self.KneeLineL, self.Joints3D[position+self.n3dElements*12+1] + dy * self.KneeExtent)
                self.KneeLineL = np.append(self.KneeLineL, self.Joints3D[position+self.n3dElements*12+2] + dz * self.KneeExtent)
                self.KneeLineR =                             self.Joints3D[position+self.n3dElements*9]    - dx * self.KneeExtent
                self.KneeLineR = np.append(self.KneeLineR, self.Joints3D[position+self.n3dElements*9+1]  - dy * self.KneeExtent)
                self.KneeLineR = np.append(self.KneeLineR, self.Joints3D[position+self.n3dElements*9+2]  - dz * self.KneeExtent)
            else:
                self.PCenter3D = np.append(self.PCenter3D, self.Joints3D[position+self.n3dElements*7])
                self.PCenter3D = np.append(self.PCenter3D, self.Joints3D[position+self.n3dElements*7 + 1])
                self.PCenter3D = np.append(self.PCenter3D, self.Joints3D[position+self.n3dElements*7 + 2])
                self.PCenter3D = np.append(self.PCenter3D, self.Joints3D[position+self.n3dElements*7 + 3])
                dx = self.Joints3D[position+self.n3dElements*4]   - self.Joints3D[position+self.n3dElements*1]
                dy = self.Joints3D[position+self.n3dElements*4+1] - self.Joints3D[position+self.n3dElements*1+1]
                dz = self.Joints3D[position+self.n3dElements*4+2] - self.Joints3D[position+self.n3dElements*1+2]
                self.ShoulderLineL = np.append(self.ShoulderLineL, self.Joints3D[position+self.n3dElements*4]   + dx * self.ShoulderExtent)
                self.ShoulderLineL = np.append(self.ShoulderLineL, self.Joints3D[position+self.n3dElements*4+1] + dy * self.ShoulderExtent)
                self.ShoulderLineL = np.append(self.ShoulderLineL, self.Joints3D[position+self.n3dElements*4+2] + dz * self.ShoulderExtent)
                self.ShoulderLineR = np.append(self.ShoulderLineR, self.Joints3D[position+self.n3dElements*1]   - dx * self.ShoulderExtent)
                self.ShoulderLineR = np.append(self.ShoulderLineR, self.Joints3D[position+self.n3dElements*1+1] - dy * self.ShoulderExtent)
                self.ShoulderLineR = np.append(self.ShoulderLineR, self.Joints3D[position+self.n3dElements*1+2] - dz * self.ShoulderExtent)
                dx = self.Joints3D[position+self.n3dElements*11]   - self.Joints3D[position+self.n3dElements*8]
                dy = self.Joints3D[position+self.n3dElements*11+1] - self.Joints3D[position+self.n3dElements*8+1]
                dz = self.Joints3D[position+self.n3dElements*11+2] - self.Joints3D[position+self.n3dElements*8+2]
                self.PelvisLineL = np.append(self.PelvisLineL, self.Joints3D[position+self.n3dElements*11]   + dx * self.PelvisExtent)
                self.PelvisLineL = np.append(self.PelvisLineL, self.Joints3D[position+self.n3dElements*11+1] + dy * self.PelvisExtent)
                self.PelvisLineL = np.append(self.PelvisLineL, self.Joints3D[position+self.n3dElements*11+2] + dz * self.PelvisExtent)
                self.PelvisLineR = np.append(self.PelvisLineR, self.Joints3D[position+self.n3dElements*8]    - dx * self.PelvisExtent)
                self.PelvisLineR = np.append(self.PelvisLineR, self.Joints3D[position+self.n3dElements*8+1]  - dy * self.PelvisExtent)
                self.PelvisLineR = np.append(self.PelvisLineR, self.Joints3D[position+self.n3dElements*8+2]  - dz * self.PelvisExtent)
                dx = self.Joints3D[position+self.n3dElements*0]   - self.Joints3D[position+self.n3dElements*7]
                dy = self.Joints3D[position+self.n3dElements*0+1] - self.Joints3D[position+self.n3dElements*7+1]
                dz = self.Joints3D[position+self.n3dElements*0+2] - self.Joints3D[position+self.n3dElements*7+2]
                self.SpineHigh = np.append(self.SpineHigh, self.Joints3D[position+self.n3dElements*0]   + dx * self.SpineExtent)
                self.SpineHigh = np.append(self.SpineHigh, self.Joints3D[position+self.n3dElements*0+1] + dy * self.SpineExtent)
                self.SpineHigh = np.append(self.SpineHigh, self.Joints3D[position+self.n3dElements*0+2] + dz * self.SpineExtent)
                self.SpineLow = np.append(self.SpineLow, self.Joints3D[position+self.n3dElements*7]   - dx * self.SpineExtent)
                self.SpineLow = np.append(self.SpineLow, self.Joints3D[position+self.n3dElements*7+1] - dy * self.SpineExtent)
                self.SpineLow = np.append(self.SpineLow, self.Joints3D[position+self.n3dElements*7+2] - dz * self.SpineExtent)
                dx = self.Joints3D[position+self.n3dElements*12]   - self.Joints3D[position+self.n3dElements*9]
                dy = self.Joints3D[position+self.n3dElements*12+1] - self.Joints3D[position+self.n3dElements*9+1]
                dz = self.Joints3D[position+self.n3dElements*12+2] - self.Joints3D[position+self.n3dElements*9+2]
                self.KneeLineL = np.append(self.KneeLineL, self.Joints3D[position+self.n3dElements*12]   + dx * self.KneeExtent)
                self.KneeLineL = np.append(self.KneeLineL, self.Joints3D[position+self.n3dElements*12+1] + dy * self.KneeExtent)
                self.KneeLineL = np.append(self.KneeLineL, self.Joints3D[position+self.n3dElements*12+2] + dz * self.KneeExtent)
                self.KneeLineR = np.append(self.KneeLineR, self.Joints3D[position+self.n3dElements*9]    - dx * self.KneeExtent)
                self.KneeLineR = np.append(self.KneeLineR, self.Joints3D[position+self.n3dElements*9+1]  - dy * self.KneeExtent)
                self.KneeLineR = np.append(self.KneeLineR, self.Joints3D[position+self.n3dElements*9+2]  - dz * self.KneeExtent)
            # self.nFrames += 1
    def RemoveFrame(self, idFrame):

        print('idx= %d, deleted range= [%d:%d]' % (idFrame, idFrame*self.nJoints3D*self.n3dElements, (idFrame+1)*self.nJoints3D*self.n3dElements))

        if(self.nFrames<1):
            return
        NewArray = list(self.Joints3D)
        del NewArray[idFrame*self.nJoints3D*self.n3dElements:(idFrame+1)*self.nJoints3D*self.n3dElements]
        self.Joints3D = np.array(NewArray, dtype=np.float32)
        self.nFrames -= 1
        self.CalculateExtents()


class PoseData():
    fImageLoaded = 0
    fPoseEstimated = 0
    MaxBrightness = 0.5
    Joints = np.zeros((15,6), dtype=np.int64) # [joint index][index of Main, x, y, Index of Sub, x, y]
    ImageCropped = np.zeros((100, 50, 3), np.uint8)
    ImageCopy = np.array(ImageCropped*MaxBrightness, dtype=np.uint8)
    DegX0 = 0
    DegY0 = 0
    DegW = 100
    DegH = 100
    Angles = np.zeros(14*2, dtype=np.float32)
    def PoseEstimation(self):
        if(self.fImageLoaded == 0):
            return
        else:
            candidate0, subset = body_estimation(self.ImageCropped)
            candidate = candidate0.copy()
            self.ImageCopy = np.array(self.ImageCropped*self.MaxBrightness, dtype=np.uint8)
            for i in range(14):
                for n in range(len(subset)):
                    index = int(subset[n][i])
                    if(i<8):
                        self.Joints[i][0] = index
                    else:
                        self.Joints[i+1][0] = index
                    if index == -1:
                        continue
                    x, y = candidate[index][0:2]
                    if(i<8):
                        self.Joints[i][1] = x
                        self.Joints[i][2] = y
                    else:
                        self.Joints[i+1][1] = x
                        self.Joints[i+1][2] = y
                    if(i == 0):
                        # Face Filled Circle
                        cv2.circle(self.ImageCopy, (int(x), int(y)), 10, (8,8,8), thickness=-1)
                    else:
                        # Joints Dot + Circle
                        cv2.circle(self.ImageCopy, (int(x), int(y)), 1, (255,255,255), thickness=-1)
                        cv2.circle(self.ImageCopy, (int(x), int(y)), 4, (255,255,255), thickness=1)
            #Right Shoulder - arm
            for i in range(1, 4, 1):
                if((self.Joints[i][0] >-1) and (self.Joints[i+1][0] >-1)):
                    cv2.line(self.ImageCopy, (self.Joints[i][1], self.Joints[i][2]), (self.Joints[i+1][1], self.Joints[i+1][2]), (64, 64, 255), thickness=1)
            #Left Shoulder - arm
            if((self.Joints[1][0] >-1) and (self.Joints[5][0] >-1)):
                cv2.line(self.ImageCopy, (self.Joints[1][1], self.Joints[1][2]), (self.Joints[5][1], self.Joints[5][2]), (255, 64, 64), thickness=1)
            for i in range(5, 7, 1):
                if((self.Joints[i][0] >-1) and (self.Joints[i+1][0] >-1)):
                    cv2.line(self.ImageCopy, (self.Joints[i][1], self.Joints[i][2]), (self.Joints[i+1][1], self.Joints[i+1][2]), (255, 64, 64), thickness=1)
            #pelvis coordinate
            #index = 8 for pelvis point
            if((self.Joints[9][0] >-1) and (self.Joints[12][0] >-1)):
                self.Joints[8][1] = int((self.Joints[9][1]+self.Joints[12][1])/2.0)
                self.Joints[8][2] = int((self.Joints[9][2]+self.Joints[12][2])/2.0)
                cv2.circle(self.ImageCopy, (self.Joints[8][1], self.Joints[8][2]), 3, (255,255,255), thickness=-1)
                cv2.line(self.ImageCopy, (self.Joints[9][1], self.Joints[9][2]), (self.Joints[12][1], self.Joints[12][2]), (255,255,255), thickness=1)
                #Body center
                if(self.Joints[1][0] >-1):
                    cv2.line(self.ImageCopy, (self.Joints[1][1], self.Joints[1][2]), (self.Joints[8][1], self.Joints[8][2]), (64,255,64), thickness=1)
            #Right leg - ankle
            for i in range(9, 11, 1):
                if((self.Joints[i][0] >-1) and (self.Joints[i+1][0] >-1)):
                    cv2.line(self.ImageCopy, (self.Joints[i][1], self.Joints[i][2]), (self.Joints[i+1][1], self.Joints[i+1][2]), (64, 64, 255), thickness=1)
            #Left leg - ankle
            for i in range(12, 14, 1):
                if((self.Joints[i][0] >-1) and (self.Joints[i+1][0] >-1)):
                    cv2.line(self.ImageCopy, (self.Joints[i][1], self.Joints[i][2]), (self.Joints[i+1][1], self.Joints[i+1][2]), (255, 64, 64), thickness=1)
            self.fPoseEstimated = 1
    def Joints2Angles(self):
        #For Angles
        h, w, c = self.ImageCopy.shape
        for i in range (14):
            self.Angles[i*2+0] = float( self.DegX0 - self.DegW / w * self.Joints[i+1][1] )
            self.Angles[i*2+1] = float( self.DegY0 - self.DegH / h * self.Joints[i+1][2] )

class VideoData():
    fFileLoaded = 0
    FileName =""
    ImageRead = np.zeros((100, 50, 3), np.uint8)
    ResizeW = 600
    ResizeH = 400
    ImageResized = np.zeros((ResizeW, ResizeH, 3), np.uint8)
    CurrentFrameNumber = 0
    MaxFrameNumber = 0
    FPS = float(30000/1001)
    fCrop = 0
    CropX0= 0
    CropY0= 0
    CropX1= 0
    CropY1= 0
    fCalibrated = 0
    def InitVideo(self):
        if(self.cap.isOpened()==True):
            self.cap.release()
        self.ImageRead = np.zeros((100, 50, 3), np.uint8)
        self.ImageResized = np.zeros((self.ResizeW, self.ResizeH, 3), np.uint8)
        self.fFileLoaded = 0
    def OpenVideo(self):
        self.cap = cv2.VideoCapture(self.FileName)
        if(self.cap.isOpened()==True):
            self.fFileLoaded=1
            ret, self.ImageRead = self.cap.read()
            self.MaxFrameNumber = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
            self.CurrentFrameNumber = 0
            self.Resize()
            self.CropX0 = int(1/4* self.ResizeW / 2)*2
            self.CropX1 = int(3/4* self.ResizeW / 2)*2
            self.CropY0 = int(1/4* self.ResizeH / 2)*2
            self.CropY1 = int(3/4* self.ResizeH / 2)*2
            self.fFileLoaded = 1
    def ReadFrame(self, FrameNumber):
        if(self.cap.isOpened()==1):
            self.CurrentFrameNumber = FrameNumber
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.CurrentFrameNumber)
            ret, self.ImageRead = self.cap.read()
            self.Resize()
    def AnalyzeVideo(self):
        self.MaxFrameNumber = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
    def Resize(self):
        h, w, c = self.ImageRead.shape
        ResizeRatio = self.ResizeW/w
        self.ResizeH = int(h*ResizeRatio)
        self.ImageResized = cv2.resize(self.ImageRead,(self.ResizeW,self.ResizeH))
    def MaskCrop(self):
        self.Resize()
        self.CropX1 = int(self.CropX0 + int((self.CropX1 - self.CropX0)/2)*2)
        self.CropY1 = int(self.CropY0 + int((self.CropY1 - self.CropY0)/2)*2)
        self.ImageMask = np.array(self.ImageResized, dtype=np.uint8)
        self.ImageMask = cv2.rectangle(self.ImageMask, (0, 0), (self.ResizeW, self.ResizeH), (64,64,64), thickness=-1)
        self.ImageMask = cv2.rectangle(self.ImageMask, (self.CropX0, self.CropY0), (self.CropX1, self.CropY1), (255,255,255), thickness=-1)
        self.FloatMask = np.array(self.ImageMask/255, dtype=np.float32)
        self.FloatCropped = np.array(self.ImageResized, dtype=np.float32)
        self.ImageResized = np.array(self.FloatCropped*self.FloatMask, dtype=np.uint8)


class ThreadSave(QThread):
    # _signal = pyqtSignal(int)

    def __init__(self):
        super(ThreadSave, self).__init__()

    def run(self):
        # basename = os.path.splitext(os.path.basename(PData.ProjectFileName))[0]
        basename = PData.ProjectFileName.replace(".prj", "")
        pathname = PData.DataFolderName
        pathTemp = pathname + '/temp/'
        os.makedirs(pathTemp, exist_ok=True)

        # FileNameHeader = QFileDialog().getSaveFileName(self, 'Save File Name', '~\\Videos', "All Files (*)")
        if(MainVid.fFileLoaded==1):
            LimMinValue = ex.LimMinEntry.text()
            LimMaxValue = ex.LimMaxEntry.text()
            if(ex.VTrimCheck.isChecked()==True):
                ex.statusbar.showMessage("Processing Videos Trimming", 1000)
                SaveFileName = basename + "_Main_trim.mp4"
                # SaveFileName = FileNameHeader[0] + "_Main_trim.mp4"
                StartSec = MainVid.CurrentFrameNumber/MainVid.FPS - float(LimMinValue)
                LengthSec = float(LimMinValue)+float(LimMaxValue)
                CmdStr = 'ffmpeg -ss %f -i %s -c:v libx264 -c:a copy -t %f -y %s' % (StartSec, MainVid.FileName, LengthSec, SaveFileName)
                result = subprocess.run(CmdStr, shell=True)
                print('Subprocess: %s' % result)
                
                if(ex.CropCheck.isChecked()==True):
                    ex.statusbar.showMessage("Processing Videos Cropping", 1000)
                    SaveFileName = basename + "_Main_trim_crop.mp4"
                    h,w,c =MainVid.ImageRead.shape
                    Scale = w/MainVid.ResizeW
                    CropX = int(MainVid.CropX0 * Scale)
                    CropY = int(MainVid.CropY0 * Scale)
                    CropW = int((MainVid.CropX1 * Scale - CropX)/2)*2
                    CropH = int((MainVid.CropY1 * Scale - CropY)/2)*2
                    CropStr = '-vf crop=w=%d:h=%d:x=%d:y=%d' % (CropW, CropH, CropX, CropY)
                    CmdStr = 'ffmpeg -ss %f -i %s -c:v libx264 %s -c:a copy -t %f -y %s' % (StartSec, MainVid.FileName, CropStr, LengthSec, SaveFileName)
                    result = subprocess.run(CmdStr, shell=True)
                    print('Subprocess: %s' % result)
                
                if(SubVid.fFileLoaded == 1):
                    ex.statusbar.showMessage("Processing Sub Videos Trimming", 1000)
                    SaveFileName = basename + "_Sub_trim.mp4"
                    LimMinValue = ex.LimMinEntry.text()
                    StartSec = SubVid.CurrentFrameNumber/SubVid.FPS - float(LimMinValue)
                    LimMaxValue = ex.LimMaxEntry.text()
                    LengthSec = float(LimMinValue)+float(LimMaxValue)
                    CmdStr = 'ffmpeg -ss %f -i %s -c:v libx264 -c:a copy -t %f -y %s' % (StartSec, SubVid.FileName, LengthSec, SaveFileName)
                    result = subprocess.run(CmdStr, shell=True)
                    print('Subprocess: %s' % result)
                    if(ex.CropCheckSub.isChecked()==True):
                        ex.statusbar.showMessage("Processing Sub Videos Cropping", 1000)
                        SaveFileName = basename + "_Sub_trim_crop.mp4"
                        h,w,c =SubVid.ImageRead.shape
                        Scale = w/MainVid.ResizeW
                        CropX = int(SubVid.CropX0 * Scale)
                        CropY = int(SubVid.CropY0 * Scale)
                        CropW = int((SubVid.CropX1 * Scale - CropX)/2)*2
                        CropH = int((SubVid.CropY1 * Scale - CropY)/2)*2
                        CropStr = '-vf crop=w=%d:h=%d:x=%d:y=%d' % (CropW, CropH, CropX, CropY)
                        CmdStr = 'ffmpeg -ss %f -i %s -c:v libx264 %s -c:a copy -t %f -y %s' % (StartSec, SubVid.FileName, CropStr, LengthSec, SaveFileName)
                        result = subprocess.run(CmdStr, shell=True)
                        print('Subprocess: %s' % result)

            if(ex.PoseCheck.isChecked()==True):
                if(ex.PoseCheckSub.isChecked()==True):
                    StartFrame = SubVid.CurrentFrameNumber - int(float(LimMinValue)*SubVid.FPS)
                    SubVid.CurrentFrameNumber = StartFrame
                    SubVid.ReadFrame(SubVid.CurrentFrameNumber)
                    SubVid.Resize()
                StartFrame = MainVid.CurrentFrameNumber - int(float(LimMinValue)*MainVid.FPS)
                EndFrame = MainVid.CurrentFrameNumber + int(float(LimMaxValue)*MainVid.FPS)
                TotalFrames = EndFrame - StartFrame
                MainVid.CurrentFrameNumber = StartFrame
                MainVid.ReadFrame(MainVid.CurrentFrameNumber)
                MainVid.Resize()
                idxFrame = 0
                if(ex.Dim3Check.isChecked()==True):
                    SData.fCSVLoaded =0
                while(MainVid.CurrentFrameNumber < EndFrame):
                    ex.statusbar.showMessage('Processing Pose Frame %d/%d' % (idxFrame, TotalFrames) , 1000)
                    ex.PoseMain()
                    # pathname = os.path.dirname(MainVid.FileName)
                    SaveFrameName = pathTemp + 'Main_pose_%03d.jpg' % idxFrame
                    cv2.imwrite(SaveFrameName, MainPose.ImageCopy)
                    MainVid.CurrentFrameNumber += 1
                    MainVid.ReadFrame(MainVid.CurrentFrameNumber)
                    MainVid.Resize()
                    if(ex.PoseCheckSub.isChecked()==True):
                        ex.PoseSub()
                        # pathname = os.path.dirname(MainVid.FileName)
                        SaveFrameName = pathTemp + 'Sub_pose_%03d.jpg' % idxFrame
                        cv2.imwrite(SaveFrameName, SubPose.ImageCopy)
                        SubVid.CurrentFrameNumber += 1
                        SubVid.ReadFrame(SubVid.CurrentFrameNumber)
                        SubVid.Resize()
                        if(ex.Dim3Check.isChecked()==True):
                            ex.Model3D()
                    idxFrame += 1
                ex.statusbar.showMessage("Processing Pose Frame Videos", 1000)
                SaveFileName = basename + "_Main_Pose.mp4"
                NumberedFileName = pathTemp + 'Main_pose_%03d.jpg'
                CmdStr = 'ffmpeg -framerate %f -start_number 000 -i %s -c:v libx264 -r %f -pix_fmt yuv420p -y %s' % (MainVid.FPS, NumberedFileName, MainVid.FPS, SaveFileName)
                result = subprocess.run(CmdStr, shell=True)
                print(result)
                if(ex.PoseCheckSub.isChecked()==True):
                    SaveFileName = basename + "_Sub_Pose.mp4"
                    NumberedFileName = pathTemp + 'Sub_pose_%03d.jpg'
                    CmdStr = 'ffmpeg -framerate %f -start_number 000 -i %s -c:v libx264 -r %f -pix_fmt yuv420p -y %s' % (SubVid.FPS, NumberedFileName, SubVid.FPS, SaveFileName)
                    result = subprocess.run(CmdStr, shell=True)
                    print(result)
                    if(ex.Dim3Check.isChecked()==True):
                        SaveFileName = basename + "_joints.csv"
                        Array = np.array(SData.Joints3D)
                        SaveArray = Array.reshape(SData.nFrames*SData.nJoints3D, SData.n3dElements)
                        np.savetxt(SaveFileName , SaveArray, delimiter=',')
                shutil.rmtree(pathTemp)
                os.makedirs(pathTemp, exist_ok=True)
                if(ex.Dim3Check.isChecked()==True):
                    SData.CalculateExtents()
                ex.statusbar.showMessage("Eport Process Completed", 5000)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 600, 400)
        self.setFixedHeight(600)
        self.setWindowTitle('Swing 3D Ploting')
        self._createActions()
        self._createMenuBar()
        self._createStatusBar()
        self._connectActions()
        self.initCentral()
    #For Menu Bar
    def _createMenuBar(self):
        menuBar = self.menuBar()
        # Creating menus using a QMenu object
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        fileMenu.addAction(self.newAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.openProjAction)
        fileMenu.addAction(self.saveProjAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.openVidMAction)
        fileMenu.addAction(self.openVidSAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.openCsvAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exportAction)
        fileMenu.addSeparator()
        fileMenu.addAction(self.exitAction)
        # Creating menus using a title
        editMenu = menuBar.addMenu("&Edit")
        editMenu.addAction(self.copyAction)
        editMenu.addAction(self.pasteAction)
        editMenu.addAction(self.cutAction)
        helpMenu = menuBar.addMenu("&Help")
        helpMenu.addAction(self.helpContentAction)
        helpMenu.addAction(self.aboutAction)
    def _createActions(self):
        # Creating action using the first constructor
        self.newAction = QAction(self)
        self.newAction.setText("&New Project")
        # Creating actions using the second constructor
        self.openProjAction = QAction("&Open Project", self)
        self.saveProjAction = QAction("&Save Project", self)
        self.openVidMAction = QAction("&Load Main video", self)
        self.openVidSAction = QAction("&Load Sub video", self)
        self.openCsvAction = QAction("&Load 3D data", self)
        self.exportAction = QAction("&Export", self)
        self.exitAction = QAction("&Exit", self)
        self.copyAction = QAction("&Copy", self)
        self.pasteAction = QAction("&Paste", self)
        self.cutAction = QAction("C&ut", self)
        self.helpContentAction = QAction("&Help Content", self)
        self.aboutAction = QAction("&About", self)
    def _createStatusBar(self):
        self.statusbar = self.statusBar()
    def _connectActions(self):
        # Connect File actions
        self.openProjAction.triggered.connect(self.openProject)
        self.saveProjAction.triggered.connect(self.saveProject)
        self.openVidMAction.triggered.connect(self.openVidFile)
        self.openVidSAction.triggered.connect(self.openSubVidFile)
        self.openCsvAction.triggered.connect(self.openCSVFile)
        self.exportAction.triggered.connect(self.exportFile)

    #For Central Widget
    def initCentral(self):
        self.tabView = QTabWidget()
        self.tabView.currentChanged.connect(self.TabChange)
        self.tabVideo = QWidget()
        self.tabCalib = QWidget()
        self.tabGLPlot = QWidget()
        self.tabCSV = QWidget()
        self.tabView.addTab(self.tabVideo,   'Videos')
        self.tabView.addTab(self.tabCalib,   'Camera Calibration')
        self.tabView.addTab(self.tabGLPlot,  '3D Plot')
        self.tabView.addTab(self.tabCSV,     '3D Data')
        self.InitTabVideo()
        self.InitTabGLPlot()
        self.InitTabCalib()
        self.InitTabCSV()
        self.setCentralWidget(self.tabView)
        self.show()

    def openProject(self):
        filter = "Project files (*.prj)"
        PData.ProjectFileName =QFileDialog.getOpenFileName(self, 'Select Project file', '~\\Videos', filter)
        if PData.ProjectFileName[0]:
            fileContents = open(PData.ProjectFileName[0], 'r')
            lines = list(fileContents)
            #Load Data
            PData.ProjectFileName = lines[0].replace("\n", "")
            PData.MainVideoFileName = lines[1].replace("\n", "")
            PData.SubVideoFileName = lines[2].replace("\n", "")
            PData.DataFolderName = lines[3].replace("\n", "")
            CSVFile = lines[0].replace(".prj\n", "_joints.csv")
            if(os.path.exists(CSVFile) == True):
                self.LoadJoints3D(CSVFile)
            ValueStr= lines[4].replace("\n", "")
            PData.MainCameraFovW=int(ValueStr)
            ValueStr= lines[5].replace("\n", "")
            PData.MainCameraFovH=int(ValueStr)
            ValueStr= lines[6].replace("\n", "")
            PData.MainCameraDist=float(ValueStr)
            ValueStr= lines[7].replace("\n", "")
            PData.MainCameraX=float(ValueStr)
            ValueStr= lines[8].replace("\n", "")
            PData.MainCameraY=float(ValueStr)
            ValueStr= lines[9].replace("\n", "")
            PData.MainCameraZ=float(ValueStr)
            ValueStr= lines[10].replace("\n", "")
            PData.MainCameraCenterAngle=float(ValueStr)
            ValueStr= lines[15].replace("\n", "")
            PData.MainVideoFramePosition =int(ValueStr)
            ValueStr= lines[16].replace("\n", "")
            PData.SubCameraFovW=int(ValueStr)
            ValueStr= lines[17].replace("\n", "")
            PData.SubCameraFovH=int(ValueStr)
            ValueStr= lines[18].replace("\n", "")
            PData.SubCameraDist=float(ValueStr)
            ValueStr= lines[19].replace("\n", "")
            PData.SubCameraX=float(ValueStr)
            ValueStr= lines[20].replace("\n", "")
            PData.SubCameraY=float(ValueStr)
            ValueStr= lines[21].replace("\n", "")
            PData.SubCameraZ=float(ValueStr)
            ValueStr= lines[22].replace("\n", "")
            PData.SubCameraCenterAngle = float(ValueStr)
            ValueStr= lines[24].replace("\n", "")
            PData.SubVideoFrameDelta = int(ValueStr)
            fileContents.close()

            MainVid.FileName = PData.MainVideoFileName
            MainVid.OpenVideo()
            self.UpdateImageLabel()
            if(MainVid.cap.isOpened()==True):
                MainVid.CurrentFrameNumber = int(PData.MainVideoFramePosition)
                self.FrameSlider.setRange(0, int(MainVid.MaxFrameNumber))
                self.FrameSlider.setValue(MainVid.CurrentFrameNumber)
                MainVid.ReadFrame(MainVid.CurrentFrameNumber)
                self.UpdateImageLabel()
            if(len(PData.SubVideoFileName)>1):
                SubVid.FileName = PData.SubVideoFileName
                SubVid.OpenVideo()
                self.UpdateImageLabelSub()
                if(SubVid.cap.isOpened()==True):
                    self.Delta = PData.SubVideoFrameDelta
                    self.DeltaEntry.setText(str(self.Delta))
                    SubVid.CurrentFrameNumber = PData.MainVideoFramePosition + self.Delta
                    self.DeltaSlider.setValue(self.Delta)
                    SubVid.ReadFrame(SubVid.CurrentFrameNumber)
                    self.UpdateImageLabelSub()

            self.DistEntry.setText('%2.1f' % float(PData.MainCameraDist))
            self.FOVEntry.setText(str(PData.MainCameraFovW))
            self.PxEntry.setText('%2.1f' % float(PData.MainCameraX))
            self.PyEntry.setText('%2.1f' % float(PData.MainCameraY))
            self.PzEntry.setText('%2.1f' % float(PData.MainCameraZ))
            self.CenterEntry.setText('%2.1f' % float(PData.MainCameraCenterAngle))
            self.Center=float(PData.MainCameraCenterAngle)
            ValueStr= lines[11]
            self.TargetEntry.setText(str(ValueStr))
            self.TargetPixel = int(ValueStr)
            self.TargetSlider.setValue(self.TargetPixel)
            ValueStr= lines[12]
            self.HLEntry.setText(str(ValueStr))
            self.HLPixel = int(ValueStr)
            self.HLSlider.setValue(self.HLPixel)
            ValueStr= lines[13]
            self.LimMinEntry.setText('%2.1f' % float(str(ValueStr)))
            ValueStr= lines[14]
            self.LimMaxEntry.setText('%2.1f' % float(str(ValueStr)))

            self.DistEntryS.setText('%2.1f' % float(PData.SubCameraDist))
            self.FOVEntryS.setText(str(PData.SubCameraFovW))
            self.PxEntryS.setText('%2.1f' % float(PData.SubCameraX))
            self.PyEntryS.setText('%2.1f' % float(PData.SubCameraY))
            self.PzEntryS.setText('%2.1f' % float(PData.SubCameraZ))
            self.CenterEntryS.setText('%2.1f' % float(PData.SubCameraCenterAngle))
            self.CenterS=float(PData.SubCameraCenterAngle)
            ValueStr= lines[23]
            self.TargetEntryS.setText(str(ValueStr))
            self.TargetPixelS = int(ValueStr)
            self.TargetSliderS.setValue(self.TargetPixelS)
            
 
    def saveProject(self):
        filter = "Project files (*.prj)"
        OpenFile = QFileDialog().getSaveFileName(self, 'Save Project', '~\\Videos', filter)
        fileContents = open(OpenFile[0], 'w')
        #Save Data
        PData.ProjectFileName = OpenFile[0]
        fileContents.write(PData.ProjectFileName+'\n')
        PData.MainVideoFileName=MainVid.FileName
        fileContents.write(PData.MainVideoFileName+'\n')
        PData.SubVideoFileName=SubVid.FileName
        fileContents.write(PData.SubVideoFileName+'\n')
        PData.DataFolderName = os.path.dirname(PData.ProjectFileName)
        fileContents.write(PData.DataFolderName+'\n')
        fileContents.write('%d\n' % PData.MainCameraFovW)
        fileContents.write('%d\n' % PData.MainCameraFovH)
        fileContents.write('%f\n' % PData.MainCameraDist)
        fileContents.write('%f\n' % PData.MainCameraX)
        fileContents.write('%f\n' % PData.MainCameraY)
        fileContents.write('%f\n' % PData.MainCameraZ)
        PData.MainCameraCenterAngle = self.Center
        fileContents.write('%f\n' % PData.MainCameraCenterAngle)
        fileContents.write('%d\n' % self.TargetPixel)
        fileContents.write('%d\n' % self.HLPixel)
        fileContents.write('%f\n' % float(self.LimMinEntry.text()))
        fileContents.write('%f\n' % float(self.LimMaxEntry.text()))
        PData.MainVideoFramePosition = MainVid.CurrentFrameNumber
        fileContents.write('%d\n' % PData.MainVideoFramePosition) 
        fileContents.write('%d\n' % PData.SubCameraFovW)
        fileContents.write('%d\n' % PData.SubCameraFovH)
        fileContents.write('%f\n' % PData.SubCameraDist)
        fileContents.write('%f\n' % PData.SubCameraX)
        fileContents.write('%f\n' % PData.SubCameraY)
        fileContents.write('%f\n' % PData.SubCameraZ)
        PData.SubCameraCenterAngle = self.CenterS
        fileContents.write('%f\n' % PData.SubCameraCenterAngle)
        PData.SubVideoFrameDelta = self.Delta
        fileContents.write('%d\n' % self.TargetPixelS)
        fileContents.write('%d\n' % PData.SubVideoFrameDelta)
        fileContents.close()

#For TabVideo ==============================================================
    def InitTabVideo(self):
        self.hBoxVideo = QHBoxLayout(self.tabVideo)
        # Main Video Widget
        self.VidMain = QWidget()
        self.vBoxMain = QVBoxLayout()
        self.TitleBar = QWidget()
        self.TitleHBox = QHBoxLayout()
        self.MainTitle = QLabel("Main")
        self.TitleHBox.addWidget(self.MainTitle)
        self.TitleBar.setLayout(self.TitleHBox)
        self.CropCheck = QCheckBox("Body Area Crop")
        self.TitleHBox.addWidget(self.CropCheck)
        self.CropCheck.stateChanged.connect(self.CropVideo)
        self.PoseCheck = QCheckBox("Pose Estimation (Openpose module)")
        self.PoseCheck.stateChanged.connect(self.PoseMain)
        self.TitleHBox.addWidget(self.PoseCheck)
        self.BGLevel = QLabel("BG Leveel")
        self.BGLevel.setMaximumHeight(10)
        self.BGLevel.setAlignment(Qt.AlignRight)
        self.TitleHBox.addWidget(self.BGLevel)
        self.BGLSlider = QSlider(Qt.Horizontal)
        self.BGLSlider.setValue(5)
        self.BGLSlider.setPageStep(1)
        self.BGLSlider.setRange(0, 10)
        self.BGLSlider.setMaximumWidth(50)
        self.BGLSlider.valueChanged.connect(self.ChangeBGL)
        self.TitleHBox.addWidget(self.BGLSlider)
        self.VTrimCheck = QCheckBox("Video Trim")
        self.TitleHBox.addWidget(self.VTrimCheck)
        self.vBoxMain.addWidget(self.TitleBar)
        self.ImageLabel = QLabel()
        ImageRGB = cv2.cvtColor(MainVid.ImageResized, cv2.COLOR_BGR2RGB)
        qimg = QImage(ImageRGB.flatten(), MainVid.ResizeW, MainVid.ResizeH, QImage.Format_RGB888)
        self.ImageLabel.setPixmap(QPixmap.fromImage(qimg))
        self.ImageLabel.mousePressEvent = self.MouseClick
        self.ImageLabel.mouseReleaseEvent = self.MouseRelease
        self.ImageLabel.mouseMoveEvent = self.MouseMove
        self.ImageLabel.setAlignment(Qt.AlignCenter)
        self.vBoxMain.addWidget(self.ImageLabel)
        # Frame Seek Tool Buttons
        self.SeekBar = QWidget()
        self.SeekBox = QHBoxLayout()
        self.LimMinLabel = QLabel("Limit Min.[s]")
        self.LimMinLabel.setFixedWidth(50)
        self.SeekBox.addWidget(self.LimMinLabel)
        self.LimMinEntry = QLineEdit('1.2')
        self.LimMinEntry.setFixedWidth(30)
        self.SeekBox.addWidget(self.LimMinEntry)
        self.BackButton = QPushButton("<")
        self.BackButton.clicked.connect(self.BackVidFrame)
        self.BackButton.setFixedWidth(30)
        self.SeekBox.addWidget(self.BackButton)
        self.PositionLabel = QLabel("00:00:00")
        self.PositionLabel.setFixedWidth(40)
        self.SeekBox.addWidget(self.PositionLabel)
        self.FrameSlider = QSlider(Qt.Horizontal)
        self.FrameSlider.setMinimumWidth(200)
        self.FrameSlider.setPageStep(30)
        self.FrameSlider.valueChanged.connect(self.SeekVidFrame)
        self.SeekBox.addWidget(self.FrameSlider)
        self.FFButton = QPushButton(">")
        self.FFButton.clicked.connect(self.ForwardVidFrame)
        self.FFButton.setFixedWidth(30)
        self.SeekBox.addWidget(self.FFButton)
        self.LimMaxLabel = QLabel("Limit Max. [s]")
        self.LimMaxLabel.setFixedWidth(50)
        self.SeekBox.addWidget(self.LimMaxLabel)
        self.LimMaxEntry = QLineEdit('0.8')
        self.LimMaxEntry.setFixedWidth(30)
        self.SeekBox.addWidget(self.LimMaxEntry)
        self.SeekBar.setLayout(self.SeekBox)
        self.vBoxMain.addWidget(self.SeekBar)
        self.VidMain.setLayout(self.vBoxMain)
        # End of Main Video Widget
        self.hBoxVideo.addWidget(self.VidMain)

        # Sub Video Widget
        self.VidSub = QWidget()
        self.vBoxSub = QVBoxLayout()
        self.TitleBarSub = QWidget()
        self.TitleHBoxSub = QHBoxLayout()
        self.SubTitle = QLabel("Sub")
        self.TitleHBoxSub.addWidget(self.SubTitle)
        self.TitleBarSub.setLayout(self.TitleHBoxSub)
        self.CropCheckSub = QCheckBox("Body Area Crop")
        self.CropCheckSub.stateChanged.connect(self.CropSubVideo)
        self.TitleHBoxSub.addWidget(self.CropCheckSub)
        self.PoseCheckSub = QCheckBox("Pose Estimation (Openpose module)")
        self.PoseCheckSub.stateChanged.connect(self.PoseSub)
        self.TitleHBoxSub.addWidget(self.PoseCheckSub)
        self.BGLevelSub = QLabel("BG Leveel")
        self.BGLevelSub.setMaximumHeight(10)
        self.BGLevelSub.setAlignment(Qt.AlignRight)
        self.TitleHBoxSub.addWidget(self.BGLevelSub)
        self.BGLSliderSub = QSlider(Qt.Horizontal)
        self.BGLSliderSub.setValue(5)
        self.BGLSliderSub.setPageStep(1)
        self.BGLSliderSub.setRange(0, 10)
        self.BGLSliderSub.setMaximumWidth(50)
        self.BGLSliderSub.valueChanged.connect(self.ChangeBGLSub)
        self.TitleHBoxSub.addWidget(self.BGLSliderSub)
        self.Dim3Check = QCheckBox("3D Data Save")
        self.TitleHBoxSub.addWidget(self.Dim3Check)
        self.vBoxSub.addWidget(self.TitleBarSub)
        self.ImageLabelSub = QLabel()
        ImageRGBSub = cv2.cvtColor(SubVid.ImageResized, cv2.COLOR_BGR2RGB)
        qimg2 = QImage(ImageRGBSub.flatten(), SubVid.ResizeW, SubVid.ResizeH, QImage.Format_RGB888)
        self.ImageLabelSub.setPixmap(QPixmap.fromImage(qimg2))
        self.ImageLabelSub.mousePressEvent = self.MouseClickSub
        self.ImageLabelSub.mouseReleaseEvent = self.MouseReleaseSub
        self.ImageLabelSub.mouseMoveEvent = self.MouseMoveSub
        self.ImageLabelSub.setAlignment(Qt.AlignCenter)
        self.vBoxSub.addWidget(self.ImageLabelSub)
        # Frame Seek Tool Buttons
        self.SeekBarSub = QWidget()
        self.SeekBoxSub = QHBoxLayout()
        self.DeltaLabel = QLabel("Delta Frames")
        self.DeltaLabel.setFixedWidth(70)
        self.SeekBoxSub.addWidget(self.DeltaLabel)
        self.DeltaEntry = QLineEdit('0')
        self.DeltaEntry.setFixedWidth(40)
        self.DeltaEntry.returnPressed.connect(self.DeltaEnter)
        self.SeekBoxSub.addWidget(self.DeltaEntry)
        self.Delta = 0
        self.BackButtonSub = QPushButton("<")
        self.BackButtonSub.clicked.connect(self.BackDelta)
        self.BackButtonSub.setFixedWidth(30)
        self.SeekBoxSub.addWidget(self.BackButtonSub)
        self.DeltaSlider = QSlider(Qt.Horizontal)
        self.DeltaSlider.setMinimumWidth(200)
        self.DeltaSlider.setPageStep(30)
        self.DeltaSlider.setRange(-1000, 1000)
        self.DeltaSlider.valueChanged.connect(self.SeekDelta)
        self.SeekBoxSub.addWidget(self.DeltaSlider)
        self.FFButtonSub = QPushButton(">")
        self.FFButtonSub.clicked.connect(self.ForwardDelta)
        self.FFButtonSub.setFixedWidth(30)
        self.SeekBoxSub.addWidget(self.FFButtonSub)
        self.ModelButton = QPushButton("3D Frame")
        self.ModelButton.clicked.connect(self.Model3D)
        self.SeekBoxSub.addWidget(self.ModelButton)
        self.SeekBarSub.setLayout(self.SeekBoxSub)
        self.vBoxSub.addWidget(self.SeekBarSub)
        self.VidSub.setLayout(self.vBoxSub)
        # End of Sub Video Widget
        self.hBoxVideo.addWidget(self.VidSub)

    def openVidFile(self):
        filter = "mpeg4(*.mp4)"
        VidFileName = QFileDialog.getOpenFileName(self, 'Open Video file', '~\\Videos', filter)
        if VidFileName[0]:
            MainVid.FileName = VidFileName[0]
            MainVid.OpenVideo()
            self.FrameSlider.setRange(0, int(MainVid.MaxFrameNumber))
            self.FrameSlider.setValue(0)
            self.UpdateImageLabel()
            self.LoadCameras()
    def openSubVidFile(self):
        filter = "mpeg4(*.mp4)"
        VidFileName = QFileDialog.getOpenFileName(self, 'Open Video file', '~\\Videos', filter)
        if VidFileName[0]:
            SubVid.FileName = VidFileName[0]
            SubVid.OpenVideo()
            self.DeltaSlider.setValue(0)
            self.UpdateImageLabelSub()
            self.LoadSubCameras()
    def exportFile(self):
        if(MainVid.fFileLoaded == 0):
            msgBox = QMessageBox(QMessageBox.Information, 'Notification', 'Main Video is NOT loaded')
            msgBox.setStandardButtons(QMessageBox.Ok)
            ret = msgBox.exec()
            if(ret == QMessageBox.Ok):
                return
        self.ThSave = ThreadSave()
        self.ThSave.start()


    def UpdateImageLabel(self):
        ImageRGB = cv2.cvtColor(MainVid.ImageResized, cv2.COLOR_BGR2RGB)
        qimg = QImage(ImageRGB, MainVid.ResizeW, MainVid.ResizeH, 3*MainVid.ResizeW, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.ImageLabel.resize(MainVid.ResizeW, MainVid.ResizeH)
        self.ImageLabel.setPixmap(pixmap)
        self.ImageLabel.update()
    def SeekVidFrame(self, value):
        if(MainVid.fFileLoaded == 0):
            return
        MainVid.ReadFrame(int(value))
        if(self.CropCheck.isChecked()==True):
            MainVid.MaskCrop()
            if(self.PoseCheck.isChecked()==True):
                MainPose.fImageLoaded=0
                MainPose.fPoseEstimated=0
                self.PoseMain()
        self.UpdateImageLabel()
        Minutes = int(MainVid.CurrentFrameNumber/MainVid.FPS/60)
        Seconds = int(MainVid.CurrentFrameNumber/MainVid.FPS) - Minutes*60
        SubSec = int((float(MainVid.CurrentFrameNumber/MainVid.FPS) - Minutes*60 - Seconds)*100)
        PositionSec = '%02d:%02d:%02d' % (Minutes, Seconds, SubSec)
        self.PositionLabel.setText(PositionSec)
        if(SubVid.fFileLoaded==1):
            self.SeekDelta(self.Delta)
    def BackVidFrame(self):
        MainVid.CurrentFrameNumber -=1 
        if(MainVid.CurrentFrameNumber < 0):
            MainVid.CurrentFrameNumber = 0
        self.SeekVidFrame(MainVid.CurrentFrameNumber)
    def ForwardVidFrame(self):
        MainVid.CurrentFrameNumber +=1 
        if(MainVid.CurrentFrameNumber > MainVid.MaxFrameNumber):
            MainVid.CurrentFrameNumber = MainVid.MaxFrameNumber
        self.SeekVidFrame(MainVid.CurrentFrameNumber)
    def CropVideo(self):
        if(self.CropCheck.isChecked()==True):
            MainVid.MaskCrop()
            self.UpdateImageLabel()
            if(self.PoseCheck.isChecked()==True):
                self.PoseMain()
        else:
            MainVid.Resize()
            self.UpdateImageLabel()
    def MouseClick(self, event):
        if(self.CropCheck.isChecked()==False):
            return
        else:
            if(event.button() == Qt.LeftButton):
                if(MainVid.fCrop==0):
                    local = event.pos()
                    MainVid.CropX0=local.x()
                    MainVid.CropY0=local.y()
                    MainVid.CropX1=MainVid.CropX0+1
                    MainVid.CropY1=MainVid.CropY0+1
                    MainVid.MaskCrop()
                    self.UpdateImageLabel()
                    MainVid.fCrop=1
    def MouseMove(self, event):
        if(self.CropCheck.isChecked()==False):
            return
        else:
            if(MainVid.fCrop==1):
                local = event.pos()
                MainVid.CropX1=local.x()
                MainVid.CropY1=local.y()
                MainVid.MaskCrop()
                self.UpdateImageLabel()
    def MouseRelease(self, event):
        if(self.CropCheck.isChecked()==False):
            return
        else:
            if(event.button() == Qt.LeftButton):
                if(MainVid.fCrop==1):
                    local = event.pos()
                    MainVid.CropX1=local.x()
                    MainVid.CropY1=local.y()
                    MainVid.MaskCrop()
                    self.UpdateImageLabel()
                    MainVid.fCrop=0
    def PoseMain(self):
        if(self.PoseCheck.isChecked()==1):
            if(self.CropCheck.isChecked()==1):
                MainPose.ImageCropped = MainVid.ImageResized[MainVid.CropY0:MainVid.CropY1, MainVid.CropX0:MainVid.CropX1]
                MainPose.fImageLoaded = 1
                MainPose.PoseEstimation()
                MainVid.ImageResized = cv2.rectangle(MainVid.ImageResized, (0, 0), (MainVid.ResizeW, MainVid.ResizeH), (0,0,0), thickness=-1)
                MainVid.ImageResized[MainVid.CropY0:MainVid.CropY1, MainVid.CropX0:MainVid.CropX1] = MainPose.ImageCopy
                MainPose.DegX0 = SData.pixel2deg(MainVid.CropX0, self.TargetPixel, MainVid.ResizeW, PData.MainCameraFovW)
                MainPose.DegY0 = SData.pixel2deg(MainVid.CropY0, self.HLPixel, MainVid.ResizeH, PData.MainCameraFovH)
                DegX1 = SData.pixel2deg(MainVid.CropX1, self.TargetPixel, MainVid.ResizeW, PData.MainCameraFovW)
                DegY1 = SData.pixel2deg(MainVid.CropY1, self.HLPixel, MainVid.ResizeH, PData.MainCameraFovH)
                MainPose.DegW = MainPose.DegX0 - DegX1
                MainPose.DegH = MainPose.DegY0 - DegY1
                MainPose.Joints2Angles()
                SData.AnglesMain = MainPose.Angles.copy()
        else:
            if(self.CropCheck.isChecked()==1):
                MainVid.MaskCrop()
            else:
                MainVid.Resize()
        self.UpdateImageLabel()
    def ChangeBGL(self, value):
        MainPose.MaxBrightness = float(value/10)
        if(self.PoseCheck.isChecked()==True):
            MainVid.Resize()
            self.PoseMain()

    def UpdateImageLabelSub(self):
        if(self.CropCheckSub.isChecked()==True):
            if(self.PoseCheckSub.isChecked()==True):
                SubVid.ImageResized = cv2.rectangle(SubVid.ImageResized, (0, 0), (SubVid.ResizeW, SubVid.ResizeH), (0,0,0), thickness=-1)
                SubVid.ImageResized[SubVid.CropY0:SubVid.CropY1, SubVid.CropX0:SubVid.CropX1] = SubPose.ImageCopy
            else:
                SubVid.MaskCrop()
        ImageRGBSub = cv2.cvtColor(SubVid.ImageResized, cv2.COLOR_BGR2RGB)
        qimg2 = QImage(ImageRGBSub.flatten(), SubVid.ResizeW, SubVid.ResizeH, 3*SubVid.ResizeW, QImage.Format_RGB888)
        self.ImageLabelSub.setPixmap(QPixmap.fromImage(qimg2))
        self.ImageLabelSub.resize(SubVid.ResizeW, SubVid.ResizeH)
        self.ImageLabelSub.update()
    def DeltaEnter(self):
        self.Delta = int(self.DeltaEntry.text())
        if(self.Delta < self.DeltaSlider.minimum()):
            self.DeltaSlider.setRange(self.DeltaSlider.minimum()-1000, self.DeltaSlider.maximum())
        if(self.Delta > self.DeltaSlider.maximum()):
            self.DeltaSlider.setRange(self.DeltaSlider.minimum(), self.DeltaSlider.maximum()+1000)
        SubVid.CurrentFrameNumber = MainVid.CurrentFrameNumber+int(self.Delta)
        SubVid.ReadFrame(SubVid.CurrentFrameNumber)
        if(self.PoseCheckSub.isChecked()==True):
            SubPose.fImageLoaded=0
            SubPose.fPoseEstimated=0
        self.UpdateImageLabelSub()
        self.DeltaSlider.setValue(self.Delta)
    def BackDelta(self):
        self.Delta -= 1
        if(self.Delta < self.DeltaSlider.minimum()):
            self.DeltaSlider.setRange(self.DeltaSlider.minimum()-1000, self.DeltaSlider.maximum())
        self.DeltaEntry.setText(str(self.Delta))
        self.DeltaSlider.setValue(int(self.Delta))
        self.SeekDelta(self.Delta)
    def ForwardDelta(self):
        self.Delta += 1
        if(self.Delta > self.DeltaSlider.maximum()):
            self.DeltaSlider.setRange(self.DeltaSlider.minimum(), self.DeltaSlider.maximum()+1000)
        self.DeltaEntry.setText(str(self.Delta))
        self.DeltaSlider.setValue(int(self.Delta))
        self.SeekDelta(self.Delta)
    def SeekDelta(self, value):
        self.Delta = int(value)
        self.DeltaEntry.setText(str(self.Delta))
        if(SubVid.fFileLoaded == 0):
            return
        SubVid.CurrentFrameNumber = MainVid.CurrentFrameNumber+int(self.Delta)
        SubVid.ReadFrame(SubVid.CurrentFrameNumber)
        if(self.PoseCheckSub.isChecked()==True):
            SubPose.fImageLoaded=0
            SubPose.fPoseEstimated=0
            self.PoseSub()
        self.UpdateImageLabelSub()
    def CropSubVideo(self):
        if(self.CropCheckSub.isChecked()==True):
            SubVid.MaskCrop()
            self.UpdateImageLabelSub()
        else:
            SubVid.Resize()
            self.UpdateImageLabelSub()
    def MouseClickSub(self, event):
        if(self.CropCheckSub.isChecked()==False):
            return
        else:
            if(event.button() == Qt.LeftButton):
                if(SubVid.fCrop==0):
                    local = event.pos()
                    SubVid.CropX0=local.x()
                    SubVid.CropY0=local.y()
                    SubVid.CropX1=SubVid.CropX0+1
                    SubVid.CropY1=SubVid.CropY0+1
                    SubVid.MaskCrop()
                    self.UpdateImageLabelSub()
                    SubVid.fCrop=1
    def MouseMoveSub(self, event):
        if(self.CropCheckSub.isChecked()==False):
            return
        else:
            if(SubVid.fCrop==1):
                local = event.pos()
                SubVid.CropX1=local.x()
                SubVid.CropY1=local.y()
                SubVid.MaskCrop()
                self.UpdateImageLabelSub()
    def MouseReleaseSub(self, event):        
        if(self.CropCheckSub.isChecked()==False):
            return
        else:
            if(event.button() == Qt.LeftButton):
                if(SubVid.fCrop==1):
                    local = event.pos()
                    SubVid.CropX1=local.x()
                    SubVid.CropY1=local.y()
                    SubVid.MaskCrop()
                    self.UpdateImageLabelSub()
                    SubVid.fCrop=0
    def PoseSub(self):
        if(self.PoseCheckSub.isChecked()==1):
            if(self.CropCheckSub.isChecked()==1):
                SubPose.ImageCropped = SubVid.ImageResized[SubVid.CropY0:SubVid.CropY1, SubVid.CropX0:SubVid.CropX1]
                SubPose.fImageLoaded = 1
                SubPose.PoseEstimation()
                SubVid.ImageResized = cv2.rectangle(SubVid.ImageResized, (0, 0), (SubVid.ResizeW, SubVid.ResizeH), (0,0,0), thickness=-1)
                SubVid.ImageResized[SubVid.CropY0:SubVid.CropY1, SubVid.CropX0:SubVid.CropX1] = SubPose.ImageCopy
                SubPose.DegX0 = SData.pixel2deg(SubVid.CropX0, self.TargetPixelS, SubVid.ResizeW, PData.SubCameraFovW)
                SubPose.DegY0 = SData.pixel2deg(SubVid.CropY0, int(SubVid.ResizeH/2),  SubVid.ResizeH, PData.SubCameraFovH)
                DegX1 = SData.pixel2deg(SubVid.CropX1, self.TargetPixelS, SubVid.ResizeW, PData.SubCameraFovW)
                DegY1 = SData.pixel2deg(SubVid.CropY1, int(SubVid.ResizeH/2), SubVid.ResizeH, PData.SubCameraFovH)
                SubPose.DegW = SubPose.DegX0 - DegX1
                SubPose.DegH = SubPose.DegY0 - DegY1
                SubPose.DegW = SubPose.DegX0 - SData.pixel2deg(SubVid.CropX1, self.TargetPixelS, SubVid.ResizeW, PData.SubCameraFovW)
                SubPose.DegH = SubPose.DegY0 - SData.pixel2deg(SubVid.CropY1, int(SubVid.ResizeH/2), SubVid.ResizeH, PData.SubCameraFovH)
                SubPose.Joints2Angles()
                SData.AnglesSub = SubPose.Angles.copy()
        else:
            if(self.CropCheckSub.isChecked()==1):
                SubVid.MaskCrop()
            else:
                SubVid.Resize()
        self.UpdateImageLabelSub()
    def ChangeBGLSub(self, value):
        SubPose.MaxBrightness = float(value/10)
        if(self.PoseCheck.isChecked()==True):
            SubVid.Resize()
            self.PoseSub()
    def Model3D(self):
        if((MainPose.fPoseEstimated==0)or(SubPose.fPoseEstimated==0)):
            return
        self.table.setRowCount(SData.nJoints3D)
        self.table.setColumnCount(SData.n3dElements)
        
        SData.Ang2Pos()

        for joint in range(SData.nJoints3D):
            self.table.setItem(joint, 0, QTableWidgetItem('%4.2f' % SData.Joints3DSingle[joint*SData.n3dElements+0]))
            self.table.setItem(joint, 1, QTableWidgetItem('%4.2f' % SData.Joints3DSingle[joint*SData.n3dElements+1]))
            self.table.setItem(joint, 2, QTableWidgetItem('%4.2f' % SData.Joints3DSingle[joint*SData.n3dElements+2]))
            self.table.setItem(joint, 3, QTableWidgetItem('%4.2f' % SData.Joints3DSingle[joint*SData.n3dElements+3]))
        if(SData.fCSVLoaded==0):
            SData.Joints3D = SData.Joints3DSingle.copy()
            SData.nFrames = 1
        else:
            SData.Joints3D += SData.Joints3DSingle.copy()
            SData.nFrames += 1
            SData.CalculateExtents()
        SData.fCSVLoaded =1

#For TabCalib ==============================================================
    def InitTabCalib(self):
        self.gBoxCalib = QGridLayout(self.tabCalib)
        self.tabCalib.setLayout(self.gBoxCalib)
        #For Main Camera
        self.TitleCalib = QLabel('Main Camera Settings (For X and Z axes)')
        self.TitleCalib.setAlignment(Qt.AlignCenter)
        self.gBoxCalib.addWidget(self.TitleCalib, 0, 0, 1, 10)
        self.DistLabel = QLabel('Horizontal Distance from ball to Camera [m]')
        self.gBoxCalib.addWidget(self.DistLabel, 1, 1, 1, 3)
        self.DistEntry = QLineEdit('0.1')
        self.DistEntry.setFixedWidth(30)
        self.DistEntry.returnPressed.connect(self.DistEnter)
        self.gBoxCalib.addWidget(self.DistEntry, 1, 4)
        self.FOVLabel = QLabel('Horizontal Field of view of Camera [degrees]')
        self.FOVLabel.setAlignment(Qt.AlignCenter)
        self.gBoxCalib.addWidget(self.FOVLabel, 1, 5, 1, 2)
        self.FOVEntry = QLineEdit('360')
        self.FOVEntry.setFixedWidth(30)
        self.FOVEntry.returnPressed.connect(self.FOVEnter)
        self.gBoxCalib.addWidget(self.FOVEntry, 1, 7)
        self.HLLabel = QLabel('Level\n[pixel]')
        self.HLLabel.setMaximumHeight(25)
        self.HLLabel.setAlignment(Qt.AlignCenter)
        self.gBoxCalib.addWidget(self.HLLabel, 2, 0)
        self.HLEntry = QLineEdit('1000')
        self.HLEntry.setMaximumHeight(15)
        self.HLEntry.setFixedWidth(30)
        self.HLEntry.returnPressed.connect(self.HLEnter)
        self.gBoxCalib.addWidget(self.HLEntry, 3, 0)
        self.HLSlider = QSlider(Qt.Vertical)
        self.HLSlider.setMaximumHeight(300)
        self.HLSlider.setRange(0, MainVid.ResizeH)
        self.HLSlider.setValue(int(MainVid.ResizeH/2))
        self.HLSlider.valueChanged.connect(self.HLChange)
        self.gBoxCalib.addWidget(self.HLSlider, 4, 0, 5, 1)
        self.ImageLabel2 = QLabel()
        ImageRGB = cv2.cvtColor(MainVid.ImageResized, cv2.COLOR_BGR2RGB)
        qimg = QImage(ImageRGB.flatten(), MainVid.ResizeW, MainVid.ResizeH, QImage.Format_RGB888)
        self.ImageLabel2.setPixmap(QPixmap.fromImage(qimg))
        self.ImageLabel2.mousePressEvent = self.BallClick
        self.ImageLabel2.setAlignment(Qt.AlignLeft)
        self.gBoxCalib.addWidget(self.ImageLabel2, 2, 1, 6, 9)
        self.TargetLabel = QLabel('Target Direction [pixel]')
        self.TargetLabel.setAlignment(Qt.AlignRight)
        self.gBoxCalib.addWidget(self.TargetLabel, 9, 1, 1, 1)
        self.TargetEntry = QLineEdit('3000')
        self.TargetEntry.setFixedWidth(30)
        self.TargetEntry.returnPressed.connect(self.TargetEnter)
        self.gBoxCalib.addWidget(self.TargetEntry, 9, 2)
        self.LeftButton = QPushButton("<", self)
        self.LeftButton.setMaximumWidth(30)
        self.LeftButton.clicked.connect(self.TargetLeft)
        self.gBoxCalib.addWidget(self.LeftButton, 9,3)
        self.TargetSlider = QSlider(Qt.Horizontal)
        self.TargetSlider.setMaximumHeight(300)
        self.TargetSlider.setRange(0, int(MainVid.ResizeW/2)*3)
        self.TargetSlider.valueChanged.connect(self.TargetChange)
        self.gBoxCalib.addWidget(self.TargetSlider, 9, 4, 1, 2)
        self.RightButton = QPushButton(">", self)
        self.RightButton.setMaximumWidth(30)
        self.RightButton.clicked.connect(self.TargetRight)
        self.gBoxCalib.addWidget(self.RightButton, 9,6)
        self.CalibButton = QPushButton("Calibrate", self)
        self.CalibButton.clicked.connect(self.CameraCalib)
        self.gBoxCalib.addWidget(self.CalibButton, 9,7)
        self.CPositionLabel = QLabel('Camera Position [m]  ->  [X] [Y] [Z]')
        self.CPositionLabel.setAlignment(Qt.AlignRight)
        self.gBoxCalib.addWidget(self.CPositionLabel, 10, 0, 1, 2)
        self.PxEntry = QLineEdit('0')
        self.PxEntry.setFixedWidth(30)
        self.gBoxCalib.addWidget(self.PxEntry, 10, 2)
        self.PyEntry = QLineEdit('0')
        self.PyEntry.setFixedWidth(30)
        self.gBoxCalib.addWidget(self.PyEntry, 10, 3)
        self.PzEntry = QLineEdit('0')
        self.PzEntry.setFixedWidth(30)
        self.gBoxCalib.addWidget(self.PzEntry, 10, 4)
        self.CenterLabel = QLabel('Angle at Image Center [degrees]')
        self.CenterLabel.setAlignment(Qt.AlignCenter)
        self.gBoxCalib.addWidget(self.CenterLabel, 10, 5, 1, 1)
        self.CenterEntry = QLineEdit('0')
        self.CenterEntry.setFixedWidth(30)
        self.gBoxCalib.addWidget(self.CenterEntry, 10, 6)
        self.UpdateButton = QPushButton("Update", self)
        self.UpdateButton.clicked.connect(self.UpdateCamera)
        self.gBoxCalib.addWidget(self.UpdateButton, 10,7)

        #For Sub Camera
        self.TitleCalibS = QLabel('Sub Camera Settings (For Y axix)')
        self.TitleCalibS.setAlignment(Qt.AlignCenter)
        self.gBoxCalib.addWidget(self.TitleCalibS, 0, 10, 1, 10)
        self.DistLabelS = QLabel('Horiz. Dist. from ball to Camera [m]')
        self.gBoxCalib.addWidget(self.DistLabelS, 1, 11, 1, 3)
        self.DistEntryS = QLineEdit('2.5')
        self.DistEntryS.setFixedWidth(30)
        self.DistEntryS.returnPressed.connect(self.DistEnterSub)
        self.gBoxCalib.addWidget(self.DistEntryS, 1, 14)
        self.FOVLabelS = QLabel('   Horizontal Field of view of Camera [degrees]')
        self.gBoxCalib.addWidget(self.FOVLabelS, 1, 15, 1, 2)
        self.FOVEntryS = QLineEdit('120')
        self.FOVEntryS.setFixedWidth(30)
        self.FOVEntryS.returnPressed.connect(self.FOVEnterSub)
        self.gBoxCalib.addWidget(self.FOVEntryS, 1, 17)
        self.ImageLabel2S = QLabel()
        ImageRGB = cv2.cvtColor(SubVid.ImageResized, cv2.COLOR_BGR2RGB)
        qimg = QImage(ImageRGB.flatten(), SubVid.ResizeW, SubVid.ResizeH, QImage.Format_RGB888)
        self.ImageLabel2S.setPixmap(QPixmap.fromImage(qimg))
        self.ImageLabel2S.setAlignment(Qt.AlignLeft)
        self.ImageLabel2S.mousePressEvent = self.BallClickSub
        self.gBoxCalib.addWidget(self.ImageLabel2S, 2, 10, 6, 8)
        self.TargetLabelS = QLabel('Target Direction [pixel]')
        self.gBoxCalib.addWidget(self.TargetLabelS, 9, 11, 1, 1)
        self.TargetEntryS = QLineEdit('3000')
        self.TargetEntryS.setFixedWidth(30)
        self.TargetEntryS.returnPressed.connect(self.TargetEnterSub)
        self.gBoxCalib.addWidget(self.TargetEntryS, 9, 12)
        self.LeftButtonS = QPushButton("<", self)
        self.LeftButtonS.setMaximumWidth(30)
        self.LeftButtonS.clicked.connect(self.TargetLeftSub)
        self.gBoxCalib.addWidget(self.LeftButtonS, 9,13)
        self.TargetSliderS = QSlider(Qt.Horizontal)
        self.TargetSliderS.setMaximumHeight(300)
        self.TargetSliderS.setRange(0, int(SubVid.ResizeW/2)*3)
        self.TargetSliderS.valueChanged.connect(self.TargetChangeSub)
        self.gBoxCalib.addWidget(self.TargetSliderS, 9, 14, 1, 2)
        self.RightButtonS = QPushButton(">", self)
        self.RightButtonS.setMaximumWidth(30)
        self.RightButtonS.clicked.connect(self.TargetRightSub)
        self.gBoxCalib.addWidget(self.RightButtonS, 9,16)
        self.CalibButtonS = QPushButton("Calibrate", self)
        self.CalibButtonS.clicked.connect(self.CameraCalibSub)
        self.gBoxCalib.addWidget(self.CalibButtonS, 9,17)
        self.CPositionLabelS = QLabel('Camera Position [X][Y][Z]')
        self.CPositionLabelS.setAlignment(Qt.AlignRight)
        self.gBoxCalib.addWidget(self.CPositionLabelS, 10, 10, 1, 2)
        self.PxEntryS = QLineEdit('0')
        self.PxEntryS.setFixedWidth(30)
        self.gBoxCalib.addWidget(self.PxEntryS, 10, 12)
        self.PyEntryS = QLineEdit('0')
        self.PyEntryS.setFixedWidth(30)
        self.gBoxCalib.addWidget(self.PyEntryS, 10, 13)
        self.PzEntryS = QLineEdit('0')
        self.PzEntryS.setFixedWidth(30)
        self.gBoxCalib.addWidget(self.PzEntryS, 10, 14)
        self.CenterLabelS = QLabel('Angle at Image Center [degrees]')
        self.CenterLabelS.setAlignment(Qt.AlignCenter)
        self.gBoxCalib.addWidget(self.CenterLabelS, 10, 15, 1, 1)
        self.CenterEntryS = QLineEdit('120')
        self.CenterEntryS.setFixedWidth(30)
        self.gBoxCalib.addWidget(self.CenterEntryS, 10, 16)
        self.UpdateButtonS = QPushButton("Update", self)
        self.UpdateButtonS.clicked.connect(self.UpdateCameraSub)
        self.gBoxCalib.addWidget(self.UpdateButtonS, 10,17)
            
    #Class Member variables?
    HLPixel = 0
    TargetPixel = 0
    BallAngleX = 0
    BallAngleY = 0
    TargetPixelS = 0
    BallAngleXS = 0
    BallAngleYS = 0
    Center = 0
    CenterS = 0

    def LoadCameras(self):
        self.HLSlider.setRange(0, MainVid.ResizeH)
        self.HLSlider.setValue(int(MainVid.ResizeH/2))
        self.HLEntry.setText(str(int(MainVid.ResizeH/2)))
        self.HLPixel = int(MainVid.ResizeH/2)
        self.TargetEntry.setText(str(int(MainVid.ResizeW/2)))
        self.TargetSlider.setRange(0, 2*MainVid.ResizeW)
        self.TargetSlider.setValue(int(MainVid.ResizeW/2))
        self.TargetPixel = int(MainVid.ResizeW/2)
        self.BallAngleX = int(MainVid.ResizeW*3/4)
        self.BallAngleY = int(MainVid.ResizeH*3/4)
    def LoadSubCameras(self):
        self.TargetEntryS.setText(str(int(SubVid.ResizeW/2)))
        self.TargetSliderS.setRange(0, 2*SubVid.ResizeW)
        self.TargetSliderS.setValue(int(SubVid.ResizeW/2))
        self.TargetPixelS = int(SubVid.ResizeW/2)
        self.BallAngleXS = int(SubVid.ResizeW*3/4)
        self.BallAngleYS = int(SubVid.ResizeH*3/4)
    def UpdateCalibMain(self):
        MainVid.Resize()
        ImageRGB = cv2.cvtColor(MainVid.ImageResized, cv2.COLOR_BGR2RGB)
        for i in range(12):
            n90 = (i-4)* 90
            Pix90 = SData.deg2pixel(n90, self.TargetPixel, MainVid.ResizeW, PData.MainCameraFovW)
            if(i%4 == 1):
                cv2.line(ImageRGB, (Pix90, 0), (Pix90, MainVid.ResizeH), (255,0,0), thickness=1)
            elif(i%4 == 3):
                cv2.line(ImageRGB, (Pix90, 0), (Pix90, MainVid.ResizeH), (0,0,255), thickness=1)
            else:
                cv2.line(ImageRGB, (Pix90, 0), (Pix90, MainVid.ResizeH), (255,255,255), thickness=1)
        cv2.line(ImageRGB, (0, self.HLPixel), (MainVid.ResizeW, self.HLPixel), (255,255,255), thickness=1)
        cv2.circle(ImageRGB, (self.BallAngleX, self.BallAngleY), 8, (255,255,255), thickness=1)
        qimg = QImage(ImageRGB.flatten(), MainVid.ResizeW, MainVid.ResizeH, 3*MainVid.ResizeW, QImage.Format_RGB888)
        self.ImageLabel2.setPixmap(QPixmap.fromImage(qimg))
        self.ImageLabel2.resize(MainVid.ResizeW, MainVid.ResizeH)
        self.ImageLabel2.update()
    def UpdateCalibSub(self):
        SubVid.Resize()
        ImageRGB = cv2.cvtColor(SubVid.ImageResized, cv2.COLOR_BGR2RGB)
        for i in range(12):
            n90 = (i-4)* 90
            Pix90 = SData.deg2pixel(n90, self.TargetPixelS, SubVid.ResizeW, PData.SubCameraFovW)
            if(i%4 == 1):
                cv2.line(ImageRGB, (Pix90, 0), (Pix90, SubVid.ResizeH), (255,0,0), thickness=1)
            elif(i%4 == 3):
                cv2.line(ImageRGB, (Pix90, 0), (Pix90, SubVid.ResizeH), (0,0,255), thickness=1)
            else:
                cv2.line(ImageRGB, (Pix90, 0), (Pix90, SubVid.ResizeH), (255,255,255), thickness=1)
        cv2.circle(ImageRGB, (self.BallAngleXS, self.BallAngleYS), 8, (255,255,255), thickness=1)
        qimg = QImage(ImageRGB.flatten(), SubVid.ResizeW, SubVid.ResizeH, 3*SubVid.ResizeW, QImage.Format_RGB888)
        self.ImageLabel2S.setPixmap(QPixmap.fromImage(qimg))
        self.ImageLabel2S.resize(SubVid.ResizeW, SubVid.ResizeH)
        self.ImageLabel2S.update()
    def TabChange(self, TabId):
        if(TabId == 2):
            self.UpdateCalibMain()
            self.UpdateCalibSub()
    def BallClick(self, event):
        if(MainVid.fFileLoaded==0):
            return
        else:
            if(event.button() == Qt.LeftButton):
                local = event.pos()
                self.BallAngleX = local.x()
                self.BallAngleY = local.y()
                self.UpdateCalibMain()
                self.CameraCalib()
    def BallClickSub(self, event):
        if(SubVid.fFileLoaded==0):
            return
        else:
            if(event.button() == Qt.LeftButton):
                local = event.pos()
                self.BallAngleXS = local.x()
                self.BallAngleYS = local.y()
                self.UpdateCalibSub()
                self.CameraCalibSub()
    def HLChange(self, value):
        if(MainVid.fFileLoaded==0):
            return
        else:
            self.HLPixel = int(MainVid.ResizeH - value)
            self.HLEntry.setText(str(self.HLPixel))
            self.UpdateCalibMain()
    def HLEnter(self):
        if(MainVid.fFileLoaded==0):
            return
        else:
            self.HLPixel = MainVid.ResizeH-int(self.HLEntry.text())
            self.HLSlider.setValue(self.HLPixel)
            self.UpdateCalibMain()
    def TargetChange(self, value):
        if(MainVid.fFileLoaded==0):
            return
        else:
            self.TargetPixel =int(value)
            self.TargetEntry.setText(str(self.TargetPixel))
            self.Center = SData.pixel2deg(int(MainVid.ResizeW/2), self.TargetPixel, MainVid.ResizeW, PData.MainCameraFovW)
            if(self.Center > 720):
                self.Center -= 720
            elif(self.Center > 360):
                self.Center -= 360
            self.CenterEntry.setText('%4.1f' % self.Center)
            self.UpdateCalibMain()
    def TargetLeft(self):
        if(MainVid.fFileLoaded==0):
            return
        else:
            self.TargetPixel -= 1 
            self.TargetEntry.setText(str(self.TargetPixel))
            self.Center = SData.pixel2deg(int(MainVid.ResizeW/2), self.TargetPixel, MainVid.ResizeW, PData.MainCameraFovW)
            if(self.Center > 720):
                self.Center -= 720
            elif(self.Center > 360):
                self.Center -= 360
            self.CenterEntry.setText('%4.1f' % self.Center)
            self.UpdateCalibMain()
    def TargetRight(self):
        if(MainVid.fFileLoaded==0):
            return
        else:
            self.TargetPixel += 1 
            self.TargetEntry.setText(str(self.TargetPixel))
            self.Center = SData.pixel2deg(int(MainVid.ResizeW/2), self.TargetPixel, MainVid.ResizeW, PData.MainCameraFovW)
            if(self.Center > 720):
                self.Center -= 720
            elif(self.Center > 360):
                self.Center -= 360
            self.CenterEntry.setText('%4.1f' % self.Center)
            self.UpdateCalibMain()
    def TargetChangeSub(self, value):
        if(SubVid.fFileLoaded==0):
            return
        else:
            self.TargetPixelS = int(value)
            self.TargetEntryS.setText(str(self.TargetPixelS))
            self.CenterS = SData.pixel2deg(int(SubVid.ResizeW/2), self.TargetPixelS, SubVid.ResizeW, PData.SubCameraFovW)
            if(self.CenterS > 720):
                self.CenterS -= 720
            elif(self.CenterS > 360):
                self.CenterS -= 360
            self.CenterEntryS.setText('%4.1f' % self.CenterS)
            self.UpdateCalibSub()
    def TargetLeftSub(self):
        if(SubVid.fFileLoaded==0):
            return
        else:
            self.TargetPixelS -= 1 
            self.TargetEntryS.setText(str(self.TargetPixelS))
            self.CenterS = SData.pixel2deg(int(SubVid.ResizeW/2), self.TargetPixelS, SubVid.ResizeW, PData.SubCameraFovW)
            if(self.CenterS > 720):
                self.CenterS -= 720
            elif(self.CenterS > 360):
                self.CenterS -= 360
            self.CenterEntryS.setText('%4.1f' % self.CenterS)
            self.UpdateCalibSub()
    def TargetRightSub(self):
        if(SubVid.fFileLoaded==0):
            return
        else:
            self.TargetPixelS += 1 
            self.TargetEntryS.setText(str(self.TargetPixelS))
            self.CenterS = SData.pixel2deg(int(SubVid.ResizeW/2), self.TargetPixelS, SubVid.ResizeW, PData.SubCameraFovW)
            if(self.CenterS > 720):
                self.CenterS -= 720
            elif(self.CenterS > 360):
                self.CenterS -= 360
            self.CenterEntryS.setText('%4.1f' % self.CenterS)
            self.UpdateCalibSub()
    def DistEnter(self):
        if(MainVid.fFileLoaded==0):
            return
        else:
            PData.MainCameraDist = float(self.DistEntry.text())
    def DistEnterSub(self):
        if(SubVid.fFileLoaded==0):
            return
        else:
            PData.SubCameraDist = float(self.DistEntryS.text())
    def FOVEnter(self):
        if(MainVid.fFileLoaded==0):
            return
        else:
            PData.MainCameraFovW = float(self.FOVEntry.text())
            PData.MainCameraFovH = float(PData.MainCameraFovW / MainVid.ResizeW * MainVid.ResizeH)
            self.UpdateCalibMain()
    def FOVEnterSub(self):
        if(SubVid.fFileLoaded==0):
            return
        else:
            PData.SubCameraFovW = float(self.FOVEntryS.text())
            PData.SubCameraFovH = float(PData.SubCameraFovW / SubVid.ResizeW * SubVid.ResizeH)
            self.UpdateCalibSub()
    def TargetEnter(self):
        if(MainVid.fFileLoaded==0):
            return
        else:
            self.TargetPixel = int(self.TargetEntry.text())
            self.TargetSlider.setValue(self.TargetPixel)
            self.UpdateCalibMain()
    def TargetEnterSub(self):
        if(SubVid.fFileLoaded==0):
            return
        else:
            self.TargetPixelS = int(self.TargetEntryS.text())
            self.TargetSliderS.setValue(self.TargetPixelS)
            self.UpdateCalibSub()
    def CameraCalib(self):
        if(MainVid.fFileLoaded==0):
            return
        else:
            self.FOVEnter()
            DegX = SData.pixel2deg(self.BallAngleX, self.TargetPixel, MainVid.ResizeW, PData.MainCameraFovW)
            DegY = SData.pixel2deg(self.BallAngleY, self.HLPixel, MainVid.ResizeH, PData.MainCameraFovH)
            (PData.MainCameraX, PData.MainCameraY, PData.MainCameraZ) = SData.cameraPosition(PData.MainCameraDist, DegX,  DegY)
            self.PxEntry.setText('%4.1f' % PData.MainCameraX)
            self.PyEntry.setText('%4.1f' % PData.MainCameraY)
            self.PzEntry.setText('%4.1f' % PData.MainCameraZ)
            MainVid.fCalibrated = 1
    def CameraCalibSub(self):
        if(SubVid.fFileLoaded==0):
            return
        else:
            self.FOVEnterSub()
            DegX = SData.pixel2deg(self.BallAngleXS, self.TargetPixelS, SubVid.ResizeW, PData.SubCameraFovW)
            DegY = SData.pixel2deg(self.BallAngleYS, int(SubVid.ResizeH/2), SubVid.ResizeH, PData.SubCameraFovH)
            (PData.SubCameraX, PData.SubCameraY, PData.SubCameraZ) = SData.cameraPosition(PData.SubCameraDist, DegX,  DegY)
            self.PxEntryS.setText('%4.1f' % PData.SubCameraX)
            self.PyEntryS.setText('%4.1f' % PData.SubCameraY)
            self.PzEntryS.setText('%4.1f' % PData.SubCameraZ)
            SubVid.fCalibrated = 1
    def UpdateCamera(self):
        if(MainVid.fFileLoaded==0):
            return
        else:
            PData.MainCameraX = float(self.PxEntry.text())
            PData.MainCameraY = float(self.PyEntry.text())
            PData.MainCameraZ = float(self.PzEntry.text())
            MainVid.fCalibrated = 1
    def UpdateCameraSub(self):
        if(SubVid.fFileLoaded==0):
            return
        else:
            PData.SubCameraX = float(self.PxEntryS.text())
            PData.SubCameraY = float(self.PyEntryS.text())
            PData.SubCameraZ = float(self.PzEntryS.text())
            SubVid.fCalibrated = 1
                
#For TabGLPlot ==============================================================
    def InitTabGLPlot(self):
        # Plotting
        self.gBox3 = QGridLayout(self.tabGLPlot)
        self.tabGLPlot.setLayout(self.gBox3)
        GLView.mousePressEvent = GLView.MouseClickGL
        GLView.mouseMoveEvent = GLView.MouseMoveGL
        GLView.wheelEvent = GLView.MouseWheelGL
        # GLView.resize(GLView.Width, GLView.Height)
        self.gBox3.addWidget(GLView, 0, 0, 20, 3)
        # Controls
        self.FrameBack = QPushButton("< 1 Frame", self)
        self.FrameBack.setMaximumWidth(100)
        self.FrameBack.clicked.connect(self.FrameRW)
        self.gBox3.addWidget(self.FrameBack, 1,3)
        self.PlayButton3 = QPushButton("Play >", self)
        self.PlayButton3.setMaximumWidth(100)
        self.PlayButton3.clicked.connect(GLView.ToggleGLPlay)
        self.gBox3.addWidget(self.PlayButton3, 1,4)
        self.FrameForward = QPushButton(" 1 Frame >", self)
        self.FrameForward.setMaximumWidth(100)
        self.FrameForward.clicked.connect(self.FrameFF)
        self.gBox3.addWidget(self.FrameForward, 1,5)
        self.FrameRemove = QPushButton("Remove Frame", self)
        self.FrameRemove.setMaximumWidth(100)
        self.FrameRemove.clicked.connect(self.FrameRem)
        self.gBox3.addWidget(self.FrameRemove, 1,6)
        self.PlaySpeed = QLabel("Play Speed    Slow")
        self.PlaySpeed.setAlignment(Qt.AlignRight)
        self.gBox3.addWidget(self.PlaySpeed, 2, 3)
        self.msTimerInterval = 30
        self.PlayTimerSlider = QSlider(Qt.Horizontal)
        # self.PlayTimerSlider.setMinimumWidth(10)
        self.PlayTimerSlider.setValue(20)
        self.PlayTimerSlider.setRange(1, 20)
        self.PlayTimerSlider.setMaximumWidth(200)
        self.PlayTimerSlider.valueChanged.connect(self.ChangeTimer)
        self.gBox3.addWidget(self.PlayTimerSlider, 2, 4, 1, 2)
        self.NormalSpeed = QLabel("Normal")
        self.NormalSpeed.setMaximumWidth(100)
        self.gBox3.addWidget(self.NormalSpeed, 2, 6)

        self.RotButton3 = QPushButton("< Rotate CCW", self)
        self.RotButton3.clicked.connect(GLView.GLRotationCCW)
        self.gBox3.addWidget(self.RotButton3, 3,3)
        self.RotButton3 = QPushButton("Stop Rotation", self)
        self.RotButton3.clicked.connect(GLView.GLRotationStop)
        self.gBox3.addWidget(self.RotButton3, 3,4)
        self.RotButton3 = QPushButton("Rotate CW >", self)
        self.RotButton3.clicked.connect(GLView.GLRotationCW)
        self.gBox3.addWidget(self.RotButton3, 3,5)

        # self.AltZButton = QPushButton("Alter Z data", self)
        # self.AltZButton.clicked.connect(SData.AlterZData)
        # self.gBox3.addWidget(self.AltZButton, 4,4)


        self.SpineLButton = QPushButton("Spine LIne", self)
        self.SpineLButton.clicked.connect(GLView.ToggleSpineLine)
        self.gBox3.addWidget(self.SpineLButton, 5,3)
        self.SpineLSlider = QSlider(Qt.Horizontal)
        self.SpineLSlider.setValue(20)
        self.SpineLSlider.setRange(0, 50)
        self.SpineLSlider.setMaximumWidth(100)
        self.SpineLSlider.valueChanged.connect(GLView.ChangeSpine)
        self.gBox3.addWidget(self.SpineLSlider, 5, 4)
        self.SpineTButton = QPushButton("Spine Trace", self)
        self.SpineTButton.clicked.connect(GLView.ToggleSpineTrace)
        self.gBox3.addWidget(self.SpineTButton, 5,5)

        self.ShoulderLButton = QPushButton("Shoulder LIne", self)
        self.ShoulderLButton.clicked.connect(GLView.ToggleShoulderLine)
        self.gBox3.addWidget(self.ShoulderLButton, 6,3)
        self.ShoulderLSlider = QSlider(Qt.Horizontal)
        self.ShoulderLSlider.setValue(20)
        self.ShoulderLSlider.setRange(0, 50)
        self.ShoulderLSlider.setMaximumWidth(100)
        self.ShoulderLSlider.valueChanged.connect(GLView.ChangeShoulder)
        self.gBox3.addWidget(self.ShoulderLSlider, 6, 4)
        self.ShoulderTButton = QPushButton("Shoulder Trace", self)
        self.ShoulderTButton.clicked.connect(GLView.ToggleShoulderTrace)
        self.gBox3.addWidget(self.ShoulderTButton, 6,5)
        self.ArmButton = QPushButton("Show/Hide Arms", self)
        self.ArmButton.clicked.connect(GLView.ToggleShowArms)
        self.gBox3.addWidget(self.ArmButton, 6,6)

        self.PelvisLButton = QPushButton("Pelvis LIne", self)
        self.PelvisLButton.clicked.connect(GLView.TogglePelvisLine)
        self.gBox3.addWidget(self.PelvisLButton, 7,3)
        self.PelvisLSlider = QSlider(Qt.Horizontal)
        self.PelvisLSlider.setValue(20)
        self.PelvisLSlider.setRange(0, 50)
        self.PelvisLSlider.setMaximumWidth(100)
        self.PelvisLSlider.valueChanged.connect(GLView.ChangePelvis)
        self.gBox3.addWidget(self.PelvisLSlider, 7, 4)
        self.PelvisTButton = QPushButton("Pelvis Trace", self)
        self.PelvisTButton.clicked.connect(GLView.TogglePelvisTrace)
        self.gBox3.addWidget(self.PelvisTButton, 7,5)
        self.PCTButton3 = QPushButton("Pelvis Center Trace", self)
        self.PCTButton3.clicked.connect(GLView.ToggleGLPCenter)
        self.gBox3.addWidget(self.PCTButton3, 7,6)

        self.KneeLButton = QPushButton("Knee LIne", self)
        self.KneeLButton.clicked.connect(GLView.ToggleKneeLine)
        self.gBox3.addWidget(self.KneeLButton, 8,3)
        self.KneeLSlider = QSlider(Qt.Horizontal)
        self.KneeLSlider.setValue(20)
        self.KneeLSlider.setRange(0, 50)
        self.KneeLSlider.setMaximumWidth(100)
        self.KneeLSlider.valueChanged.connect(GLView.ChangeKnee)
        self.gBox3.addWidget(self.KneeLSlider, 8, 4)
        self.KneeTButton = QPushButton("Knee Trace", self)
        self.KneeTButton.clicked.connect(GLView.ToggleKneeTrace)
        self.gBox3.addWidget(self.KneeTButton, 8,5)

        self.Translation = QLabel("Translation [From Ball]  [Towards Target]  [Above Ground]")
        self.gBox3.addWidget(self.Translation, 11, 3, 1, 3)
        self.TrYSlider = QSlider(Qt.Horizontal)
        self.TrYSlider.setValue(0)
        self.TrYSlider.setRange(-50, 50)
        self.TrYSlider.setMaximumWidth(100)
        self.TrYSlider.valueChanged.connect(GLView.TranslateY)
        self.gBox3.addWidget(self.TrYSlider, 13, 3)
        self.TrXSlider = QSlider(Qt.Vertical)
        self.TrXSlider.setValue(0)
        self.TrXSlider.setRange(-50, 50)
        self.TrXSlider.setMaximumWidth(30)
        self.TrXSlider.setMaximumHeight(50)
        self.TrXSlider.valueChanged.connect(GLView.TranslateX)
        self.gBox3.addWidget(self.TrXSlider, 12, 5, 2, 1)
        self.TrZSlider = QSlider(Qt.Vertical)
        self.TrZSlider.setValue(0)
        self.TrZSlider.setRange(-50, 50)
        self.TrZSlider.setMaximumWidth(30)
        self.TrZSlider.setMaximumHeight(70)
        self.TrZSlider.valueChanged.connect(GLView.TranslateZ)
        self.gBox3.addWidget(self.TrZSlider, 11, 6, 3, 1)



        # Timer for 3D plot Animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(GLView.update)
        self.timer.start(self.msTimerInterval)

    def FrameRW(self):
        GLView.idxFrame -=1
        if(GLView.idxFrame<0):
            GLView.idxFrame = 0
        GLView.repaint()

    def FrameFF(self):
        GLView.idxFrame +=1
        if(GLView.idxFrame>SData.nFrames -1):
            GLView.idxFrame = SData.nFrames -1
        GLView.repaint()
        
    def FrameRem(self):
        SData.RemoveFrame(GLView.idxFrame)
        if(GLView.idxFrame>1):
            GLView.idxFrame -=1
        GLView.repaint()

    def ChangeTimer(self, value):
        self.msTimerInterval = int(600/(float(value)))
        self.timer.setInterval(self.msTimerInterval)                
        # self.timer.start(self.msTimerInterval)



# For TabCSV ==============================================================
    def InitTabCSV(self):
        self.vBox1 = QVBoxLayout(self.tabCSV)
        self.table = QTableWidget(10, 10, self)
        self.vBox1.addWidget(self.table)
        self.table.resize(600,400)
    def LoadJoints3D(self, FileName):
        f = open(FileName, 'r')
        ReadCSV = csv.reader(f)
        lines = list(ReadCSV)
        # TempJoints = np.zeros(1, dtype=np.float32)
        self.table.setRowCount(len(lines))
        self.table.setColumnCount(len(lines[0]))
        idx_Row = 0
        idx_Col = 0
        for row in lines:
            for value in row:
                if((idx_Row==0) and (idx_Col==0)):
                    SData.Joints3D = float(value)
                else:
                    SData.Joints3D = np.append(SData.Joints3D, float(value))
                value_str = '%4.3f' % float(value)
                self.table.setItem(idx_Row, idx_Col, QTableWidgetItem(value_str))
                # print('Table: %d, %d = %s' % (idx_Row, idx_Col, value_str))
                idx_Col +=1
            idx_Col = 0
            idx_Row += 1
        SData.nFrames = int(len(SData.Joints3D)/(SData.n3dElements*SData.nJoints3D))
        f.close()
        SData.fCSVLoaded = 1
        SData.CalculateExtents()
        if(MainVid.fFileLoaded == 1):
            MainVid.InitVideo()
            self.UpdateImageLabel()
        if(SubVid.fFileLoaded == 1):
            SubVid.InitVideo()
            self.UpdateImageLabelSub()
    def openCSVFile(self):
        filter = "csv(*.csv)"
        fname = QFileDialog.getOpenFileName(self, 'Open file', '~\\Videos', filter)
        if fname[0]:
            self.LoadJoints3D(fname[0])


# ========================================================================
#
#                   OpenGL Widget
#
# ========================================================================
class GLWidget(QOpenGLWidget):
    Width = 600
    Height = 400
    ModelViewW = 3.0
    ModelViewH = 2.0
    ViewScale = ModelViewW/Width
    ViweAspect = ModelViewH/ModelViewW
    
    AngleRotation = 0
    AngleView = 10
    idxFrame = 0
    fPlay = 0
    fRotation = 0
    fShowArms = 1
    fPCenterTrace = 0
    fShoulderLine = 0
    fPelvisLine = 0
    fKneeLine = 0
    fSpineLine = 0
    fShoulderLineTrace = 0
    fPelvisLineTrace = 0
    fKneeLineTrace = 0
    fSpineLineTrace = 0
    TrX = 0
    TrY = 0
    TrZ = 0
    
    def ToggleGLPlay(self):
        if(self.fPlay == 0):
            self.fPlay = 1
        else:
            self.fPlay = 0
    def GLRotationCW(self):
        if(self.fRotation != 1):
            self.fRotation = 1
        else:
            self.fRotation = 0
    def GLRotationCCW(self):
        if(self.fRotation != -1):
            self.fRotation = -1
        else:
            self.fRotation = 0
    def GLRotationStop(self):
        self.fRotation = 0
    def ToggleGLPCenter(self):
        if(self.fPCenterTrace == 0):
            self.fPCenterTrace = 1
        else:
            self.fPCenterTrace = 0
    def ToggleShowArms(self):
        if(self.fShowArms == 0):
            self.fShowArms = 1
        else:
            self.fShowArms = 0
    def ToggleShoulderLine(self):
        if(self.fShoulderLine == 0):
            self.fShoulderLine = 1
        else:
            self.fShoulderLine = 0
    def ChangeShoulder(self, value):
        SData.ShoulderExtent = 0.08 * float(value)
        SData.CalculateExtents()
    def ToggleShoulderTrace(self):
        if(self.fShoulderLineTrace == 0):
            self.fShoulderLineTrace = 1
        else:
            self.fShoulderLineTrace = 0
    def TogglePelvisLine(self):
        if(self.fPelvisLine == 0):
            self.fPelvisLine = 1
        else:
            self.fPelvisLine = 0
    def ChangePelvis(self, value):
        SData.PelvisExtent = 0.1 * float(value)
        SData.CalculateExtents()
    def TogglePelvisTrace(self):
        if(self.fPelvisLineTrace == 0):
            self.fPelvisLineTrace = 1
        else:
            self.fPelvisLineTrace = 0
    def ToggleKneeLine(self):
        if(self.fKneeLine == 0):
            self.fKneeLine = 1
        else:
            self.fKneeLine = 0
    def ChangeKnee(self, value):
        SData.KneeExtent = 0.06 * float(value)
        SData.CalculateExtents()
    def ToggleKneeTrace(self):
        if(self.fKneeLineTrace == 0):
            self.fKneeLineTrace = 1
        else:
            self.fKneeLineTrace = 0
    def ToggleSpineLine(self):
        if(self.fSpineLine == 0):
            self.fSpineLine = 1
        else:
            self.fSpineLine = 0
    def ChangeSpine(self, value):
        SData.SpineExtent = 0.04 * float(value)
        SData.CalculateExtents()
    def ToggleSpineTrace(self):
        if(self.fSpineLineTrace == 0):
            self.fSpineLineTrace = 1
        else:
            self.fSpineLineTrace = 0
    def TranslateX(self, value):
        self.TrX = 0.05 * float(value)
    def TranslateY(self, value):
        self.TrY = -0.05 * float(value)
    def TranslateZ(self, value):
        self.TrZ = 0.02 * float(value)
    def CCW(self):
        self.AngleRotation -= 1
        if(self.AngleRotation<0):
            self.AngleRotation = 359
    def CW(self):
        self.AngleRotation += 1
        if(self.AngleRotation>=360):
            self.AngleRotation = 0
    def UpView(self):
        self.AngleView += 5
        if(self.AngleView>=90):
            self.AngleView = 90
    def DownView(self):
        self.AngleView -= 5
        if(self.AngleView<=0):
            self.AngleView = 0
    def MouseClickGL(self, event):
        if(self.fRotation!=0):
            self.fRotation = 0
        if(event.button() == Qt.LeftButton):
            self.DragStart = event.pos()
    def MouseMoveGL(self, event):
        if(self.fRotation==1):
            self.fRotation = 0
        self.DragCurrent = event.pos()
        self.LeftRight = self.DragCurrent.x() - self.DragStart.x()
        self.AngleRotation += self.LeftRight * 0.5
        if(self.AngleRotation<=0):
            self.AngleRotation += 360
        if(self.AngleRotation>=360):
            self.AngleRotation -= 360
        self.UpDown = self.DragCurrent.y() - self.DragStart.y()
        self.AngleView += self.UpDown*0.5
        if(self.AngleView>=90):
            self.AngleView = 90
        if(self.AngleView<=0):
            self.AngleView = 0
        self.DragStart = self.DragCurrent
    def MouseWheelGL(self, event):
        self.ViewScale -= event.angleDelta().y()*0.00001
        if(self.ViewScale < 3/1800):
            self.ViewScale = 3/1800
        elif(self.ViewScale > 3/400):
            self.ViewScale = 3/400

    def __init__(self):
        super().__init__()

    def initializeGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glClearColor(0.0, 0.0, 0.2, 1.0)
        GL.glClearDepth(1.0)              
        GL.glDepthFunc(GL.GL_LESS)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_BLEND)
        GL.glShadeModel(GL.GL_SMOOTH)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-2.5, 2.5, -1.1, 1.4, -3.0, 2.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

    def resizeGL(self, w, h):
        self.Width = w
        self.Height = h
        GL.glViewport(0, 0, self.Width, self.Height)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-self.ViewScale*self.Width/2, self.ViewScale*self.Width/2, \
                   -self.ViewScale*self.Height/2, self.ViewScale*self.Height/2, \
                       -self.ViewScale*self.Width/2, self.ViewScale*self.Width/2)
        
        # GL.glOrtho(-2.5, 2.5, -1.1, 1.4, -3.0, 2.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        GL.glRotatef(-90+self.AngleView, 1, 0, 0)
        GL.glRotatef(self.AngleRotation, 0, 0, 1)
        GL.glTranslatef(0.5, -1.5, -1.0)
        
        GL.glPushMatrix()

        #Draw Ground level
        GL.glColor4f( 0.0, 0.5, 0.0, 1.0 )
        GL.glBegin(GL.GL_LINES)
        for y in range(7):
            GL.glVertex3f(-2, y*0.5, 0)
            GL.glVertex3f(1, y*0.5, 0)
        GL.glEnd()
        GL.glBegin(GL.GL_LINES)
        for x in range(7):
            GL.glVertex3f(-2+x*0.5, 0, 0)
            GL.glVertex3f(-2+x*0.5, 3, 0)
        GL.glEnd()

        #Model Translation
        GL.glTranslatef(self.TrX, self.TrY, self.TrZ)


        if(SData.fCSVLoaded == 1):
            offset = int(self.idxFrame*SData.nJoints3D*SData.n3dElements)
            for i in range(SData.nJoints3D):
                x=SData.Joints3D[offset+i*SData.n3dElements]
                y=SData.Joints3D[offset+i*SData.n3dElements+1]
                z=SData.Joints3D[offset+i*SData.n3dElements+2]
                scale = 0.03
                if(self.fShowArms == 1):
                    self.DrawDiamond(x, y, z, scale)
                elif((i != 2) and (i != 3) and (i !=5) and (i != 6)):
                    self.DrawDiamond(x, y, z, scale)
            #Body Center
            x=np.zeros(4, dtype=np.float32)
            y=np.zeros(4, dtype=np.float32)
            z=np.zeros(4, dtype=np.float32)
            x[0] = SData.Joints3D[offset+0*SData.n3dElements]
            y[0] = SData.Joints3D[offset+0*SData.n3dElements+1]
            z[0] = SData.Joints3D[offset+0*SData.n3dElements+2]
            x[1] = SData.Joints3D[offset+7*SData.n3dElements]
            y[1] = SData.Joints3D[offset+7*SData.n3dElements+1]
            z[1] = SData.Joints3D[offset+7*SData.n3dElements+2]
            GL.glColor4f( 0.2, 0.7, 0.2, 0.9 )
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(x[0], y[0], z[0])
            GL.glVertex3f(x[1], y[1], z[1])
            GL.glEnd()
            #RightSide
            x[0] = SData.Joints3D[offset+1*SData.n3dElements]
            y[0] = SData.Joints3D[offset+1*SData.n3dElements+1]
            z[0] = SData.Joints3D[offset+1*SData.n3dElements+2]
            x[1] = SData.Joints3D[offset+8*SData.n3dElements]
            y[1] = SData.Joints3D[offset+8*SData.n3dElements+1]
            z[1] = SData.Joints3D[offset+8*SData.n3dElements+2]
            x[2] = SData.Joints3D[offset+9*SData.n3dElements]
            y[2] = SData.Joints3D[offset+9*SData.n3dElements+1]
            z[2] = SData.Joints3D[offset+9*SData.n3dElements+2]
            x[3] = SData.Joints3D[offset+10*SData.n3dElements]
            y[3] = SData.Joints3D[offset+10*SData.n3dElements+1]
            z[3] = SData.Joints3D[offset+10*SData.n3dElements+2]
            GL.glColor4f( 0.7, 0.2, 0.2, 0.9 )
            GL.glBegin(GL.GL_LINE_STRIP)
            # GL.glVertex3f(x[0], y[0], z[0])
            GL.glVertex3f(x[1], y[1], z[1])
            GL.glVertex3f(x[2], y[2], z[2])
            GL.glVertex3f(x[3], y[3], z[3])
            GL.glEnd()
            #RightHip
            x[0] = SData.Joints3D[offset+7*SData.n3dElements]
            y[0] = SData.Joints3D[offset+7*SData.n3dElements+1]
            z[0] = SData.Joints3D[offset+7*SData.n3dElements+2]
            x[1] = SData.Joints3D[offset+8*SData.n3dElements]
            y[1] = SData.Joints3D[offset+8*SData.n3dElements+1]
            z[1] = SData.Joints3D[offset+8*SData.n3dElements+2]
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(x[0], y[0], z[0])
            GL.glVertex3f(x[1], y[1], z[1])
            GL.glEnd()
            #RightArm
            x[0] = SData.Joints3D[offset+0*SData.n3dElements]
            y[0] = SData.Joints3D[offset+0*SData.n3dElements+1]
            z[0] = SData.Joints3D[offset+0*SData.n3dElements+2]
            x[1] = SData.Joints3D[offset+1*SData.n3dElements]
            y[1] = SData.Joints3D[offset+1*SData.n3dElements+1]
            z[1] = SData.Joints3D[offset+1*SData.n3dElements+2]
            x[2] = SData.Joints3D[offset+2*SData.n3dElements]
            y[2] = SData.Joints3D[offset+2*SData.n3dElements+1]
            z[2] = SData.Joints3D[offset+2*SData.n3dElements+2]
            x[3] = SData.Joints3D[offset+3*SData.n3dElements]
            y[3] = SData.Joints3D[offset+3*SData.n3dElements+1]
            z[3] = SData.Joints3D[offset+3*SData.n3dElements+2]
            GL.glBegin(GL.GL_LINE_STRIP)
            GL.glVertex3f(x[0], y[0], z[0])
            GL.glVertex3f(x[1], y[1], z[1])
            if(self.fShowArms == 1):
                GL.glVertex3f(x[2], y[2], z[2])
                GL.glVertex3f(x[3], y[3], z[3])
            GL.glEnd()
            #LefttSide
            x[0] = SData.Joints3D[offset+4*SData.n3dElements]
            y[0] = SData.Joints3D[offset+4*SData.n3dElements+1]
            z[0] = SData.Joints3D[offset+4*SData.n3dElements+2]
            x[1] = SData.Joints3D[offset+11*SData.n3dElements]
            y[1] = SData.Joints3D[offset+11*SData.n3dElements+1]
            z[1] = SData.Joints3D[offset+11*SData.n3dElements+2]
            x[2] = SData.Joints3D[offset+12*SData.n3dElements]
            y[2] = SData.Joints3D[offset+12*SData.n3dElements+1]
            z[2] = SData.Joints3D[offset+12*SData.n3dElements+2]
            x[3] = SData.Joints3D[offset+13*SData.n3dElements]
            y[3] = SData.Joints3D[offset+13*SData.n3dElements+1]
            z[3] = SData.Joints3D[offset+13*SData.n3dElements+2]
            GL.glColor4f( 0.2, 0.2, 1.0, 0.9 )
            GL.glBegin(GL.GL_LINE_STRIP)
            # GL.glVertex3f(x[0], y[0], z[0])
            GL.glVertex3f(x[1], y[1], z[1])
            GL.glVertex3f(x[2], y[2], z[2])
            GL.glVertex3f(x[3], y[3], z[3])
            GL.glEnd()
            #LeftHip
            x[0] = SData.Joints3D[offset+7*SData.n3dElements]
            y[0] = SData.Joints3D[offset+7*SData.n3dElements+1]
            z[0] = SData.Joints3D[offset+7*SData.n3dElements+2]
            x[1] = SData.Joints3D[offset+11*SData.n3dElements]
            y[1] = SData.Joints3D[offset+11*SData.n3dElements+1]
            z[1] = SData.Joints3D[offset+11*SData.n3dElements+2]
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(x[0], y[0], z[0])
            GL.glVertex3f(x[1], y[1], z[1])
            GL.glEnd()
            #LeftArm
            x[0] = SData.Joints3D[offset+0*SData.n3dElements]
            y[0] = SData.Joints3D[offset+0*SData.n3dElements+1]
            z[0] = SData.Joints3D[offset+0*SData.n3dElements+2]
            x[1] = SData.Joints3D[offset+4*SData.n3dElements]
            y[1] = SData.Joints3D[offset+4*SData.n3dElements+1]
            z[1] = SData.Joints3D[offset+4*SData.n3dElements+2]
            x[2] = SData.Joints3D[offset+5*SData.n3dElements]
            y[2] = SData.Joints3D[offset+5*SData.n3dElements+1]
            z[2] = SData.Joints3D[offset+5*SData.n3dElements+2]
            x[3] = SData.Joints3D[offset+6*SData.n3dElements]
            y[3] = SData.Joints3D[offset+6*SData.n3dElements+1]
            z[3] = SData.Joints3D[offset+6*SData.n3dElements+2]
            GL.glBegin(GL.GL_LINE_STRIP)
            GL.glVertex3f(x[0], y[0], z[0])
            GL.glVertex3f(x[1], y[1], z[1])
            if(self.fShowArms == 1):
                GL.glVertex3f(x[2], y[2], z[2])
                GL.glVertex3f(x[3], y[3], z[3])
            GL.glEnd()
            
            if(self.fPCenterTrace == 1):
                x1=np.zeros(SData.nFrames, dtype=np.float32)
                y1=np.zeros(SData.nFrames, dtype=np.float32)
                z1=np.zeros(SData.nFrames, dtype=np.float32)
                GL.glColor4f( 0.8, 0.8, 0.8, 0.5 )
                GL.glBegin(GL.GL_LINE_STRIP)
                for idx in range(SData.nFrames):
                    x1[idx] = SData.PCenter3D[idx*4]
                    y1[idx] = SData.PCenter3D[idx*4+1]
                    z1[idx] = SData.PCenter3D[idx*4+2]
                    GL.glVertex3f(x1[idx], y1[idx], z1[idx])
                GL.glEnd()

            if(self.fShoulderLineTrace == 1):
                x1=np.zeros(SData.nFrames, dtype=np.float32)
                y1=np.zeros(SData.nFrames, dtype=np.float32)
                z1=np.zeros(SData.nFrames, dtype=np.float32)
                GL.glColor4f( 0.4, 0.4, 0.7, 0.4 )
                GL.glBegin(GL.GL_LINE_STRIP)
                for idx in range(SData.nFrames):
                    x1[idx] = SData.ShoulderLineL[idx*3]
                    y1[idx] = SData.ShoulderLineL[idx*3+1]
                    z1[idx] = SData.ShoulderLineL[idx*3+2]
                    GL.glVertex3f(x1[idx], y1[idx], z1[idx])
                GL.glEnd()
                GL.glColor4f( 0.7, 0.4, 0.4, 0.4 )
                GL.glBegin(GL.GL_LINE_STRIP)
                for idx in range(SData.nFrames):
                    x1[idx] = SData.ShoulderLineR[idx*3]
                    y1[idx] = SData.ShoulderLineR[idx*3+1]
                    z1[idx] = SData.ShoulderLineR[idx*3+2]
                    GL.glVertex3f(x1[idx], y1[idx], z1[idx])
                GL.glEnd()

            if(self.fShoulderLine == 1):
                GL.glColor4f( 0.8, 0.8, 0.1, 0.7 )
                GL.glBegin(GL.GL_LINES)
                GL.glVertex3f(SData.ShoulderLineL[self.idxFrame*3], SData.ShoulderLineL[self.idxFrame*3+1], SData.ShoulderLineL[self.idxFrame*3+2])
                GL.glVertex3f(SData.ShoulderLineR[self.idxFrame*3], SData.ShoulderLineR[self.idxFrame*3+1], SData.ShoulderLineR[self.idxFrame*3+2])
                GL.glEnd()

            if(self.fPelvisLineTrace == 1):
                x1=np.zeros(SData.nFrames, dtype=np.float32)
                y1=np.zeros(SData.nFrames, dtype=np.float32)
                z1=np.zeros(SData.nFrames, dtype=np.float32)
                GL.glColor4f( 0.3, 0.3, 0.6, 0.4 )
                GL.glBegin(GL.GL_LINE_STRIP)
                for idx in range(SData.nFrames):
                    x1[idx] = SData.PelvisLineL[idx*3]
                    y1[idx] = SData.PelvisLineL[idx*3+1]
                    z1[idx] = SData.PelvisLineL[idx*3+2]
                    GL.glVertex3f(x1[idx], y1[idx], z1[idx])
                GL.glEnd()
                GL.glColor4f( 0.6, 0.3, 0.3, 0.4 )
                GL.glBegin(GL.GL_LINE_STRIP)
                for idx in range(SData.nFrames):
                    x1[idx] = SData.PelvisLineR[idx*3]
                    y1[idx] = SData.PelvisLineR[idx*3+1]
                    z1[idx] = SData.PelvisLineR[idx*3+2]
                    GL.glVertex3f(x1[idx], y1[idx], z1[idx])
                GL.glEnd()

            if(self.fPelvisLine == 1):
                GL.glColor4f( 0.8, 0.8, 0.1, 0.7 )
                GL.glBegin(GL.GL_LINES)
                GL.glVertex3f(SData.PelvisLineL[self.idxFrame*3], SData.PelvisLineL[self.idxFrame*3+1], SData.PelvisLineL[self.idxFrame*3+2])
                GL.glVertex3f(SData.PelvisLineR[self.idxFrame*3], SData.PelvisLineR[self.idxFrame*3+1], SData.PelvisLineR[self.idxFrame*3+2])
                GL.glEnd()
            
            if(self.fKneeLineTrace == 1):
                x1=np.zeros(SData.nFrames, dtype=np.float32)
                y1=np.zeros(SData.nFrames, dtype=np.float32)
                z1=np.zeros(SData.nFrames, dtype=np.float32)
                GL.glColor4f( 0.3, 0.3, 0.5, 0.4 )
                GL.glBegin(GL.GL_LINE_STRIP)
                for idx in range(SData.nFrames):
                    x1[idx] = SData.KneeLineL[idx*3]
                    y1[idx] = SData.KneeLineL[idx*3+1]
                    z1[idx] = SData.KneeLineL[idx*3+2]
                    GL.glVertex3f(x1[idx], y1[idx], z1[idx])
                GL.glEnd()
                GL.glColor4f( 0.5, 0.3, 0.3, 0.4 )
                GL.glBegin(GL.GL_LINE_STRIP)
                for idx in range(SData.nFrames):
                    x1[idx] = SData.KneeLineR[idx*3]
                    y1[idx] = SData.KneeLineR[idx*3+1]
                    z1[idx] = SData.KneeLineR[idx*3+2]
                    GL.glVertex3f(x1[idx], y1[idx], z1[idx])
                GL.glEnd()

            if(self.fKneeLine == 1):
                GL.glColor4f( 0.8, 0.8, 0.1, 0.7 )
                GL.glBegin(GL.GL_LINES)
                GL.glVertex3f(SData.KneeLineL[self.idxFrame*3], SData.KneeLineL[self.idxFrame*3+1], SData.KneeLineL[self.idxFrame*3+2])
                GL.glVertex3f(SData.KneeLineR[self.idxFrame*3], SData.KneeLineR[self.idxFrame*3+1], SData.KneeLineR[self.idxFrame*3+2])
                GL.glEnd()
            
            if(self.fSpineLineTrace == 1):
                x1=np.zeros(SData.nFrames, dtype=np.float32)
                y1=np.zeros(SData.nFrames, dtype=np.float32)
                z1=np.zeros(SData.nFrames, dtype=np.float32)
                GL.glColor4f( 0.2, 0.9, 0.2, 0.4 )
                GL.glBegin(GL.GL_LINE_STRIP)
                for idx in range(SData.nFrames):
                    x1[idx] = SData.SpineHigh[idx*3]
                    y1[idx] = SData.SpineHigh[idx*3+1]
                    z1[idx] = SData.SpineHigh[idx*3+2]
                    GL.glVertex3f(x1[idx], y1[idx], z1[idx])
                GL.glEnd()
                GL.glBegin(GL.GL_LINE_STRIP)
                for idx in range(SData.nFrames):
                    x1[idx] = SData.SpineLow[idx*3]
                    y1[idx] = SData.SpineLow[idx*3+1]
                    z1[idx] = SData.SpineLow[idx*3+2]
                    GL.glVertex3f(x1[idx], y1[idx], z1[idx])
                GL.glEnd()

            if(self.fSpineLine == 1):
                GL.glColor4f( 0.1, 0.8, 0.1, 0.5 )
                GL.glBegin(GL.GL_LINES)
                GL.glVertex3f(SData.SpineHigh[self.idxFrame*3], SData.SpineHigh[self.idxFrame*3+1], SData.SpineHigh[self.idxFrame*3+2])
                GL.glVertex3f(SData.SpineLow[self.idxFrame*3], SData.SpineLow[self.idxFrame*3+1], SData.SpineLow[self.idxFrame*3+2])
                GL.glEnd()

            if(self.fRotation == 1):
                self.AngleRotation += 1
                if(self.AngleRotation>=360):
                    self.AngleRotation -= 360
            elif(self.fRotation == -1):
                self.AngleRotation -= 1
                if(self.AngleRotation<=0):
                    self.AngleRotation += 360
            if(self.fPlay == 1):
                self.idxFrame +=1
                if(self.idxFrame>SData.nFrames-1):
                    self.idxFrame = 0

        GL.glPopMatrix()


        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        # GL.glScalef(1, 1, 5)

        GL.glFlush()
    
    def DrawCube(self, X, Y, Z, S):
        x = float(X)
        y = float(Y)
        z = float(Z)
        scale = float(S)
        GL.glBegin(GL.GL_LINE_STRIP)
        GL.glVertex3f((x)-(scale), (y)-(scale), (z)-(scale))
        GL.glVertex3f((x)+(scale), (y)-(scale), (z)-(scale))
        GL.glVertex3f((x)+(scale), (y)+(scale), (z)-(scale))
        GL.glVertex3f((x)-(scale), (y)+(scale), (z)-(scale))
        GL.glVertex3f((x)-(scale), (y)-(scale), (z)-(scale))
        GL.glEnd()
        GL.glBegin(GL.GL_LINE_STRIP)
        GL.glVertex3f((x)-(scale), (y)-(scale), (z)+(scale))
        GL.glVertex3f((x)+(scale), (y)-(scale), (z)+(scale))
        GL.glVertex3f((x)+(scale), (y)+(scale), (z)+(scale))
        GL.glVertex3f((x)-(scale), (y)+(scale), (z)+(scale))
        GL.glVertex3f((x)-(scale), (y)-(scale), (z)+(scale))
        GL.glEnd()
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f((x)-(scale), (y)-(scale), (z)-(scale))
        GL.glVertex3f((x)-(scale), (y)-(scale), (z)+(scale))
        GL.glEnd()
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f((x)+(scale), (y)-(scale), (z)-(scale))
        GL.glVertex3f((x)+(scale), (y)-(scale), (z)+(scale))
        GL.glEnd()
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f((x)+(scale), (y)+(scale), (z)-(scale))
        GL.glVertex3f((x)+(scale), (y)+(scale), (z)+(scale))
        GL.glEnd()
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f((x)-(scale), (y)+(scale), (z)-(scale))
        GL.glVertex3f((x)-(scale), (y)+(scale), (z)+(scale))
        GL.glEnd()
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f((x)-(scale), (y)-(scale), (z)-(scale))
        GL.glVertex3f((x)+(scale), (y)+(scale), (z)+(scale))
        GL.glEnd()
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f((x)-(scale), (y)+(scale), (z)-(scale))
        GL.glVertex3f((x)+(scale), (y)-(scale), (z)+(scale))
        GL.glEnd()
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f((x)+(scale), (y)+(scale), (z)-(scale))
        GL.glVertex3f((x)-(scale), (y)-(scale), (z)+(scale))
        GL.glEnd()
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f((x)+(scale), (y)-(scale), (z)-(scale))
        GL.glVertex3f((x)-(scale), (y)+(scale), (z)+(scale))
        GL.glEnd()

    def DrawDiamond(self, X, Y, Z, S):
        x = float(X)
        y = float(Y)
        z = float(Z)
        scale = float(S)
        GL.glColor4f( 0.7, 0.7, 0.7, 0.7 )
        GL.glBegin(GL.GL_LINE_STRIP)
        GL.glVertex3f(x, y,       z+scale)
        GL.glVertex3f(x, y+scale, z      )
        GL.glVertex3f(x, y      , z-scale)
        GL.glVertex3f(x, y-scale, z      )
        GL.glVertex3f(x, y      , z+scale)
        GL.glEnd()
        GL.glBegin(GL.GL_LINE_STRIP)
        GL.glVertex3f(x      , y, z+scale)
        GL.glVertex3f(x+scale, y, z      )
        GL.glVertex3f(x      , y, z-scale)
        GL.glVertex3f(x-scale, y, z      )
        GL.glVertex3f(x      , y, z+scale)
        GL.glEnd()
        GL.glBegin(GL.GL_LINE_STRIP)
        GL.glVertex3f(x      , y+scale, z)
        GL.glVertex3f(x+scale, y      , z)
        GL.glVertex3f(x      , y-scale, z)
        GL.glVertex3f(x-scale, y      , z)
        GL.glVertex3f(x      , y+scale, z)
        GL.glEnd()

if __name__ == '__main__':
    # QApplication.setAttribute(Qt.AA_UseDesktopOpenGL)
    app = QApplication([])
    PData = ProjectData()
    SData = SkeletonData()
    GLView = GLWidget()
    MainVid = VideoData()
    SubVid = VideoData()
    MainPose = PoseData()
    SubPose = PoseData()
    # CordSys = Coordinate3D()

    # app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
    if(MainWindow.capVid.isOpened()):
        MainWindow.capVid.release()
