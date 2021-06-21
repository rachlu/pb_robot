#!/usr/bin/env python

#from __future__ import print_function

import os
import copy
import numpy
import math
import IPython
from tsr.tsr import TSR, TSRChain
import pb_robot
import pybullet

def simpleExample(robot):
    t0 = robot.arm.GetEETransform()
    q0 = robot.arm.GetJointValues()
    tgoal = copy.deepcopy(t0)
    tgoal[0, 3] += 0.25
    pb_robot.viz.draw_pose(pb_robot.geometry.pose_from_tform(tgoal), width=5, length=0.05)

    ### Create TSRS
    # Effectly constrain to move in straight path
    Bw_test = numpy.zeros((6, 2))
    Bw_test[0, 0] = -0.3
    Bw_test[0, 1] = 0.3
    Bw_test[1:3, 0] = -0.1
    Bw_test[1:3, 1] = 0.1
    Bw_test[3:6, 0] = -numpy.pi/8.0
    Bw_test[3:6, 1] = numpy.pi/8.0

    test_tsr = TSR(T0_w = t0, Tw_e = numpy.eye(4), Bw = Bw_test)
    chain = TSRChain(sample_start=False, sample_goal=False,
                     constrain=True, TSRs=[test_tsr])

    path = robot.arm.cbirrt.PlanToEndEffectorPose(robot.arm, q0, tgoal, pathTSRs=chain)
    return path

if __name__ == '__main__':
    # Launch pybullet
    pb_robot.utils.connect(use_gui=True)
    pb_robot.utils.disable_real_time()
    pb_robot.utils.set_default_camera()
    pybullet.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0, 0, 0])

    # Create robot object 
    robot = pb_robot.panda.Panda() 
 
    # Add floor object 
    objects_path = pb_robot.helper.getDirectory()
    floor_file = os.path.join(objects_path, 'furniture/short_floor.urdf')
    floor = pb_robot.body.createBody(floor_file)

    q0 = robot.arm.GetJointValues()
    q1 = robot.arm.randomConfiguration()
    path0 = robot.arm.cbirrt.PlanToConfiguration(robot.arm, q0, q1)
    ee1 = robot.arm.ComputeFK(q1)
    path1 = robot.arm.cbirrt.PlanToEndEffectorPose(robot.arm, q0, ee1)



    IPython.embed()
    
    # Close out Pybullet
    pb_robot.utils.wait_for_user()
    pb_robot.utils.disconnect()
