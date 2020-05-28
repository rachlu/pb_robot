#!/usr/bin/env python

#from __future__ import print_function

import os
import IPython
import pb_robot

if __name__ == '__main__':
    # Launch pybullet
    pb_robot.utils.connect(use_gui=True)
    pb_robot.utils.disable_real_time()
    pb_robot.utils.set_default_camera()

    # Create robot object 
    robot = pb_robot.panda.Panda() 
 
    # Add floor object 
    objects_path = pb_robot.helper.getDirectory()
    floor_file = os.path.join(objects_path, 'furniture/short_floor.urdf')
    floor = pb_robot.body.createBody(floor_file)

    # Example function on body object
    print floor.get_transform()

    # Example functions over robot arm
    q = robot.arm.GetJointValues()
    pose = robot.arm.ComputeFK(q)
    pose[2, 3] -= 0.1
    pb_robot.viz.draw_pose(pb_robot.geometry.pose_from_tform(pose), length=0.5, width=10)
    newq = robot.arm.ComputeIK(pose)
    if newq is not None:
        raw_input("Move to desired pose?")
        robot.arm.SetJointValues(newq)

    IPython.embed()
    
    # Close out Pybullet
    pb_robot.utils.wait_for_user()
    pb_robot.utils.disconnect()
