#!/usr/bin/env python

#from __future__ import print_function

import IPython
import numpy
import os
import pb_robot
import pb_robot.utils_noBase as utils

def testIK(robot):
    total = 0
    for i in xrange(100):
        if i % 10 == 0: print(total)

        q = robot.arm.randomConfiguration()
        pose = robot.arm.ComputeFK(q)
        #pb_robot.viz.draw_pose(pb_robot.geometry.pose_from_tform(pose), length=0.5, width=10)
        #raw_input("GO?")

        full_solved_q = robot.arm.ComputeIK(pose)
        #full_solved_q = robot.arm.RandomIK(pose)

        if full_solved_q is None:
            continue
        solved_q = full_solved_q[0:7]
        solved_pose = robot.arm.ComputeFK(solved_q)
        error = pb_robot.geometry.GeodesicDistance(pose, solved_pose)
        second_error = utils.is_pose_close(pb_robot.geometry.pose_from_tform(pose), 
                                           pb_robot.geometry.pose_from_tform(solved_pose))
        #robot.arm.SetJointValues(full_solved_q)
        total += (error < 0.002) and second_error
        #raw_input("Next?")
    print(total)

def main(): 
    utils.connect(use_gui=True)
    utils.disable_real_time()

    robot = pb_robot.panda.Panda() 
    objects_path = pb_robot.helper.getDirectory()
    floor_file = os.path.join(objects_path, 'furniture/short_floor.urdf')
    floor = pb_robot.body.createBody(floor_file)
    utils.set_default_camera()
    #utils.dump_world()

    robot.arm.SetJointValues([2, 0, 0, 0, 0, 0, 0])

    #pb_robot.viz.draw_pose(target_p, length=0.5, width=10)

    qs = [[0, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0]]

    IPython.embed()
    
    print('Quit?')
    utils.wait_for_user()
    utils.disconnect()

if __name__ == '__main__':
    main()
