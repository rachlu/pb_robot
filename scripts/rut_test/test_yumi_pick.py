#!/usr/bin/env python

from __future__ import print_function

import os
import IPython
import ss_pybullet
import ss_pybullet.utils_noBase as utils
import ss_pybullet.viz as viz
import ss_pybullet.placements as placements
import ss_pybullet.geometry as geometry
from ss_pybullet.primitives import BodyPose, BodyConf, BodyPath, Command, \
    get_ik_fn, get_free_motion_gen, get_holding_motion_gen

from ss_pybullet.utils_noBase import WorldSaver, enable_gravity, connect, dump_world, \
    set_default_camera, wait_for_user, disconnect, user_input, update_state, disable_real_time
from ss_pybullet.geometry import Pose, Point
import numpy
import time
import util
import planner

def getDirectory():
    '''Get the file path for the location of kinbody
    @return object_path (string) Path to objects folder'''
    from catkin.find_in_workspaces import find_in_workspaces
    package_name = 'mcube_objects'
    directory = 'data'
    objects_path = find_in_workspaces(
        search_dirs=['share'],
        project=package_name,
        path=directory,
        first_match_only=True)
    if len(objects_path) == 0:
        raise RuntimeError('Can\'t find directory {}/{}'.format(
            package_name, directory))
    else:
        objects_path = objects_path[0]
    return objects_path

def PlanGraspPaths(robot, start_q, end_q):
    rrt = planner.ConstrainedPlanner(robot)
    path_j = rrt.PlanToConfiguration(robot.right_arm, start_q, end_q)
    
    if path_j is None: print('Couldnt find path')
    else: print('Found path')
    raw_input("Go to next?")

    end_pose = robot.right_arm.ComputeFK(end_q)
    if end_pose is None:
        raw_input("No ik. Terminating")
    viz.draw_pose(geometry.pose_from_tform(end_pose), length=0.5, width=10)
    raw_input("Go to next?")
    path_p = rrt.PlanToEndEffectorPose(robot.right_arm, start_q, end_pose)

    print(path_j)
    print(path_p)
    raw_input("Continue?")
    return path_j

def plan(robot, block, fixed, teleport):
    grasp_gen = robot.get_grasp_gen(robot.right_hand, 'top')
    ik_fn = get_ik_fn(robot, fixed=fixed, teleport=teleport)
    free_motion_fn = get_free_motion_gen(robot, fixed=([block] + fixed), teleport=teleport)
    holding_motion_fn = get_holding_motion_gen(robot, fixed=fixed, teleport=teleport)

    pose0 = BodyPose(block)
    conf0 = BodyConf(robot)
    saved_world = WorldSaver()  
    for grasp, in grasp_gen(block):
        saved_world.restore()
        result1 = ik_fn(block, pose0, grasp)
        print('Result 1')
        print(result1)
        if result1 is None:
            continue
        conf1, path2 = result1
        pose0.assign()
        result2 = free_motion_fn(conf0, conf1)
        print('Result 2 ') 
        print(result2)
        if result2 is None:
            continue
        path1, = result2
        result3 = holding_motion_fn(conf1, conf0, block, grasp)
        print('Result 3 ')
        print(result3)
        if result3 is None:
            continue
        path3, = result3
        return Command(path1.body_paths + path2.body_paths + path3.body_paths)
    return None

def main(display='execute'): # control | execute | step
    connect(use_gui=True)
    disable_real_time()

    yumi = ss_pybullet.yumi.Yumi()
    objects_path = getDirectory()
    floor_file = os.path.join(objects_path, 'furniture/short_floor.urdf')
    block_file = os.path.join(objects_path, 'objects/block_for_pick_and_place_mid_size.urdf')
    floor = ss_pybullet.body.createBody(floor_file)
    block = ss_pybullet.body.createBody(block_file, fixed_base=False)
    block.set_pose(Pose(Point(y=0., x=0.5, z=placements.stable_z(block, floor))))
    set_default_camera()
    dump_world()

    # Move left arm
    t = [1, 1, 1, 1, 1, 1, 1]
    yumi.left_arm.SetJointValues(t)

    current_t = yumi.right_hand.get_link_pose()
    #viz.draw_pose(current_t, length=0.5, width=10)

    start_q = yumi.right_arm.GetJointValues()
    end_q = start_q + numpy.random.random(7)*0.1
    path = PlanGraspPaths(yumi, start_q, end_q)
    command = Command([BodyPath(yumi, path, joints=yumi.right_arm.joints)]) #TODO unless the input is joint names?
    raw_input("Execute?")
    command.execute(time_step=0.005)


    '''
    saved_world = WorldSaver()
    command = plan(yumi, block, fixed=[floor], teleport=False)
    IPython.embed()
    if (command is None) or (display is None):
        print('Unable to find a plan!')
        return

    #saved_world.restore()
    update_state()
    user_input('{}?'.format(display))
    if display == 'control':
        enable_gravity()
        command.control(real_time=False, dt=0)
    elif display == 'execute':
        command.refine(num_steps=10).execute(time_step=0.005)
    elif display == 'step':
        command.step()
    else:
        raise ValueError(display)
    '''
    print('Quit?')
    wait_for_user()
    disconnect()

if __name__ == '__main__':
    main()
