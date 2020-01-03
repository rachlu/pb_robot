#/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Various utility functions for constrained tool manipulation
'''

import numpy
import random
from tsr import TSR, TSRChain

def generatePath(i): return i #TODO replace later

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

def SampleTSRForPose(tsr_chain):
    '''Shortcutting function for randomly samping a 
    pose from a tsr chain
    @param tsr_chain Chain to sample from
    @return ee_pose Pose from tsr chain'''
    tsr_idx = random.randint(0, len(tsr_chain)-1)
    sampled_tsr = tsr_chain[tsr_idx]
    ee_pose = sampled_tsr.sample()
    return ee_pose

def CreateTSRFromPose(manip, pose):
    '''Create a TSR that, when sampled, produces one pose. This 
    simply creates a common interface for the planner
    @param manip Manipulator to use use (required for tsr)
    @param pose 4x4 transform to center TSR on
    @return tsr_chain chain with single pose TSR'''
    #manipulator_index = manip.GetManipulatorIndex()
    goal_tsr = TSR(T0_w=pose) #, manipindex=manipulator_index)
    tsr_chain = TSRChain(sample_goal=True, TSR=goal_tsr)
    return tsr_chain

def actionPath_hand(tool_path, grasp_toolF):
    '''Given a series of waypoints of the tool in the world frame we
    create a path for the hand in the world frame by transforming 
    using the grasp
    @param tool_path Series of waypoints of tool in world frame
    @param grasp_toolF Pose of the end effector when grasping in the tool frame
    @return hand_pose Series of waypoints of the hand in the world frame'''
    return numpy.array([numpy.dot(tool_path[i], grasp_toolF) for i in xrange(len(tool_path))])

