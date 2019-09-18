#!/usr/bin/env python

from __future__ import print_function

import IPython
import ss_pybullet
import ss_pybullet.utils_noBase as utils
import ss_pybullet.viz as viz
import ss_pybullet.placements as placements
from ss_pybullet.yumi_primitives import get_grasp_gen
from ss_pybullet.primitives import BodyPose, BodyConf, Command, \
    get_ik_fn, get_free_motion_gen, get_holding_motion_gen
from ss_pybullet.utils_noBase import WorldSaver, enable_gravity, connect, dump_world, \
    set_default_camera, YUMI_URDF, \
    BLOCK_URDF, wait_for_user, disconnect, user_input, update_state, disable_real_time
from ss_pybullet.geometry import Pose, Point

def plan(robot, block, fixed, teleport):
    grasp_gen = get_grasp_gen(robot, 'top')
    ik_fn = get_ik_fn(robot, fixed=fixed, teleport=teleport)
    free_motion_fn = get_free_motion_gen(robot, fixed=([block] + fixed), teleport=teleport)
    holding_motion_fn = get_holding_motion_gen(robot, fixed=fixed, teleport=teleport)

    pose0 = BodyPose(block)
    conf0 = BodyConf(robot)
    saved_world = WorldSaver()
    for grasp, in grasp_gen(block):
        saved_world.restore()
        result1 = ik_fn(block, pose0, grasp)
        if result1 is None:
            continue
        conf1, path2 = result1
        pose0.assign()
        result2 = free_motion_fn(conf0, conf1)
        if result2 is None:
            continue
        path1, = result2
        result3 = holding_motion_fn(conf1, conf0, block, grasp)
        if result3 is None:
            continue
        path3, = result3
        return Command(path1.body_paths + path2.body_paths + path3.body_paths)
    return None

def main(display='execute'): # control | execute | step
    connect(use_gui=True)
    disable_real_time()
    #robot = load_model(YUMI_URDF) 
    #floor = load_model('models/short_floor.urdf')
    #block = load_model(BLOCK_URDF, fixed_base=False)

    yumi = ss_pybullet.body.Body(YUMI_URDF)
    floor = ss_pybullet.body.Body('models/short_floor.urdf')
    block = ss_pybullet.body.Body(BLOCK_URDF, fixed_base=False)
    block.set_pose(Pose(Point(y=0., x=0.5, z=placements.stable_z(block, floor))))
    set_default_camera()
    #dump_world()

    right_hand = yumi.link_from_name('gripper_r_base') # definition would go in robot.py
    current_t = right_hand.get_link_pose()
    new_p = (0.58, 0.0, 0.515) 
    target_p = (new_p, current_t[1])
    viz.draw_pose(target_p, length=0.5, width=10)
    #f = utils.inverse_kinematics(yumi, right_hand, target_p)

    IPython.embed()

    saved_world = WorldSaver()
    #command = plan(robot, block, fixed=[floor], teleport=False)
    #if (command is None) or (display is None):
    #    print('Unable to find a plan!')
    #    return

    saved_world.restore()
    '''update_state()
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
