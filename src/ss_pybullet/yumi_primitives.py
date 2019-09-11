
from .pr2_utils import get_top_grasps
from .utils import INF, GraspInfo
from .primitives import link_from_name, get_body_name, BodyGrasp
from .geometry import Pose, Point
GRASP_INFO = {}


GRASP_INFO = {
    'top': GraspInfo(lambda body: get_top_grasps(body, under=True, tool_pose=Pose(), max_width=INF, grasp_length=0),
                     approach_pose=Pose(0.1*Point(z=1))),
}


ARM_NAMES = ('left', 'right')

def arm_from_arm(arm): # TODO: rename
    assert (arm in ARM_NAMES)
    return '{}_arm'.format(arm)

def gripper_from_arm(arm):
    assert (arm in ARM_NAMES)
    return '{}_gripper'.format(arm)


YUMI_GROUPS = {
    arm_from_arm('left'): ['yumi_joint_1_l', 'yumi_joint_2_l', 'yumi_joint_7_l', 'yumi_joint_3_l', 
                           'yumi_joint_4_l', 'yumi_joint_5_l', 'yumi_joint_6_l'],
    arm_from_arm('right'): ['yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_7_r', 'yumi_joint_3_r', 
                            'yumi_joint_4_r', 'yumi_joint_5_r', 'yumi_joint_6_r'], 
    gripper_from_arm('left'): ['gripper_l_joint', 'gripper_l_joint_m'],
    gripper_from_arm('right'): ['gripper_r_joint', 'gripper_r_joint_m'],
}

TOOL_FRAMES = {
    'yumi': 'gripper_r_base',
    #'left': 'l_gripper_tool_frame',  # l_gripper_palm_link | l_gripper_tool_frame
    #'right': 'r_gripper_tool_frame',  # r_gripper_palm_link | r_gripper_tool_frame
}

YUMI_GRIPPER_ROOTS = {
    'left': 'gripper_r_base',
    'right': 'gripper_l_base',
}

YUMI_BASE_LINK = 'yumi_body'


def get_grasp_gen(robot, grasp_name):
    grasp_info = GRASP_INFO[grasp_name]
    end_effector_link = link_from_name(robot, TOOL_FRAMES[get_body_name(robot)])
    def gen(body):
        grasp_poses = grasp_info.get_grasps(body)
        for grasp_pose in grasp_poses:
            body_grasp = BodyGrasp(body, grasp_pose, grasp_info.approach_pose,
                                   robot, end_effector_link)
            yield (body_grasp,)
    return gen
