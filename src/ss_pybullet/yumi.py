from collections import defaultdict, deque, namedtuple
import numpy as np
import pybullet as p
import ss_pybullet.utils_noBase as utils
import primitives
import geometry
from .utils_noBase import GraspInfo
from .body import Body
from .pr2_utils import get_top_grasps

#TODO generalize such that this is robot class, store robot information separately

class Yumi(Body):
    def __init__(self, **kwargs):
        self.urdf_file = 'models/yumi_description/yumi.urdf'
        Body.__init__(self, self.urdf_file, **kwargs)

        # TODO pull this information from like a yaml or setup file
        self.arm_names = ('left_arm', 'right_arm')
        self.left_arm = ['yumi_joint_1_l', 'yumi_joint_2_l', 'yumi_joint_7_l', 'yumi_joint_3_l', 'yumi_joint_4_l', 'yumi_joint_5_l', 'yumi_joint_6_l']
        self.right_arm_joint_names = ['yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_7_r', 'yumi_joint_3_r', 'yumi_joint_4_r', 'yumi_joint_5_r', 'yumi_joint_6_r']
        self.right_hand_link_name = 'gripper_r_base'
        self.left_hand_link_name = 'gripper_l_base'
        self.right_hand = self.link_from_name(self.right_hand_link_name)
        self.left_hand = self.link_from_name(self.left_hand_link_name)

        self.grasp_info = {
            'top': GraspInfo(lambda body: get_top_grasps(body, under=True, tool_pose=geometry.Pose(), max_width=np.inf, grasp_length=0),
                             approach_pose=geometry.Pose(0.1*geometry.Point(z=1))),
        }

        
    def ik(self):
        # call ik but with left or right hand as the link?
        return 5

    def get_grasp_gen(self, hand, grasp_name):
        grasp_info = self.grasp_info[grasp_name]
        def gen(body):
            grasp_poses = grasp_info.get_grasps(body)
            for grasp_pose in grasp_poses:
                body_grasp = primitives.BodyGrasp(body, grasp_pose, grasp_info.approach_pose,
                                                  self, hand)
                yield (body_grasp,)
        return gen

