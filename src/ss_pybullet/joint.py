import pybullet as p
from collections import namedtuple
from .utils import CLIENT, INFO_FROM_BODY
import ss_pybullet.utils as utils
import ss_pybullet.geometry as geometry

class Joint(object): # inherit what?
    def __init__(self, bodyID, jointID):
        self.bodyID = bodyID
        self.jointID = jointID
        self.JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])
        self.JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])

    def get_joint_info(self):
        return self.JointInfo(*p.getJointInfo(self.bodyID, self.jointID, physicsClientId=CLIENT))

    def get_joint_name(self):
        return self.get_joint_info().jointName # .decode('UTF-8')

    def get_joint_state(self):
        return self.JointState(*p.getJointState(self.bodyID, self.jointID, physicsClientId=CLIENT))

    def get_joint_position(self): 
        return self.get_joint_state().jointPosition

    def get_joint_velocity(self):
        return self.get_joint_state().jointVelocity

    def get_joint_reaction_force(self):
        return self.get_joint_state().jointReactionForces

    def get_joint_torque(self):
        return self.get_joint_state().appliedJointMotorTorque

    def get_joint_type(self):
        return self.get_joint_info().jointType

    def is_fixed(self):
        return self.get_joint_type() == p.JOINT_FIXED

    def is_movable(self):
        return not self.is_fixed()

    def is_circular(self):
        joint_info = self.get_joint_info()
        if joint_info.jointType == p.JOINT_FIXED:
            return False
        return joint_info.jointUpperLimit < joint_info.jointLowerLimit

    def get_joint_limits(self):
        if self.is_circular():
            # TODO: return UNBOUNDED_LIMITS
            return CIRCULAR_LIMITS
        joint_info = self.get_joint_info()
        return joint_info.jointLowerLimit, joint_info.jointUpperLimit

    def get_min_limit(self):
        # TODO: rename to min_position
        return self.get_joint_limits()[0]

    def get_max_limit(self):
        return self.get_joint_limits()[1]

    def get_max_velocity(self):
        return self.get_joint_info().jointMaxVelocity

    def get_max_force(self):
        return self.get_joint_info().jointMaxForce

    def get_joint_q_index(self):
        return self.get_joint_info().qIndex

    def get_joint_v_index(self):
        return self.get_joint_info().uIndex

    def get_joint_axis(self):
        return self.get_joint_info().jointAxis

    def get_joint_parent_frame(self):
        joint_info = self.get_joint_info()
        return joint_info.parentFramePos, joint_info.parentFrameOrn

    def set_joint_position(self, value):
        p.resetJointState(self.bodyID, self.jointID, value, targetVelocity=0, physicsClientId=CLIENT)

    def violates_limit(self, value):
        if self.is_circular():
            return False
        lower, upper = self.get_joint_limits()
        return (value < lower) or (upper < value)

    def wrap_position(self, position):
        if self.is_circular():
            return helper.wrap_angle(position)
        return position

#############
def get_joint_names(body, joints):
    return [get_joint_name(body, joint) for joint in joints]

def joints_from_names(body, names):
    return tuple(joint_from_name(body, name) for name in names)

def get_joint_positions(body, joints): # joints=None):
    return tuple(get_joint_position(body, joint) for joint in joints)

def get_joint_velocities(body, joints):
    return tuple(get_joint_velocity(body, joint) for joint in joints)

def wrap_positions(body, joints, positions):
    assert len(joints) == len(positions)
    return [wrap_position(body, joint, position)
            for joint, position in zip(joints, positions)]

def violates_limits(body, joints, values):
    return any(violates_limit(body, joint, value) for joint, value in zip(joints, values))

def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        set_joint_position(body, joint, value)

def get_configuration(body):
    return get_joint_positions(body, get_movable_joints(body))

def set_configuration(body, values):
    set_joint_positions(body, get_movable_joints(body), values)

def get_full_configuration(body):
    # Cannot alter fixed joints
    return get_joint_positions(body, get_joints(body))

def get_labeled_configuration(body):
    movable_joints = get_movable_joints(body)
    return dict(zip(get_joint_names(body, movable_joints),
                    get_joint_positions(body, movable_joints)))

def get_min_limits(body, joints):
    return [get_min_limit(body, joint) for joint in joints]

def get_max_limits(body, joints):
    return [get_max_limit(body, joint) for joint in joints]

def prune_fixed_joints(body, joints):
    return [joint for joint in joints if is_movable(body, joint)]

def get_movable_joints(body): # 45 / 87 on pr2
    return prune_fixed_joints(body, get_joints(body))

def movable_from_joints(body, joints):
    movable_from_original = {o: m for m, o in enumerate(get_movable_joints(body))}
    return [movable_from_original[joint] for joint in joints]

def joint_from_name(body, name):
    for joint in get_joints(body):
        if get_joint_name(body, joint) == name:
            return joint
    raise ValueError(body, name)

def has_joint(body, name):
    try:
        joint_from_name(body, name)
    except ValueError:
        return False
    return True

def joint_from_movable(body, index):
    return get_joints(body)[index]

def get_custom_limits(body, joints, custom_limits={}, circular_limits=UNBOUNDED_LIMITS):
    joint_limits = []
    for joint in joints:
        if joint in custom_limits:
            joint_limits.append(custom_limits[joint])
        elif is_circular(body, joint):
            joint_limits.append(circular_limits)
        else:
            joint_limits.append(get_joint_limits(body, joint))
    return zip(*joint_limits)
