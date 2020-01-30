from collections import namedtuple
import pb_robot.geometry as geometry
import pybullet as p

CLIENT = 0
BASE_LINK = -1

# Constraints - applies forces when not satisfied

def get_constraints():
    """
    getConstraintUniqueId will take a serial index in range 0..getNumConstraints,  and reports the constraint unique id.
    Note that the constraint unique ids may not be contiguous, since you may remove constraints.
    """
    return [p.getConstraintUniqueId(i, physicsClientId=CLIENT)
            for i in range(p.getNumConstraints(physicsClientId=CLIENT))]

def remove_constraint(constraint):
    p.removeConstraint(constraint, physicsClientId=CLIENT)

ConstraintInfo = namedtuple('ConstraintInfo', ['parentBodyUniqueId', 'parentJointIndex',
                                               'childBodyUniqueId', 'childLinkIndex', 'constraintType',
                                               'jointAxis', 'jointPivotInParent', 'jointPivotInChild',
                                               'jointFrameOrientationParent', 'jointFrameOrientationChild', 'maxAppliedForce'])

def get_constraint_info(constraint): # getConstraintState
    # TODO: four additional arguments
    return ConstraintInfo(*p.getConstraintInfo(constraint, physicsClientId=CLIENT)[:11])

def get_fixed_constraints():
    fixed_constraints = []
    for constraint in get_constraints():
        constraint_info = get_constraint_info(constraint)
        if constraint_info.constraintType == p.JOINT_FIXED:
            fixed_constraints.append(constraint)
    return fixed_constraints

def add_fixed_constraint(body, robot, robot_link, max_force=None):
    body_link = BASE_LINK
    body_pose = body.get_pose()
    end_effector_pose = robot_link.get_com_pose() 
    grasp_pose = geometry.multiply(geometry.invert(end_effector_pose), body_pose)
    grasp_point, grasp_quat = grasp_pose
    constraint = p.createConstraint(robot, robot_link, body, body_link,
                                    p.JOINT_FIXED, jointAxis=geometry.unit_point(),
                                    parentFramePosition=grasp_point,
                                    childFramePosition=geometry.unit_point(),
                                    parentFrameOrientation=grasp_quat,
                                    childFrameOrientation=geometry.unit_quat(),
                                    physicsClientId=CLIENT)
    if max_force is not None:
        p.changeConstraint(constraint, maxForce=max_force, physicsClientId=CLIENT)
    return constraint

def Grab(robot, objectBody, grasp_tform, max_force=None):
    # this only holds effective if you step through dynamics
    body_link = BASE_LINK
    robot_link = robot.arm.eeFrame.linkID
    grasp_pose = geometry.pose_from_tform(grasp_tform)
    grasp_point, grasp_quat = grasp_pose
    constraint = p.createConstraint(robot.id, robot_link, objectBody.id, body_link,  
                                    p.JOINT_FIXED, jointAxis=geometry.unit_point(),
                                    parentFramePosition=grasp_point,
                                    childFramePosition=geometry.unit_point(),
                                    parentFrameOrientation=grasp_quat,
                                    childFrameOrientation=geometry.unit_quat(),
                                    physicsClientId=CLIENT)
    if max_force is not None:
        p.changeConstraint(constraint, maxForce=max_force, physicsClientId=CLIENT)
    return constraint


def remove_fixed_constraint(body, robot, robot_link):
    for constraint in get_fixed_constraints():
        constraint_info = get_constraint_info(constraint)
        if (body == constraint_info.childBodyUniqueId) and \
                (BASE_LINK == constraint_info.childLinkIndex) and \
                (robot == constraint_info.parentBodyUniqueId) and \
                (robot_link == constraint_info.parentJointIndex):
            remove_constraint(constraint)

