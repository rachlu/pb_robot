from scipy import optimize
import numpy
import random
import time
import pybullet as p
import pb_robot
import pb_robot.utils_noBase as utils
import geometry
import body
import helper

from pb_robot.ikfast.ikfast import closest_inverse_kinematics, ikfast_inverse_kinematics

class Panda(body.Body):
    def __init__(self, **kwargs):
        self.urdf_file = 'models/franka_description/robots/panda_arm_hand.urdf'
        #self.urdf_file = 'models/franka_description/robots/panda_arm.urdf'

        with helper.HideOutput(): 
            with utils.LockRenderer():
                self.id = utils.load_model(self.urdf_file, fixed_base=True)
        body.Body.__init__(self, self.id)

        self.arm_joint_names = ['panda_joint{}'.format(i) for i in xrange(1, 8)]
        self.arm_joints = [self.joint_from_name(n) for n in self.arm_joint_names]
        self.arm = PandaArm(self.id, self.arm_joints, 'panda_hand')


class PandaArm(object):
    def __init__(self, bodyID, joints, handName):
        self.__robot = body.Body(bodyID)
        self.joints = joints
        self.jointsID = [j.jointID for j in self.joints]
        self.eeFrame = self.__robot.link_from_name(handName)
        self.hand = PandaHand(bodyID, 'panda_finger_joint1', 'panda_finger_joint2')
        self.torque_limits = [87, 87, 87, 87, 12, 12, 12]
        self.startq =  [0, -numpy.pi/4.0, 0, -0.75*numpy.pi, 0, numpy.pi/2.0, numpy.pi/4.0]

        self.grabbedRelations = dict()
        self.grabbedObjects = dict()
        self.ik_info = pb_robot.ikfast.utils.IKFastInfo(module_name='franka_panda.ikfast_panda_arm', 
                          base_link='panda_link0',
                          ee_link='panda_link8', 
                          free_joints=['panda_joint7'])

        self.SetJointValues(self.startq) 

    def GetJointValues(self):
        return self.__robot.get_joint_positions(self.joints)
    
    def SetJointValues(self, q):
        self.__robot.set_joint_positions(self.joints, q)

        #If exists grabbed object, update its position too
        if len(self.grabbedObjects.keys()) > 0:
            hand_worldF = self.GetEETransform()
            for i in self.grabbedObjects.keys():
                obj = self.grabbedObjects[i]
                grasp_objF = self.grabbedRelations[i]
                obj_worldF = numpy.dot(hand_worldF, numpy.linalg.inv(grasp_objF))
                obj.set_transform(obj_worldF)

    def GetJointLimits(self):
        return (self.__robot.get_min_limits(self.joints), 
                self.__robot.get_max_limits(self.joints))

    def Grab(self, body, relation):
        self.grabbedRelations[body.get_name()] = relation
        self.grabbedObjects[body.get_name()] = body

    def Release(self, body):
        self.grabbedObjects.pop(body.get_name(), None)
        self.grabbedRelations.pop(body.get_name(), None)

    def GetEETransform(self):
        return geometry.tform_from_pose(self.eeFrame.get_link_pose())

    def ComputeFK(self, q):
        old_q = self.GetJointValues()
        self.SetJointValues(q)
        pose = self.GetEETransform()
        self.SetJointValues(old_q)
        return pose 

    def randomConfiguration(self):
        (lower, upper) = self.GetJointLimits()
        dofs = numpy.zeros(len(lower))
        for i in xrange(len(lower)):
            dofs[i] = random.uniform(lower[i], upper[i])
        return dofs

    def ComputeIK(self, transform, seed_q=None):
        pose = geometry.pose_from_tform(transform)
        q = next(ikfast_inverse_kinematics(self.__robot.id, self.ik_info, 
                                           self.eeFrame.linkID, pose, max_time=0.05), None)
        #q = next(closest_inverse_kinematics(self.__robot.id, self.ik_info, self.hand.linkID,
        #                                        pose, max_distance=0.05, max_time=0.05), None)
        return q 

    def IsCollisionFree(self, q, self_collisions=True):
        # This is to cover that the collision function sets joints, but not using the arm version
        oldq = self.GetJointValues()
        self.SetJointValues(oldq)

        obstacles = [b.id for b in utils.get_bodies() if 'panda' not in b.get_name() and b.get_name() not in self.grabbedObjects.keys()]
        attachments = [g.id for g in self.grabbedObjects.values()]
        collisionfn = pb_robot.collisions.get_collision_fn(self.__robot.id, self.jointsID, obstacles, 
                                                         attachments, self_collisions)

        # Restore configuration
        self.SetJointValues(oldq)
        return not collisionfn(q)

    def ExecutePath(self, path, timestep=0.05):
        # Will need to generalize with more control methods. This copies from Caelan's command 
        for i in xrange(len(path)):
            self.SetJointValues(path[i])
            time.sleep(timestep)

    def GetJacobian(self, q): 
        # Must include finger joints and then remove them
        allq = numpy.append(q, [0, 0]).tolist()
        (translate, rotate) = pb_robot.planning.compute_jacobian(self.__robot, self.eeFrame, positions=allq)
        t = numpy.array(translate)
        r = numpy.array(rotate)
        jacobian = numpy.hstack((translate, rotate))
        return numpy.transpose(jacobian[0:7, :])

    def InsideTorqueLimits(self, q, forces):
        '''Check if configuration is within torque limits, given force
        @param q Configuration
        @param force 6D array of force to check against
        @return check True if within torque limits '''
        jacobian = self.GetJacobian(q)
        torques = numpy.dot(numpy.transpose(jacobian), forces)
        inside = all(numpy.less(abs(torques), self.torque_limits)) # Assuming symmetric
        return inside

class PandaHand(object):
    def __init__(self, bodyID, left_finger_name, right_finger_name):
        self.__robot = body.Body(bodyID)
        self.left_finger = self.__robot.joint_from_name(left_finger_name)
        self.right_finger = self.__robot.joint_from_name(right_finger_name)

    def Open(self):
        self.left_finger.set_joint_position(0.04)
        self.right_finger.set_joint_position(0.04)

    def Close(self):
        self.left_finger.set_joint_position(0)
        self.right_finger.set_joint_position(0)

    def MoveTo(self, distance):
        if not (0 <= distance <= 0.08):
            raise IOError("Invalid distance request. The value must be between 0 and 0.08")
        finger_distance = distance / 2.0
        self.left_finger.set_joint_position(finger_distance)
        self.right_finger.set_joint_position(finger_distance)

    def GetJointPositions(self):
        return (self.left_finger.get_joint_position(), self.right_finger.get_joint_position())
