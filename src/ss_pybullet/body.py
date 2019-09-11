import pybullet as p
from collections import namedtuple
from .utils import CLIENT, INFO_FROM_BODY
import ss_pybullet.utils as utils
import ss_pybullet.geometry as geometry

JOINT_TYPES = {
    p.JOINT_REVOLUTE: 'revolute', # 0
    p.JOINT_PRISMATIC: 'prismatic', # 1
    p.JOINT_SPHERICAL: 'spherical', # 2
    p.JOINT_PLANAR: 'planar', # 3
    p.JOINT_FIXED: 'fixed', # 4
    p.JOINT_POINT2POINT: 'point2point', # 5
    p.JOINT_GEAR: 'gear', # 6
}

class Body(object):
    def __init__(self, info):
        self.id = utils.load_model(info)
        self.BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])

    def get_info(self):
        return self.BodyInfo(*p.getBodyInfo(self.id, physicsClientId=CLIENT))

    def get_base_name(self):
        return self.get_info().base_name.decode(encoding='UTF-8')

    def get_body_name(self):
        return self.get_info().body_name.decode(encoding='UTF-8')

    def get_name(self):
        name = self.get_body_name()
        if name == '':
            name = 'body'
        return '{}{}'.format(name, int(self.id))

    def remove_body(self):
        if (CLIENT, self.id) in INFO_FROM_BODY:
            del INFO_FROM_BODY[CLIENT, self.id]
        return p.removeBody(self.id, physicsClientId=CLIENT)

    def get_pose(self):
        return p.getBasePositionAndOrientation(self.id, physicsClientId=CLIENT)

    def get_point(self):
        return self.get_pose()[0]

    def get_quat(self):
        return self.get_pose()[1] # [x,y,z,w]

    def get_euler(self):
        return geometry.euler_from_quat(self.get_quat())

    def get_base_values(self):
        return utils.base_values_from_pose(self.get_pose())

    def set_pose(self, pose):
        (point, quat) = pose
        p.resetBasePositionAndOrientation(self.id, point, quat, physicsClientId=CLIENT)

    def set_point(self, point):
        self.set_pose((point, self.get_quat()))

    def set_quat(self, quat):
        self.set_pose((self.get_point(), quat))

    def set_euler(self, euler):
        self.set_quat(geometry.quat_from_euler(euler))

    def set_base_values(self, values):
        _, _, z = self.get_point()
        x, y, theta = values
        self.set_point((x, y, z))
        self.set_quat(geometry.z_rotation(theta))

    def get_velocity(self):
        linear, angular = p.getBaseVelocity(self.id, physicsClientId=CLIENT)
        return linear, angular # [x,y,z], [wx,wy,wz]

    def set_velocity(self, linear=None, angular=None):
        if linear is not None:
            p.resetBaseVelocity(self.id, linearVelocity=linear, physicsClientId=CLIENT)
        if angular is not None:
            p.resetBaseVelocity(self.id, angularVelocity=angular, physicsClientId=CLIENT)

    def get_num_joints(self):
        return p.getNumJoints(self.id, physicsClientId=CLIENT)

    def get_joints(self):
        return list(range(self.get_num_joints()))

    def is_rigid_body(self):
        for joint in self.get_joints():
            if joint.is_movable():
                return False
        return True

    def is_fixed_base(self):
        return utils.get_mass(self.id) == STATIC_MASS

    def dump_body(self, body):
        #TODO
        print('Body id: {} | Name: {} | Rigid: {} | Fixed: {}'.format(
            self.id, self.get_body_name(), self.is_rigid_body(), self.is_fixed_base()))
        for joint in self.get_joints():
            if joint.is_movable():
                print('Joint id: {} | Name: {} | Type: {} | Circular: {} | Limits: {}'.format(
                    joint, get_joint_name(body, joint), JOINT_TYPES[get_joint_type(body, joint)],
                    is_circular(body, joint), get_joint_limits(body, joint)))
        link = -1
        print('Link id: {} | Name: {} | Mass: {} | Collision: {} | Visual: {}'.format(
            link, get_base_name(body), get_mass(body),
            len(get_collision_data(body, link)), -1)) # len(get_visual_data(body, link))))
        for link in get_links(body):
            joint = parent_joint_from_link(link)
            joint_name = JOINT_TYPES[get_joint_type(body, joint)] if is_fixed(body, joint) else get_joint_name(body, joint)
            print('Link id: {} | Name: {} | Joint: {} | Parent: {} | Mass: {} | Collision: {} | Visual: {}'.format(
                link, get_link_name(body, link), joint_name,
                get_link_name(body, get_link_parent(body, link)), get_mass(body, link),
                len(get_collision_data(body, link)), -1)) # len(get_visual_data(body, link))))
