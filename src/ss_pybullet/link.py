from collections import namedtuple
import pybullet as p
import ss_pybullet.utils_noBase as utils
import ss_pybullet.geometry as geometry

class Link(object):
    def __init__(self, body, linkID):
        self.body = body
        self.bodyID = self.body.id
        self.linkID = linkID
        self.parentJointID = linkID # Parent joint and link have same id
        if linkID == -1:
            self.parentJoint = None
        else:
            self.parentJoint = self.body.joints[linkID]
        # have a get joint from id? returns the class? 
        self.base_link = body.base_link

        self.LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                                  'localInertialFramePosition', 'localInertialFrameOrientation',
                                                  'worldLinkFramePosition', 'worldLinkFrameOrientation'])

        #parent_link_from_joint = get_link_parent

    def get_link_name(self):  
        if self.linkID == self.base_link:
            return self.body.get_base_name()
        return self.parentJoint.get_joint_info().linkName.decode('UTF-8')

    def get_link_parent(self):
        if self.linkID == self.base_link:
            return None
        return self.parentJoint.get_joint_info().parentIndex

    def get_link_state(self, kinematics=True, velocity=True):
        # TODO: the defaults are set to False?
        # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c
        return self.LinkState(*p.getLinkState(self.body.id, self.linkID, 
                                              #computeLinkVelocity=velocity, 
                                              #computeForwardKinematics=kinematics,
                                              physicsClientId=utils.CLIENT))

    def get_com_pose(self): # COM = center of mass
        link_state = self.get_link_state()
        return link_state.linkWorldPosition, link_state.linkWorldOrientation

    def get_link_inertial_pose(self):
        link_state = self.get_link_state()
        return link_state.localInertialFramePosition, link_state.localInertialFrameOrientation

    def get_link_pose(self):
        if self.linkID == self.base_link:
            return self.body.get_pose()
        # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
        link_state = self.get_link_state() #, kinematics=True, velocity=False)
        return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation

    def get_link_children(self):
        # Untested
        children = self.body.get_all_link_children() 
        return children.get(self, []) 

    def get_link_ancestors(self):
        # Untested
        parent = self.get_link_parent()
        print parent
        if parent is None:
            return []
        return parent.get_link_ancestors() + [parent.id]

    def get_joint_ancestors(self): 
        # Untested
        return self.get_link_ancestors() + [self]

    def get_movable_joint_ancestors(self):
        # Untested
        return self.body.prune_fixed_joints(self.get_joint_ancestors())

    def get_link_descendants(self, test=lambda l: True):
        # Untested 
        descendants = []
        for child in self.get_link_children():
            if test(child):
                descendants.append(child)
                descendants.extend(child.get_link_descendants(test=test))
        return descendants

    def get_link_subtree(self, **kwargs):
        # Untested
        return [self] + self.get_link_descendants(**kwargs)

    def get_local_link_pose(self): #XXX test
        parent_joint = self.body.parent_link_from_joint(self.jointID)

        #world_child = get_link_pose(body, joint)
        #world_parent = get_link_pose(body, parent_joint)
        ##return geometry.multiply(geometry.invert(world_parent), world_child)
        #return geometry.multiply(world_child, geometry.geometry.invert(world_parent))

        # https://github.com/bulletphysics/bullet3/blob/9c9ac6cba8118544808889664326fd6f06d9eeba/examples/pybullet/gym/pybullet_utils/urdfEditor.py#L169
        parent_com = self.body.get_joint_parent_frame(self.jointID)
        tmp_pose = geometry.invert(geometry.multiply(self.body.get_joint_inertial_pose(self.jointID), parent_com))
        parent_inertia = self.body.get_joint_inertial_pose(parent_joint)
        #return geometry.multiply(parent_inertia, tmp_pose) # TODO: why is this wrong...
        _, orn = geometry.multiply(parent_inertia, tmp_pose)
        pos, _ = geometry.multiply(parent_inertia, geometry.Pose(parent_com[0]))
        return (pos, orn)


