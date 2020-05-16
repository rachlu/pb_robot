import pb_robot
import pb_robot.utils_noBase as utils

from og_util import add_fixed_constraint, remove_fixed_constraint, Attachment

class BodyPose(object):
    def __init__(self, body, pose=None):
        if pose is None:
            pose = body.get_pose()
        self.body = body
        self.pose = pose
    def assign(self):
        self.body.set_pose(self.pose)
        return self.pose
    def __repr__(self):
        return 'p{}'.format(id(self) % 1000)


class BodyGrasp(object):
    def __init__(self, body, grasp_worldF, grasp_objF, robot, link):
        self.body = body
        self.grasp_worldF = grasp_worldF #tform
        self.grasp_objF = grasp_objF #Tform
        self.robot = robot
        self.link = link
    def attachment(self):
        return Attachment(self.robot.id, self.link.linkID, self.grasp_pose, self.body.id)
    def assign(self):
        return self.attachment().assign()
    def __repr__(self):
        return 'g{}'.format(id(self) % 1000)


class BodyConf(object):
    def __init__(self, body, configuration=None, joints=None):
        if joints is None:
            joints = body.get_movable_joints()
        if configuration is None:
            configuration = body.arm.GetJointValues() 
        self.body = body
        self.joints = joints
        self.configuration = configuration
    def assign(self):
        #self.body.set_joint_positions(self.joints, self.configuration)
        self.body.arm.SetJointValues(self.configuration)
        return self.configuration
    def __repr__(self):
        return 'q{}'.format(id(self) % 1000)


class BodyPath(object):
    def __init__(self, body, path, joints=None, attachments=[]):
        if joints is None:
            joints = body.get_movable_joints()
        self.body = body
        self.path = path
        self.joints = joints
        self.attachments = attachments
    def bodies(self):
        return set([self.body] + [attachment.body for attachment in self.attachments])
    def iterator(self):
        for i, configuration in enumerate(self.path):
            self.body.arm.SetJointValues(configuration)
            for grasp in self.attachments:
                #grasp.assign()
                #TODO move this to only grasp once. Also, need to be able to release
                self.body.arm.Grab(grasp.body, pb_robot.geometry.tform_from_pose(grasp.grasp_pose))
            yield i
    def control(self, real_time=False, dt=0):
        # TODO: just waypoints
        if real_time:
            utils.enable_real_time()
        else:
            utils.disable_real_time()
        for values in self.path:
            for _ in utils.joint_controller(self.body, self.joints, values):
                utils.enable_gravity()
                if not real_time:
                    utils.step_simulation()
                time.sleep(dt)
    def refine(self, num_steps=0):
        return self.__class__(self.body, pb_robot.planning.refine_path(self.body, self.joints, self.path, num_steps), self.joints, self.attachments)
    def reverse(self):
        return self.__class__(self.body, self.path[::-1], self.joints, self.attachments)
    def __repr__(self):
        return '{}({},{},{},{})'.format(self.__class__.__name__, self.body, len(self.joints), len(self.path), len(self.attachments))

class ApplyForce(object):
    def __init__(self, body, robot, link):
        self.body = body
        self.robot = robot
        self.link = link
    def bodies(self):
        return {self.body, self.robot}
    def iterator(self, **kwargs):
        return []
    def refine(self, **kwargs):
        return self
    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.robot, self.body)

class Attach(ApplyForce):
    def control(self, **kwargs):
        # TODO: store the constraint_id?
        add_fixed_constraint(self.body, self.robot, self.link)
    def reverse(self):
        return Detach(self.body, self.robot, self.link)

class Detach(ApplyForce):
    def control(self, **kwargs):
        remove_fixed_constraint(self.body, self.robot, self.link)
    def reverse(self):
        return Attach(self.body, self.robot, self.link)

class Command(object):
    def __init__(self, body_paths):
        self.body_paths = body_paths

    def step(self):
        for i, body_path in enumerate(self.body_paths):
            for j in body_path.iterator():
                msg = '{},{}) step?'.format(i, j)
                raw_input(msg)
                #print(msg)

    def execute(self, time_step=0.05):
        for i, body_path in enumerate(self.body_paths):
            raw_input("next path?")
            for j in body_path.iterator():
                #time.sleep(time_step)
                utils.wait_for_duration(time_step)

    def control(self, real_time=False, dt=0): # TODO: real_time
        for body_path in self.body_paths:
            body_path.control(real_time=real_time, dt=dt)

    def refine(self, **kwargs):
        return self.__class__([body_path.refine(**kwargs) for body_path in self.body_paths])

    def reverse(self):
        return self.__class__([body_path.reverse() for body_path in reversed(self.body_paths)])

    def __repr__(self):
        return 'c{}'.format(id(self) % 1000)
