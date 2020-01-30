from __future__ import print_function
from collections import namedtuple
from itertools import product
import numpy as np
import pybullet as p

from .utils import CLIENT, BASE_LINK
from .utils import step_simulation
from .kinematics import get_bodies, get_all_links

MAX_DISTANCE = 0.

# object_unique_id and linkIndex seem to be noise
CollisionShapeData = namedtuple('CollisionShapeData', ['object_unique_id', 'linkIndex',
                                                       'geometry_type', 'dimensions', 'filename',
                                                       'local_frame_pos', 'local_frame_orn'])
ContactResult = namedtuple('ContactResult', ['contactFlag', 'bodyUniqueIdA', 'bodyUniqueIdB',
                                             'linkIndexA', 'linkIndexB', 'positionOnA', 'positionOnB',
                                             'contactNormalOnB', 'contactDistance', 'normalForce'])
Ray = namedtuple('Ray', ['start', 'end'])
RayResult = namedtuple('RayResult', ['objectUniqueId', 'linkIndex',
                                     'hit_fraction', 'hit_position', 'hit_normal'])
#####################################


def contact_collision():
    step_simulation()
    return len(p.getContactPoints(physicsClientId=CLIENT)) != 0

#####################################

def get_ray(ray):
    start, end = ray
    return np.array(end) - np.array(start)

def ray_collision(ray):
    # TODO: be careful to disable gravity and set static masses for everything
    step_simulation() # Needed for some reason
    start, end = ray
    result, = p.rayTest(start, end, physicsClientId=CLIENT)
    # TODO: assign hit_position to be the end?
    return RayResult(*result)

def batch_ray_collision(rays, threads=1):
    assert 1 <= threads <= p.MAX_RAY_INTERSECTION_BATCH_SIZE
    if not rays:
        return []
    step_simulation() # Needed for some reason
    ray_starts = [start for start, _ in rays]
    ray_ends = [end for _, end in rays]
    return [RayResult(*tup) for tup in p.rayTestBatch(
        ray_starts, ray_ends,
        numThreads=threads,
        #parentObjectUniqueId=
        #parentLinkIndex=
        physicsClientId=CLIENT)]

#####################################

def collision_shape_from_data(data, body, link, client=None):
    client = get_client(client)
    if (data.geometry_type == p.GEOM_MESH) and (data.filename == UNKNOWN_FILE):
        return -1
    pose = multiply(get_joint_inertial_pose(body, link), get_data_pose(data))
    point, quat = pose
    # TODO: the visual data seems affected by the collision data
    return p.createCollisionShape(shapeType=data.geometry_type,
                                  radius=get_data_radius(data),
                                  # halfExtents=get_data_extents(data.geometry_type, data.dimensions),
                                  halfExtents=np.array(get_data_extents(data)) / 2,
                                  height=get_data_height(data),
                                  fileName=data.filename.decode(encoding='UTF-8'),
                                  meshScale=get_data_scale(data),
                                  planeNormal=get_data_normal(data),
                                  flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                                  collisionFramePosition=point,
                                  collisionFrameOrientation=quat,
                                  physicsClientId=client)
    #return p.createCollisionShapeArray()

def get_collision_data(body, link=BASE_LINK):
    # TODO: try catch
    return [CollisionShapeData(*tup) for tup in p.getCollisionShapeData(body, link, physicsClientId=CLIENT)]

def get_data_type(data):
    return data.geometry_type if isinstance(data, CollisionShapeData) else data.visualGeometryType

def get_data_filename(data):
    return (data.filename if isinstance(data, CollisionShapeData)
            else data.meshAssetFileName).decode(encoding='UTF-8')

def get_data_pose(data):
    if isinstance(data, CollisionShapeData):
        return (data.local_frame_pos, data.local_frame_orn)
    return (data.localVisualFrame_position, data.localVisualFrame_orientation)
