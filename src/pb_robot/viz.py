import math
from itertools import product, combinations
import pybullet as p
import numpy as np
import pb_robot.geometry as geometry
import pb_robot.utils_noBase as utils

RED = (1, 0, 0, 1)
GREEN = (0, 1, 0, 1)
BLUE = (0, 0, 1, 1)
BLACK = (0, 0, 0, 1)
WHITE = (1, 1, 1, 1)
BROWN = (0.396, 0.263, 0.129, 1)
TAN = (0.824, 0.706, 0.549, 1)
GREY = (0.5, 0.5, 0.5, 1)
YELLOW = (1, 1, 0, 1)

COLOR_FROM_NAME = {
    'red': RED,
    'green': GREEN,
    'blue': BLUE,
    'white': WHITE,
    'grey': GREY,
    'black': BLACK,
}

def get_lifetime(lifetime):
    if lifetime is None:
        return 0
    return lifetime

def add_debug_parameter():
    # TODO: make a slider that controls the step in the trajectory
    # TODO: could store a list of savers
    #targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -10, 10, 0)
    #maxForce = p.readUserDebugParameter(maxForceSlider)
    raise NotImplementedError()

def add_text(text, position=(0, 0, 0), color=(0, 0, 0), lifetime=None, parent=-1, parent_link=utils.BASE_LINK):
    return p.addUserDebugText(str(text), textPosition=position, textColorRGB=color[:3], # textSize=1,
                              lifeTime=get_lifetime(lifetime), parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=utils.CLIENT)

def add_line(start, end, color=(0, 0, 0), width=1, lifetime=None, parent=-1, parent_link=utils.BASE_LINK):
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width,
                              lifeTime=get_lifetime(lifetime), parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=utils.CLIENT)

def remove_debug(debug):
    p.removeUserDebugItem(debug, physicsClientId=utils.CLIENT)

remove_handle = remove_debug

def remove_handles(handles):
    for handle in handles:
        remove_debug(handle)

def remove_all_debug():
    p.removeAllUserDebugItems(physicsClientId=utils.CLIENT)

def add_body_name(body, name=None, **kwargs):
    if name is None:
        name = body.get_name()
    with utils.PoseSaver(body):
        body.set_pose(geometry.unit_pose())
        lower, upper = utils.get_aabb(body)
    #position = (0, 0, upper[2])
    position = upper
    return add_text(name, position=position, parent=body, **kwargs)  # removeUserDebugItem

def add_segments(points, closed=False, **kwargs):
    lines = []
    for v1, v2 in zip(points, points[1:]):
        lines.append(add_line(v1, v2, **kwargs))
    if closed:
        lines.append(add_line(points[-1], points[0], **kwargs))
    return lines

def draw_link_name(body, link=utils.BASE_LINK):
    return add_text(link.get_link_name(), position=(0, 0.2, 0),
                    parent=body, parent_link=link)

def draw_pose(pose, length=0.1, **kwargs):
    origin_world = geometry.tform_point(pose, np.zeros(3))
    handles = []
    for k in range(3):
        axis = np.zeros(3)
        axis[k] = 1
        axis_world = geometry.tform_point(pose, length*axis)
        handles.append(add_line(origin_world, axis_world, color=axis, **kwargs))
    return handles

def draw_base_limits(limits, z=1e-2, **kwargs):
    lower, upper = limits
    vertices = [(lower[0], lower[1], z), (lower[0], upper[1], z),
                (upper[0], upper[1], z), (upper[0], lower[1], z)]
    return add_segments(vertices, closed=True, **kwargs)

def draw_circle(center, radius, n=24, **kwargs):
    vertices = []
    for i in range(n):
        theta = i*2*math.pi/n
        unit = np.append(geometry.unit_from_theta(theta), [0])
        vertices.append(center+radius*unit)
    return add_segments(vertices, closed=True, **kwargs)

def draw_aabb(aabb, **kwargs):
    d = len(aabb[0])
    vertices = list(product(range(len(aabb)), repeat=d))
    lines = []
    for i1, i2 in combinations(vertices, 2):
        if sum(i1[k] != i2[k] for k in range(d)) == 1:
            p1 = [aabb[i1[k]][k] for k in range(d)]
            p2 = [aabb[i2[k]][k] for k in range(d)]
            lines.append(add_line(p1, p2, **kwargs))
    return lines

def draw_point(point, size=0.01, **kwargs):
    lines = []
    for i in range(len(point)):
        axis = np.zeros(len(point))
        axis[i] = 1.0
        p1 = np.array(point) - size/2 * axis
        p2 = np.array(point) + size/2 * axis
        lines.append(add_line(p1, p2, **kwargs))
    return lines
    #extent = size * np.ones(len(point)) / 2
    #aabb = np.array(point) - extent, np.array(point) + extent
    #return draw_aabb(aabb, **kwargs)

def get_face_edges(face):
    #return list(combinations(face, 2))
    return list(zip(face, face[1:] + face[:1]))

def draw_mesh(mesh, **kwargs):
    verts, faces = mesh
    lines = []
    for face in faces:
        for i1, i2 in get_face_edges(face):
            lines.append(add_line(verts[i1], verts[i2], **kwargs))
    return lines

def draw_ray(ray, ray_result=None, visible_color=GREEN, occluded_color=RED, **kwargs):
    if ray_result is None:
        return [add_line(ray.start, ray.end, color=visible_color, **kwargs)]
    if ray_result.objectUniqueId == -1:
        hit_position = ray.end
    else:
        hit_position = ray_result.hit_position
    return [
        add_line(ray.start, hit_position, color=visible_color, **kwargs),
        add_line(hit_position, ray.end, color=occluded_color, **kwargs),
    ]
