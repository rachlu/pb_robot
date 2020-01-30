from collections import namedtuple
from itertools import product, combinations
import numpy as np
import pybullet as p
import pb_robot.helper as helper

MAX_DISTANCE = 0
CLIENT = 0
BASE_LINK = -1
UNBOUNDED_LIMITS = -np.inf, np.inf
CIRCULAR_LIMITS = -np.pi, np.pi

def get_collision_fn(body, joints, obstacles, attachments, self_collisions, custom_limits={}, **kwargs):
    check_link_pairs = get_self_link_pairs(body, joints) if self_collisions else []
    moving_links = frozenset(get_moving_links(body, joints))
    moving_bodies = [(body, moving_links)] + attachments
    check_body_pairs = list(product(moving_bodies, obstacles))  
    lower_limits, upper_limits = get_custom_limits(body, joints, custom_limits)

    def collision_fn(q):
        if not helper.all_between(lower_limits, q, upper_limits):
            return True
        set_joint_positions(body, joints, q) #TODO need to use new one so sets attachments with it
        #TODO reset scene
        #for attachment in attachments:
        #    attachment.assign()
        for link1, link2 in check_link_pairs:
            if pairwise_link_collision(body, link1, body, link2): 
                return True
        for body1, body2 in check_body_pairs:
            if pairwise_collision(body1, body2, **kwargs): 
                return True
        return False
    return collision_fn


def get_self_link_pairs(body, joints, disabled_collisions=set(), only_moving=True):
    moving_links = get_moving_links(body, joints)
    fixed_links = list(set(get_links(body)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links))
    if only_moving:
        check_link_pairs.extend(get_moving_pairs(body, joints))
    else:
        check_link_pairs.extend(combinations(moving_links, 2))
    check_link_pairs = list(filter(lambda pair: not are_links_adjacent(body, *pair), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                   (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs

def get_moving_links(body, joints):
    moving_links = set()
    for joint in joints:
        link = child_link_from_joint(joint)
        if link not in moving_links:
            moving_links.update(get_link_subtree(body, link))
    return list(moving_links)

def get_moving_pairs(body, moving_joints):
    """
    Check all fixed and moving pairs
    Do not check all fixed and fixed pairs
    Check all moving pairs with a common
    """
    moving_links = get_moving_links(body, moving_joints)
    for link1, link2 in combinations(moving_links, 2):
        ancestors1 = set(get_joint_ancestors(body, link1)) & set(moving_joints)
        ancestors2 = set(get_joint_ancestors(body, link2)) & set(moving_joints)
        if ancestors1 != ancestors2:
            yield link1, link2

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

def pairwise_link_collision(body1, link1, body2, link2=BASE_LINK, max_distance=MAX_DISTANCE): # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  linkIndexA=link1, linkIndexB=link2,
                                  physicsClientId=CLIENT)) != 0 # getContactPoints


def body_collision(body1, body2, max_distance=MAX_DISTANCE): # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  physicsClientId=CLIENT)) != 0 # getContactPoints`

def pairwise_collision(body1, body2, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        return any_link_pair_collision(body1, links1, body2, links2, **kwargs)
    return body_collision(body1, body2, **kwargs)

def single_collision(body1, **kwargs):
    for body2 in get_bodies():
        if (body1 != body2) and pairwise_collision(body1, body2, **kwargs):
            return True
    return False

def any_link_pair_collision(body1, links1, body2, links2=None, **kwargs):
    # TODO: this likely isn't needed anymore
    if links1 is None:
        links1 = get_all_links(body1)
    if links2 is None:
        links2 = get_all_links(body2)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
            return True
    return False

def link_pairs_collision(body1, links1, body2, links2=None, **kwargs):
    if links2 is None:
        links2 = get_all_links(body2)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
            return True
    return False



######################################################
####### Remove (or add to functionality) this later #############################
#######################################################

JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])

def get_num_joints(body):
    return p.getNumJoints(body, physicsClientId=CLIENT)

def get_joints(body):
    return list(range(get_num_joints(body)))

get_links = get_joints

def child_link_from_joint(joint):
    return joint # link

def get_all_links(body):
    return [BASE_LINK] + list(get_links(body))

def expand_links(body):
    body, links = body if isinstance(body, tuple) else (body, None)
    if links is None:
        links = get_all_links(body)
    return body, links

def flatten_links(body, links=None):
    if links is None:
        links = get_all_links(body)
    return {(body, frozenset([link])) for link in links}

def set_joint_position(body, joint, value):
    p.resetJointState(body, joint, value, targetVelocity=0, physicsClientId=CLIENT)

def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        set_joint_position(body, joint, value)

def are_links_adjacent(body, link1, link2):
    return (get_link_parent(body, link1) == link2) or \
           (get_link_parent(body, link2) == link1)

def get_link_parent(body, link):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link).parentIndex

def get_joint_info(body, joint):
    return JointInfo(*p.getJointInfo(body, joint, physicsClientId=CLIENT))

def get_link_descendants(body, link, test=lambda l: True):
    descendants = []
    for child in get_link_children(body, link):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(body, child, test=test))
    return descendants

def get_link_subtree(body, link, **kwargs):
    return [link] + get_link_descendants(body, link, **kwargs)

def get_link_ancestors(body, link):
    parent = get_link_parent(body, link)
    if parent is None:
        return []
    return get_link_ancestors(body, parent) + [parent]

def get_joint_ancestors(body, link):
    return get_link_ancestors(body, link) + [link]

def is_circular(body, joint):
    joint_info = get_joint_info(body, joint)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    return joint_info.jointUpperLimit < joint_info.jointLowerLimit

def get_all_link_parents(body):
    return {link: get_link_parent(body, link) for link in get_links(body)}

def get_all_link_children(body):
    children = {}
    for child, parent in get_all_link_parents(body).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children

def get_link_children(body, link):
    children = get_all_link_children(body)
    return children.get(link, [])

def get_joint_limits(body, joint):
    # TODO: make a version for several joints?
    if is_circular(body, joint):
        # TODO: return UNBOUNDED_LIMITS
        return CIRCULAR_LIMITS
    joint_info = get_joint_info(body, joint)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit

