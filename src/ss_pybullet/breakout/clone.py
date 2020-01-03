def clone_visual_shape(body, link, client=None):
    client = get_client(client)
    #if not has_gui(client):
    #    return -1
    visual_data = get_visual_data(body, link)
    if not visual_data:
        return -1
    assert (len(visual_data) == 1)
    return visual_shape_from_data(visual_data[0], client)

def clone_collision_shape(body, link, client=None):
    client = get_client(client)
    collision_data = get_collision_data(body, link)
    if not collision_data:
        return -1
    assert (len(collision_data) == 1)
    # TODO: can do CollisionArray
    return collision_shape_from_data(collision_data[0], body, link, client)

def clone_body(body, links=None, collision=True, visual=True, client=None):
    # TODO: names are not retained
    # TODO: error with createMultiBody link poses on PR2
    # localVisualFrame_position: position of local visual frame, relative to link/joint frame
    # localVisualFrame orientation: orientation of local visual frame relative to link/joint frame
    # parentFramePos: joint position in parent frame
    # parentFrameOrn: joint orientation in parent frame
    client = get_client(client) # client is the new client for the body
    if links is None:
        links = get_links(body)
    #movable_joints = [joint for joint in links if is_movable(body, joint)]
    new_from_original = {}
    base_link = get_link_parent(body, links[0]) if links else BASE_LINK
    new_from_original[base_link] = -1

    masses = []
    collision_shapes = []
    visual_shapes = []
    positions = [] # list of local link positions, with respect to parent
    orientations = [] # list of local link orientations, w.r.t. parent
    inertial_positions = [] # list of local inertial frame pos. in link frame
    inertial_orientations = [] # list of local inertial frame orn. in link frame
    parent_indices = []
    joint_types = []
    joint_axes = []
    for i, link in enumerate(links):
        new_from_original[link] = i
        joint_info = get_joint_info(body, link)
        dynamics_info = get_dynamics_info(body, link)
        masses.append(dynamics_info.mass)
        collision_shapes.append(clone_collision_shape(body, link, client) if collision else -1)
        visual_shapes.append(clone_visual_shape(body, link, client) if visual else -1)
        point, quat = get_local_link_pose(body, link)
        positions.append(point)
        orientations.append(quat)
        inertial_positions.append(dynamics_info.local_inertial_pos)
        inertial_orientations.append(dynamics_info.local_inertial_orn)
        parent_indices.append(new_from_original[joint_info.parentIndex] + 1) # TODO: need the increment to work
        joint_types.append(joint_info.jointType)
        joint_axes.append(joint_info.jointAxis)
    # https://github.com/bulletphysics/bullet3/blob/9c9ac6cba8118544808889664326fd6f06d9eeba/examples/pybullet/gym/pybullet_utils/urdfEditor.py#L169

    base_dynamics_info = get_dynamics_info(body, base_link)
    base_point, base_quat = get_link_pose(body, base_link)
    new_body = p.createMultiBody(baseMass=base_dynamics_info.mass,
                                 baseCollisionShapeIndex=clone_collision_shape(body, base_link, client) if collision else -1,
                                 baseVisualShapeIndex=clone_visual_shape(body, base_link, client) if visual else -1,
                                 basePosition=base_point,
                                 baseOrientation=base_quat,
                                 baseInertialFramePosition=base_dynamics_info.local_inertial_pos,
                                 baseInertialFrameOrientation=base_dynamics_info.local_inertial_orn,
                                 linkMasses=masses,
                                 linkCollisionShapeIndices=collision_shapes,
                                 linkVisualShapeIndices=visual_shapes,
                                 linkPositions=positions,
                                 linkOrientations=orientations,
                                 linkInertialFramePositions=inertial_positions,
                                 linkInertialFrameOrientations=inertial_orientations,
                                 linkParentIndices=parent_indices,
                                 linkJointTypes=joint_types,
                                 linkJointAxis=joint_axes,
                                 physicsClientId=client)
    #set_configuration(new_body, get_joint_positions(body, movable_joints)) # Need to use correct client
    for joint, value in zip(range(len(links)), get_joint_positions(body, links)):
        # TODO: check if movable?
        p.resetJointState(new_body, joint, value, targetVelocity=0, physicsClientId=client)
    return new_body

def clone_world(client=None, exclude=[]):
    visual = has_gui(client)
    mapping = {}
    for body in get_bodies():
        if body not in exclude:
            new_body = clone_body(body, collision=True, visual=visual, client=client)
            mapping[body] = new_body
    return mapping
