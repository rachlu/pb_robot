# Control

def control_joint(body, joint, value):
    return p.setJointMotorControl2(bodyUniqueId=body,
                                   jointIndex=joint,
                                   controlMode=p.POSITION_CONTROL,
                                   targetPosition=value,
                                   targetVelocity=0.0,
                                   maxVelocity=get_max_velocity(body, joint),
                                   force=get_max_force(body, joint),
                                   physicsClientId=CLIENT)

def control_joints(body, joints, positions):
    # TODO: the whole PR2 seems to jitter
    #kp = 1.0
    #kv = 0.3
    #forces = [get_max_force(body, joint) for joint in joints]
    #forces = [5000]*len(joints)
    #forces = [20000]*len(joints)
    return p.setJointMotorControlArray(body, joints, p.POSITION_CONTROL,
                                       targetPositions=positions,
                                       targetVelocities=[0.0] * len(joints),
                                       physicsClientId=CLIENT) #,
                                        #positionGains=[kp] * len(joints),
                                        #velocityGains=[kv] * len(joints),)
                                        #forces=forces)

def joint_controller(body, joints, target, tolerance=1e-3):
    assert(len(joints) == len(target))
    positions = get_joint_positions(body, joints)
    while not np.allclose(positions, target, atol=tolerance, rtol=0):
        control_joints(body, joints, target)
        yield positions
        positions = get_joint_positions(body, joints)

def joint_controller_hold(body, joints, target, **kwargs):
    """
    Keeps other joints in place
    """
    movable_joints = get_movable_joints(body)
    conf = list(get_joint_positions(body, movable_joints))
    for joint, value in zip(movable_from_joints(body, joints), target):
        conf[joint] = value
    return joint_controller(body, movable_joints, conf, **kwargs)

def joint_controller_hold2(body, joints, positions, velocities=None,
                           tolerance=1e-2 * np.pi, position_gain=0.05, velocity_gain=0.01):
    """
    Keeps other joints in place
    """
    # TODO: velocity_gain causes the PR2 to oscillate
    if velocities is None:
        velocities = [0.] * len(positions)
    movable_joints = get_movable_joints(body)
    target_positions = list(get_joint_positions(body, movable_joints))
    target_velocities = [0.] * len(movable_joints)
    movable_from_original = {o: m for m, o in enumerate(movable_joints)}
    #print(list(positions), list(velocities))
    for joint, position, velocity in zip(joints, positions, velocities):
        target_positions[movable_from_original[joint]] = position
        target_velocities[movable_from_original[joint]] = velocity
    # return joint_controller(body, movable_joints, conf)
    current_conf = get_joint_positions(body, movable_joints)
    #forces = [get_max_force(body, joint) for joint in movable_joints]
    while not np.allclose(current_conf, target_positions, atol=tolerance, rtol=0):
        # TODO: only enforce velocity constraints at end
        p.setJointMotorControlArray(body, movable_joints, p.POSITION_CONTROL,
                                    targetPositions=target_positions,
                                    #targetVelocities=target_velocities,
                                    positionGains=[position_gain] * len(movable_joints),
                                    #velocityGains=[velocity_gain] * len(movable_joints),
                                    #forces=forces,
                                    physicsClientId=CLIENT)
        yield current_conf
        current_conf = get_joint_positions(body, movable_joints)

def trajectory_controller(body, joints, path, **kwargs):
    for target in path:
        for positions in joint_controller(body, joints, target, **kwargs):
            yield positions

def simulate_controller(controller, max_time=np.inf): # Allow option to sleep rather than yield?
    sim_dt = get_time_step()
    sim_time = 0.0
    for _ in controller:
        if max_time < sim_time:
            break
        step_simulation()
        sim_time += sim_dt
        yield sim_time

def velocity_control_joints(body, joints, velocities):
    #kv = 0.3
    return p.setJointMotorControlArray(body, joints, p.VELOCITY_CONTROL,
                                       targetVelocities=velocities,
                                       physicsClientId=CLIENT) #,
                                        #velocityGains=[kv] * len(joints),)
                                        #forces=forces)

