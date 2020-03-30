import pybullet as p
import time
import IPython 
import numpy
import os
import copy
import pb_robot
import pb_robot.utils_noBase as utils
import matplotlib.pyplot as plt
import pb_robot.transformations as trans

def posControl(robot, path, threshold=0.1):
    n = len(path[0])
    i = 0
    while True:
        p.setJointMotorControlArray(robot.id, robot.arm.jointsID,
                        controlMode=p.POSITION_CONTROL, 
                        targetPositions=path[i],
                        targetVelocities=[0]*n, 
                        forces=robot.arm.torque_limits,
                        positionGains=[0.1]*n, 
                        velocityGains=[1]*n)
        p.stepSimulation()
        time.sleep(0.01)

        if numpy.linalg.norm(numpy.subtract(robot.arm.GetJointValues(), path[i])) < threshold:
            if i == (len(path)-1):
                break
            else:
                i += 1

def clampTorque(robot, tau_d):
    tau_limit = robot.arm.torque_limits
    for i in xrange(len(tau_d)):
        tau_d[i] = pb_robot.helper.clip(tau_d[i], -tau_limit[i], tau_limit[i])
    return tau_d

def forceControl(robot, wrench_target):
    wrench_desired = numpy.zeros((6, 1))
    gain = 0.1 #Was 0.01
    p.setGravity(0, 0, -9.81)

    for _ in xrange(1000):
        p.setJointMotorControlArray(robot.id, robot.arm.jointsID, p.VELOCITY_CONTROL,
                                targetVelocities=[0]*7, forces=[40]*7)

        jacobian = robot.arm.GetJacobian(robot.arm.GetJointValues())
        # Feedforward. Commented out PI control because gains were 0 
        tau_d = numpy.matmul(numpy.transpose(jacobian), wrench_desired)
        tau_cmd = clampTorque(robot, tau_d)

        p.setJointMotorControlArray(robot.id, robot.arm.jointsID,
                        controlMode=p.TORQUE_CONTROL,
                        forces=tau_cmd)
                      
        wrench_desired = gain * wrench_target + (1 - gain) * wrench_desired
        p.stepSimulation()
        time.sleep(0.01)

def moveToTouch(robot, q_desired):
    n = len(q_desired)
    p.stepSimulation()
    ft_past = p.getJointState(robot.id, 8)[2] 
    i = 0
    while True:
        p.setJointMotorControlArray(robot.id, robot.arm.jointsID,
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=q_desired,
                        targetVelocities=[0]*n,
                        forces=robot.arm.torque_limits,
                        positionGains=[0.1]*n,
                        velocityGains=[1]*n)
        p.stepSimulation()
        time.sleep(0.01)

        ft = robot.arm.GetFTWristReading() #p.getJointState(robot.id, robot.ft_joint.jointID)[2]
        # So the in collision check seems incredibly accurate
        # The FT sensor check produces false positives (and stops trigging once in collision
        # So for now its check FT and then confirm with is collision. 
        if (numpy.linalg.norm(numpy.subtract(ft, ft_past)) > 100):
            if not robot.arm.IsCollisionFree(robot.arm.GetJointValues()):
                break

        if numpy.linalg.norm(numpy.subtract(robot.arm.GetJointValues(), q_desired)) < 0.01:
            if robot.arm.IsCollisionFree(robot.arm.GetJointValues()):
                raise RuntimeError("MoveToTouch did not end in contact") 

        ft_past = copy.deepcopy(ft)

def cartImpedance(robot, pose_d_target, stiffness_params):
    p.setGravity(0, 0, -9.81)

    position_d_target = pose_d_target[0:3, 3]
    stiffness_target = numpy.diag(stiffness_params)
    damping_target = numpy.diag(2.0*numpy.sqrt(stiffness_params))

    stiffness = numpy.eye(6)
    damping = numpy.eye(6)
    position_d = robot.arm.GetEETransform()[0:3, 3]
    ori_d = pb_robot.geometry.quat_from_matrix(robot.arm.GetEETransform()[0:3, 0:3])
    gain = 0.1

    while True:
        p.setJointMotorControlArray(robot.id, robot.arm.jointsID, p.VELOCITY_CONTROL,
                                    targetVelocities=[0]*7, forces=[40]*7)

        # Compute Pose Error
        error = numpy.zeros((6))
        current_pose = robot.arm.GetEETransform()
        position_error = current_pose[0:3, 3] - position_d
        error[0:3] = position_error

        current_ori = pb_robot.geometry.quat_from_matrix(current_pose[0:3, 0:3])
        # Compute different quaternion
        error_ori_quat = trans.quaternion_multiply(current_ori, trans.quaternion_inverse(ori_d))
        # Convert to axis angle
        (error_ori_angle, error_ori_axis) = pb_robot.geometry.quatToAxisAngle(error_ori_quat)
        ori_error = numpy.multiply(error_ori_axis, error_ori_angle)
        #error[3:6] = ori_error
        #TODO Adding orientation error creates wild movements. Must be debugged

        q = robot.arm.GetJointValues()
        dq = robot.arm.GetJointVelocities()
        jacobian = robot.arm.GetJacobian(q)
        tau_task = (numpy.transpose(jacobian)).dot(-stiffness.dot(error) - damping.dot(jacobian.dot(dq)))
        coriolis = robot.arm.GetCoriolosMatrix(q, dq)
        tau_d = numpy.add(tau_task, coriolis)
        tau_cmd = clampTorque(robot, tau_d)

        p.setJointMotorControlArray(robot.id, robot.arm.jointsID,
                        controlMode=p.TORQUE_CONTROL,
                        forces=tau_cmd)

        stiffness = gain * stiffness_target + (1 - gain) * stiffness
        damping = gain * damping_target + (1 - gain) * damping
        position_d = gain * position_d_target + (1 - gain) * position_d
        #TODO add filter gain orientation
        p.stepSimulation()
        time.sleep(0.01)



if __name__ == '__main__':
    utils.connect(use_gui=True)
    utils.disable_real_time()

    objects_path = pb_robot.helper.getDirectory()
    floor_file = os.path.join(objects_path, 'furniture/short_floor.urdf')
    floor = pb_robot.body.createBody(floor_file) 
    robot = pb_robot.panda.Panda()

    utils.set_default_camera()
    #utils.dump_world()
    
    qup = numpy.array([ 2.55563527,  1.00573286, -0.84798717, -1.57899963,  0.87321658,
        2.16885793,  0.63311249])
    qclose = numpy.array([ 2.49362013,  1.08166001, -0.75861633, -1.53515376,  0.90153413,
        2.25064744,  0.62651343])
    qlevel = numpy.array([ 2.47985685,  1.10600568, -0.73863667, -1.52578508,  0.90008318,
        2.26543946,  0.62807649])
    qdown = numpy.array([ 2.42067086,  1.18915922, -0.66266409, -1.4618413 ,  0.90355052,
        2.3278753 ,  0.63539486])

    wrench = numpy.zeros((6, 1))
    wrench[0] = 100
    wrench[1] = 50

    qs = [[0, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0]]


    pose = robot.arm.GetEETransform()
    pose[2, 3] += 0.2

    #p.setRealTimeSimulation(1)
    #p.setGravity(0, 0, -9.81)

    IPython.embed()

'''
def saturatedTorqueRate(tau_d, tau_J_d):
    kdeltaTauMax = 1
    tau_d_saturated = numpy.zeros((7, 1))
    for i in xrange(7):
        difference = tau_d[i] - tau_J_d[i]
        tau_d_saturated[i] = tau_J_d[i] + max(min(difference, kdeltaTauMax), -kdeltaTauMax)
    return tau_d_saturated

def forceControl(robot, wrench_target):
    wrench_desired = numpy.zeros((6, 1))
    tau_error = numpy.zeros((7, 1))
    tau_ext_initial = 0 #7x1
    kp = 0 
    ki = 0
    gain = 0.01    

    #TODO go into loop
    gravity = 0; #TODO 7x1 vector, gravity vector as a function of q
    tau_measured = robot.arm.GetJointTorques()
    tau_J_d = robot.arm.GetJointTorques() # 7x1 array. "Desired link-side joint torque sensor signals without gravity."

    tau_ext = tau_measured - gravity - tau_ext_initial
    tau_d = jacobian.transpose() * wrench_desired
    tau_error = tau_error + period.toSec() * (tau_d - tau_ext)  
    # Feedforward + PI control (but gains are 0 so comment out PI control for now)
    tau_cmd = tau_d + kp*(tau_d - tau_ext) + ki*tau_error
    tau_cmd_bounded = saturatedTorquRate(tau_cmd, tau_J_d)

    #TODO send tau_cmd_bounded

    wrench_desired = gain * wrench_target + (1 - gain) * wrench_desired
'''
