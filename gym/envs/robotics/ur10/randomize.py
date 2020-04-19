import xml.etree.ElementTree as ET
import numpy as np
import os
import mujoco_py
from gym.envs.robotics import rotations
HOME_PATH = os.getenv("HOME")

damping_arm = 200
damping_wrist = 75

friction_body = [1, 0.05, 0.0001]  # [tangential, torsional, rolling]
friction_heg = [1, 0.05, 0.0001]

#body_pos = np.array([1.57, -0.945, 1.205])
body_pos = np.array([2.19, 2.275, 0.95])
body_quat = np.array([0.6427876, 0, 0, -0.7660444])


ur_path = os.path.join(*[HOME_PATH, "DRL_SetBot-RearVentilation", "UR10"])
main_xml = os.path.join(*[ur_path, "ur10_assembly_setup_rand.xml"])

robot_body_xml = os.path.join(*[ur_path, "ur10_heg", "ur10_body_heg_rand.xml"])

car_body_xml = os.path.join(*[ur_path, "ur10_heg", "car_body_rand.xml"])

defaults_xml = os.path.join(*[ur_path, "ur10_heg", "ur10_default_rand.xml"])


def normalize_rad(angles):
    angles = np.array(angles)
    angles = angles % (2*np.pi)
    angles = (angles + 2*np.pi) % (2*np.pi)
    for i in range(len(angles)):
        if (angles[i] > np.pi):
            angles[i] -= 2*np.pi
    return angles


def randomize_ur10_xml(var_mass=0.2, var_damp=0.1, var_fr=0.1, var_grav_x_y=0.1, var_grav_z=0.2, var_body_pos=0.02,
                       var_body_rot=0.5, worker_id=1):

    # parameters:
    #
    # var_mass        : mass variance relative to actual mass
    #
    # var_damp        : damping variance relative to the standard value
    # var_fr          : friction variance relative to the standard value
    #
    # var_grav_x_y    : gravity variance in x- and y-direction (absolute)
    # var_grav_z      : gravity variance in z-direction (absolute)
    #
    # var_body_pos    : variance of body position in m per direction
    # var_body_rot    : variance of body rotation in degree per axis
    # ______________________________________________________________________
    #
    # return          :  offset of car_body to consider for goal position
    #
    # ______________________________________________________________________


    main_xml_temp = os.path.join(*[ur_path, "ur10_assembly_setup_rand_temp_{}.xml".format(worker_id)])
    robot_body_rel = os.path.join("ur10_heg", "ur10_body_heg_rand_temp_{}.xml".format(worker_id))
    car_body_rel = os.path.join("ur10_heg", "car_body_rand_temp_{}.xml".format(worker_id))
    defaults_rel = os.path.join("ur10_heg", "ur10_default_rand_temp_{}.xml".format(worker_id))
    robot_body_xml_temp = os.path.join(ur_path, robot_body_rel)
    car_body_xml_temp = os.path.join(ur_path, car_body_rel)
    defaults_xml_temp = os.path.join(ur_path, defaults_rel)

    # load robot body xml and randomize body masses
    tree = ET.parse(robot_body_xml)
    inertials = []
    for elem in tree.iter():
        if elem.tag == 'inertial':      # find inertial elements in xml
            inertials.append(elem)

    for inertial in inertials:          # change masses
        inertial.attrib['mass'] = str((np.random.randn() * var_mass + 1) * float(inertial.attrib['mass']))

    tree.write(robot_body_xml_temp)     # save new xml in temp file to be loaded in gym env

    # load defaults xml and randomize joint damping and surface friction
    damping_arm_rand = damping_arm * (1 + var_damp * np.random.randn())
    damping_wrist_rand = damping_wrist * (1 + var_damp * np.random.randn())

    friction_body_rand = friction_body * (1 + var_fr * np.random.randn(3, ))
    friction_heg_rand = friction_heg * (1 + var_fr * np.random.randn(3, ))

    friction_body_string = "{} {} {}".format(*friction_body_rand)
    friction_heg_string = "{} {} {}".format(*friction_heg_rand)

    tree = ET.parse(defaults_xml)

    for default in tree.findall('default'):
        if default.attrib['class'] == 'ur10:arm':
            default.findall('joint')[0].attrib['damping'] = str(damping_arm_rand)
        if default.attrib['class'] == 'ur10:wrist':
            default.findall('joint')[0].attrib['damping'] = str(damping_wrist_rand)
        if default.attrib['class'] == 'body':
            default.findall('geom')[0].attrib['friction'] = friction_body_string
        if default.attrib['class'] == 'heg':
            default.findall('geom')[0].attrib['friction'] = friction_heg_string

    tree.write(defaults_xml_temp)

    # load main xml and randomize gravity, also adapt includes to worker_id
    grav_x = var_grav_x_y * np.random.randn()
    grav_y = var_grav_x_y * np.random.randn()
    grav_z = var_grav_z * np.random.randn() - 9.81
    grav_string = "{} {} {}".format(grav_x, grav_y, grav_z)

    tree = ET.parse(main_xml)
    tree.findall("option")[0].attrib['gravity'] = grav_string

    for include in tree.findall("default")[0].iter("include"):
        include.attrib['file'] = defaults_rel
    world_temp_files = [car_body_rel, robot_body_rel]
    for i, include in enumerate(tree.findall("worldbody")[0].iter("include")):
        include.attrib['file'] = world_temp_files[i]

    tree.write(main_xml_temp)

    # load car body xml and change position (offset ist returned by this function)
    body_euler = normalize_rad(rotations.quat2euler(body_quat))
    body_pos_offset = np.random.uniform(-var_body_pos, var_body_pos, (3,))
    body_euler_offset = np.random.uniform(-var_body_rot/180*np.pi, var_body_rot/180*np.pi, (3,))

    body_pos_rand = body_pos + body_pos_offset
    body_euler_rand = body_euler + body_euler_offset
    body_quat_rand = rotations.euler2quat(body_euler_rand)

    body_pos_rand_string = "{} {} {}".format(*body_pos_rand)
    body_quat_rand_string = "{} {} {} {}".format(*body_quat_rand)

    tree = ET.parse(car_body_xml)
    for body in tree.findall('body'):
        if body.attrib['name'] == 'body_link':
            body.attrib['pos'] = body_pos_rand_string
            body.attrib['quat'] = body_quat_rand_string
    tree.write(car_body_xml_temp)

    offset = {
        'body_pos_offset': body_pos_offset,
        'body_euler_offset': body_euler_offset
    }

    return offset
