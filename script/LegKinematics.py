import numpy as np

# Geometric parameter
R = 0.001 * 100
theta_0 = np.deg2rad(17)
n_HF = 130 / 180
n_BC = 101 / 180

# side OA
l1 = 0.8 * R
# side BC
l3 = 2 * R * np.sin(np.pi * n_BC / 2)
# side DC
l4 = 0.88296634 * R
# side AD
l5 = 0.9 * R
# side DE
l6 = 0.4 * R
# side CF
l7 = 2 * R * np.sin((130 - 17 - 101) * np.pi / 180 / 2)
# side GF
l8 = 2 * R * np.sin(np.radians(25))
l_BF = 0


def FowardKinematics(state, state_mode="tb"):
    # state = np.array([[phiR], [phiL]]) or np.array([[theta], [beta]])
    if state_mode == "phi":
        state_tb = 1 / 2 * np.array([[1, -1], [1, 1]]) @ state + np.array([[1], [0]]) * np.deg2rad(17)
    else:
        state_tb = state

    theta = state_tb[0, 0]
    beta = state_tb[1, 0]

    OAR_ref = np.array([[l1 * np.sin(theta)], [l1 * np.cos(theta)]])

    OBR_ref = np.array([[R * np.sin(theta)], [R * np.cos(theta)]])

    alpha_1 = np.pi - theta
    alpha_2 = np.arcsin(l1 / (l5 + l6)) * np.sin(alpha_1)

    OER_ref = np.array([[0], [l1 * np.cos(theta) - (l5 + l6) * np.cos(alpha_2)]])

    ODR_ref = l5 / (l5 + l6) * OER_ref + l6 / (l5 + l6) * OAR_ref

    # Derive vector OC
    BD = ODR_ref - OBR_ref
    DB = -1 * BD
    # l_BD
    # print(BD)
    l_BD = np.linalg.norm(BD)
    # alpha_3 defined as angle BDC
    cos_alpha_3 = (l_BD**2 + l4**2 - l3**2) / (2 * l_BD * l4)
    sin_alpha_3 = np.sqrt(1 - cos_alpha_3**2)

    rot_alpha_3 = np.array([[cos_alpha_3, sin_alpha_3], [-sin_alpha_3, cos_alpha_3]])

    DC = (l4 / l_BD) * rot_alpha_3 @ DB

    OCR_ref = ODR_ref + DC

    # Derive vector OF
    CB = OBR_ref - OCR_ref
    # alpha_4 defined as angle BCF
    alpha_4 = np.arccos(l3 / (2 * R)) + np.arccos(l7 / (2 * R))
    # Rotate CB to CF direction
    rot_alpha_4 = np.array([[np.cos(alpha_4), -np.sin(alpha_4)], [np.sin(alpha_4), np.cos(alpha_4)]])

    l_BF = np.sqrt(l3**2 + l7**2 - 2 * l3 * l7 * np.cos(alpha_4))
    l_CB = np.linalg.norm(CB)
    CF = (l7 / l_CB) * rot_alpha_4 @ CB
    OFR_ref = OCR_ref + CF

    # Derive vector OG
    # alpha_5 defined as angle OGF
    # alpha_6 defined as angle GOF
    l_OF = np.linalg.norm(OFR_ref)
    sin_alpha_6 = OFR_ref[0, 0] / l_OF
    sin_alpha_5 = (l_OF / l8) * sin_alpha_6
    cos_alpha_5 = np.sqrt(1 - sin_alpha_5**2)
    OG_ref = np.array([[0], [OFR_ref[1, 0] - l8 * cos_alpha_5]])

    # Derive Left Plane Vectors
    mirror_mat = np.array([[-1, 0], [0, 1]])

    """ OAL_ref = mirror_mat @ OAR_ref
    OBL_ref = mirror_mat @ OBR_ref
    OCL_ref = mirror_mat @ OCR_ref
    ODL_ref = mirror_mat @ ODR_ref
    OEL_ref = mirror_mat @ OER_ref
    OFL_ref = mirror_mat @ OFR_ref """

    # Transform to theta beta frame
    rot_beta = np.array([[np.cos(beta), np.sin(beta)], [-np.sin(beta), np.cos(beta)]])

    """ OAR = rot_beta @ OAR_ref
    OBR = rot_beta @ OBR_ref
    OCR = rot_beta @ OCR_ref
    ODR = rot_beta @ ODR_ref
    OER = rot_beta @ OER_ref
    OFR = rot_beta @ OFR_ref
    OAL = rot_beta @ OAL_ref
    OBL = rot_beta @ OBL_ref
    OCL = rot_beta @ OCL_ref
    ODL = rot_beta @ ODL_ref
    OEL = rot_beta @ OEL_ref
    OFL = rot_beta @ OFL_ref """
    OG = rot_beta @ OG_ref

    return OG


def InverseKinematics(point_G, current_tb=np.array([[np.radians(17)], [0]]), tol=0.0001, p_gain=10):
    desired_length = np.linalg.norm(point_G)
    # Solve theta correspond to desired length
    if desired_length > 0.3428:
        return "error"
    else:
        v_OG = FowardKinematics(current_tb)
        iter_length = np.linalg.norm(v_OG)
        theta = current_tb[0, 0]

        error = desired_length - iter_length

        tolerance = 0.0001
        p_gain = 10

        while abs(error) > tolerance:
            theta = theta + (p_gain * error)
            v_OG = FowardKinematics(np.array([[theta], [0]]))
            iter_length = np.sqrt(v_OG.T @ v_OG)[0, 0]
            error = desired_length - iter_length

        u_sp = point_G / desired_length
        beta = np.arccos(-u_sp.T @ np.array([[0], [1]]))[0, 0]
        if point_G[0, 0] <= 0:
            beta_sign = 1
        else:
            beta_sign = -1
        beta = abs(beta) * beta_sign
        return np.array([[theta], [beta]])


def InverseKinematicsPoly(point_G):
    desired_length = np.linalg.norm(point_G)
    # Solve theta correspond to desired length
    if desired_length > 0.3428 or desired_length < 0.098:
        raise ValueError("Desired length exceed ", str(desired_length), "Point desired", str(point_G.T))
    else:
        p_fit_ = [
            4.11916505e05,
            -6.26591689e05,
            4.03501845e05,
            -1.42028997e05,
            2.93956532e04,
            -3.56716856e03,
            2.44492931e02,
            -7.12887643e00,
        ]

        theta = np.polyval(p_fit_, desired_length)

        u_sp = point_G / desired_length
        beta = np.arccos(-u_sp.T @ np.array([[0], [1]]))[0, 0]
        if point_G[0, 0] <= 0:
            beta_sign = 1
        else:
            beta_sign = -1
        beta = abs(beta) * beta_sign
        return np.array([[theta], [beta]])
