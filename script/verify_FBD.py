import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import warnings
import csv

sbrio_filepath = "../exp_data/20230718/1/sbrio.csv"
vicon_filepath = "../exp_data/20230718/1/vicon.csv"
orin_filepath = "../exp_data/20230718/1/orin.csv"


class PointState:
    def __init__(self, pointList):
        if pointList != "Empty":
            self.Error = False
            self.frame_idx, self.frame_sub, self.C1, self.C2, self.C3, self.AO, self.AG, self.BO, self.BG, self.CO, self.CG, self.DO, self.DG = pointList

            self.COM = self.getCOM()
            self.rotation = None
            self.quaternion = np.array([])  # [x, y, z, w]
            self.rotation_matrix = np.array([])
            self.getOrient()
        else:
            self.Error = True

    def getCOM(self):
        # print(self.AO)
        # print(self.BO)
        # print(self.CO)
        # print(self.DO)
        self.COM = (self.AO + self.BO + self.CO + self.DO)*1/4
        # print("COM = ", self.COM, "\n")

    def getOrient(self):
        v_CB = self.BO - self.CO
        v_CD = self.DO - self.CO
        v_x = v_CB / np.linalg.norm(v_CB)
        v_y = v_CD / np.linalg.norm(v_CD)
        v_z = np.cross(v_x, v_y)

        self.rotation_matrix = np.vstack((v_x, v_y, v_z)).T
        self.rotation = R.from_matrix(self.rotation_matrix)
        self.quaternion = self.rotation.as_quat()
        # self.rotation_matrix = np.column_stack((v_x.T, v_y.T, v_z.T))
        '''
        print("V_x = ", v_x)
        print("v_y = ", v_y)
        print("v_z = ", v_z)
        print("rot = ", self.rotation_matrix)
        print("quat = ", self.quaternion)
        '''


class ForceState:
    def __init__(self, forceList):
        self.frame_idx, self.frame_sub, self.F1, self.M1, self.CP1, self.F2, self.M2, self.CP2, self.F3, self.M3, self.CP3, self.F4, self.M4, self.CP4 = forceList


class MotorState:
    def __init__(self, cmd_pos, cmd_torq, cmd_Kp, cmd_Kd, rpy_pos, rpy_torq, pb_V, pb_I):
        self.cmd_pos = cmd_pos
        self.cmd_torq = cmd_torq
        self.cmd_Kp = cmd_Kp
        self.cmd_Kd = cmd_Kd
        self.rpy_pos = rpy_pos
        self.rpy_torq = rpy_torq
        self.pb_V = pb_V
        self.pb_I = pb_I


class ModuleState:
    def __init__(self, motor_R, motor_L) -> None:
        self.Motor_R = motor_R
        self.Motor_L = motor_L


class ImuState:
    def __init__(self, tstamp, acc, omega, att) -> None:
        # acc = [acc_x, acc_y, acc_z]
        # w = [w_x, w_y, w_z]
        # attitude = [x, y, z, w]
        self.tstamp = tstamp
        self.acc = acc
        self.ang_vel = omega
        self.att = att  # [x, y, z, w]
        pass


class ExperimentData:
    def __init__(self, sbrio_path, vicon_path, orin_path) -> None:
        self.sbrio_path = sbrio_path
        self.vicon_path = vicon_path
        self.orin_path = orin_path

        # Vicon Nexus Data
        self.FrameTrim = 13575  # Trim Data after Frame
        # self.FrameTrim = 18000  # Trim Data after Frame

        # Force Plate Data Format
        # |    2    |    1    |
        # | ------- | ------- |
        # |    3    |    4    |
        self.vidx_F1 = 20
        self.vidx_F2 = 11
        self.vidx_F3 = 29
        self.vidx_F4 = 2
        # Vicon Point Data Format
        self.vidx_C1 = 2
        self.vidx_C2 = 5
        self.vidx_C3 = 8
        self.vidx_AO = 11
        self.vidx_AG = 14
        self.vidx_BO = 17
        self.vidx_BG = 20
        self.vidx_CO = 23
        self.vidx_CG = 26
        self.vidx_DO = 29
        self.vidx_DG = 32
        self.vidx_trigger = 35  # first column of vicon trigger point in csv file

        self.force_plate_list = []
        self.vicon_point_list = []
        self.vicon_data_list = []
        self.vicon_trigger_frame = -1
        self.vicon_trigger_idx = -1

        # Sbrio Data
        # Module Data List = [t(seq * 0.001), A, B, C, D]
        self.sidx_AR_cmd_pos = 2
        self.sidx_AR_cmd_torq = 3
        self.sidx_AR_cmd_kp = 4
        self.sidx_AR_cmd_kd = 5
        self.sidx_AR_rpy_pos = 6
        self.sidx_AR_rpy_torq = 7
        self.sidx_AL_cmd_pos = 8
        self.sidx_AL_cmd_torq = 9
        self.sidx_AL_cmd_kp = 10
        self.sidx_AL_cmd_kd = 11
        self.sidx_AL_rpy_pos = 12
        self.sidx_AL_rpy_torq = 13

        self.sidx_BR_cmd_pos = 14
        self.sidx_BR_cmd_torq = 15
        self.sidx_BR_cmd_kp = 16
        self.sidx_BR_cmd_kd = 17
        self.sidx_BR_rpy_pos = 18
        self.sidx_BR_rpy_torq = 19
        self.sidx_BL_cmd_pos = 20
        self.sidx_BL_cmd_torq = 21
        self.sidx_BL_cmd_kp = 22
        self.sidx_BL_cmd_kd = 23
        self.sidx_BL_rpy_pos = 24
        self.sidx_BL_rpy_torq = 25

        self.sidx_CR_cmd_pos = 26
        self.sidx_CR_cmd_torq = 27
        self.sidx_CR_cmd_kp = 28
        self.sidx_CR_cmd_kd = 29
        self.sidx_CR_rpy_pos = 30
        self.sidx_CR_rpy_torq = 31
        self.sidx_CL_cmd_pos = 32
        self.sidx_CL_cmd_torq = 33
        self.sidx_CL_cmd_kp = 34
        self.sidx_CL_cmd_kd = 35
        self.sidx_CL_rpy_pos = 36
        self.sidx_CL_rpy_torq = 37

        self.sidx_DR_cmd_pos = 38
        self.sidx_DR_cmd_torq = 39
        self.sidx_DR_cmd_kp = 40
        self.sidx_DR_cmd_kd = 41
        self.sidx_DR_rpy_pos = 42
        self.sidx_DR_rpy_torq = 43
        self.sidx_DL_cmd_pos = 44
        self.sidx_DL_cmd_torq = 45
        self.sidx_DL_cmd_kp = 46
        self.sidx_DL_cmd_kd = 47
        self.sidx_DL_rpy_pos = 48
        self.sidx_DL_rpy_torq = 49

        self.sidx_pb_AR_V = 58
        self.sidx_pb_AR_I = 59
        self.sidx_pb_AL_V = 60
        self.sidx_pb_AL_I = 61

        self.sidx_pb_BR_V = 62
        self.sidx_pb_BR_I = 63
        self.sidx_pb_BL_V = 64
        self.sidx_pb_BL_I = 65

        self.sidx_pb_CR_V = 66
        self.sidx_pb_CR_I = 67
        self.sidx_pb_CL_V = 68
        self.sidx_pb_CL_I = 69

        self.sidx_pb_DR_V = 70
        self.sidx_pb_DR_I = 71
        self.sidx_pb_DL_V = 72
        self.sidx_pb_DL_I = 73

        self.module_data_list = []
        self.sbrio_trigger_seq = []
        self.sbrio_trigger_idx = -1

        # Orin Data
        # [t, imu]
        self.imu_data_list = []
        self.imu_trigger_tstamp = -1
        self.imu_trigger_idx = -1

        # Trimmed Align Data
        self.trimmed_vicon = []
        self.trimmed_sbrio = []
        self.trimmed_imu = []
        # [t, s_data, o_data, v_data]
        # [t, ModA, ModB, ModC, ModD, Imu, ForcePlate, PointState]
        self.aligned_full_data = []
        pass

    def importViconData(self):
        log_fp_on = False
        log_fp_done = False
        log_vp_on = False
        triggered = False

        with open(self.vicon_path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                if len(row) == 0:
                    log_fp_on = False
                    log_fp_done = True
                elif row[0] == "1" and not log_fp_done:
                    log_fp_on = True

                if len(row) == 0:
                    log_vp_on = False
                elif row[0] == "1" and log_fp_done:
                    log_vp_on = True

                if log_fp_on:
                    if int(row[0]) <= self.FrameTrim:
                        # only use data subframe == 0
                        if row[1] == "0":
                            # self.force_plate_list.append(self.ForcePlatePack(row))
                            self.force_plate_list.append(
                                ForceState(self.ForcePlatePack((row))))

                if log_vp_on:
                    if int(row[0]) <= self.FrameTrim:
                        # self.vicon_point_list.append(self.viconPointPack(row))
                        self.vicon_point_list.append(
                            PointState(self.viconPointPack(row)))

                        if row[self.vidx_trigger] != "" and not triggered:
                            self.vicon_trigger_frame = int(row[0])
                            triggered = True
            # print("\n")
            print("Vicon Data Read From: ", vicon_filepath)
            print("Trimmed Size: ", len(self.force_plate_list))

            for i in range(len(self.force_plate_list)):
                if self.vicon_point_list[i].frame_idx == self.vicon_trigger_frame:
                    self.vicon_trigger_idx = i

                if not self.vicon_point_list[i].Error:
                    self.vicon_data_list.append(
                        [self.force_plate_list[i].frame_idx*0.002, self.force_plate_list[i], self.vicon_point_list[i]])
                else:
                    self.vicon_data_list.append(
                        [self.force_plate_list[i].frame_idx*0.002, self.force_plate_list[i], self.vicon_point_list[i-1]])

    def viconPointPack(self, row):
        frame_idx = int(row[0])
        frame_sub = int(row[1])
        try:
            C1 = np.array(row[self.vidx_C1:self.vidx_C1+3], dtype=float)/1000
            C2 = np.array(row[self.vidx_C2:self.vidx_C2+3], dtype=float)/1000
            C3 = np.array(row[self.vidx_C3:self.vidx_C3+3], dtype=float)/1000
            AO = np.array(row[self.vidx_AO:self.vidx_AO+3], dtype=float)/1000
            AG = np.array(row[self.vidx_AG:self.vidx_AG+3], dtype=float)/1000
            BO = np.array(row[self.vidx_BO:self.vidx_BO+3], dtype=float)/1000
            BG = np.array(row[self.vidx_BG:self.vidx_BG+3], dtype=float)/1000
            CO = np.array(row[self.vidx_CO:self.vidx_CO+3], dtype=float)/1000
            CG = np.array(row[self.vidx_CG:self.vidx_CG+3], dtype=float)/1000
            DO = np.array(row[self.vidx_DO:self.vidx_DO+3], dtype=float)/1000
            DG = np.array(row[self.vidx_DG:self.vidx_DG+3], dtype=float)/1000
            return [frame_idx, frame_sub, C1, C2, C3, AO, AG, BO, BG, CO, CG, DO, DG]
        except:
            warnings.warn("[Frame %s] Point loss in Vicon Data" % frame_idx)
            return "Empty"

    def ForcePlatePack(self, row):
        frame_idx = int(row[0])
        frame_sub = int(row[1])
        F1 = np.array(row[self.vidx_F1:self.vidx_F1+3], dtype=float)
        M1 = np.array(row[self.vidx_F1+3:self.vidx_F1+6], dtype=float)
        CP1 = np.array(row[self.vidx_F1+6:self.vidx_F1+9], dtype=float)

        F2 = np.array(row[self.vidx_F2:self.vidx_F2+3], dtype=float)
        M2 = np.array(row[self.vidx_F2+3:self.vidx_F2+6], dtype=float)
        CP2 = np.array(row[self.vidx_F2+6:self.vidx_F2+9], dtype=float)

        F3 = np.array(row[self.vidx_F3:self.vidx_F3+3], dtype=float)
        M3 = np.array(row[self.vidx_F3+3:self.vidx_F3+6], dtype=float)
        CP3 = np.array(row[self.vidx_F3+6:self.vidx_F3+9], dtype=float)

        F4 = np.array(row[self.vidx_F4:self.vidx_F4+3], dtype=float)
        M4 = np.array(row[self.vidx_F4+3:self.vidx_F4+6], dtype=float)
        CP4 = np.array(row[self.vidx_F4+6:self.vidx_F4+9], dtype=float)

        return [frame_idx, frame_sub, F1, M1, CP1, F2, M2, CP2, F3, M3, CP3, F4, M4, CP4]

    def importSbrioData(self):
        trigger = False
        print("Import SBRIO Data ...")
        cnt = 0
        with open(self.sbrio_path, newline='') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                try:
                    if not trigger:
                        if int(row[1]) == 1:
                            trigger = True
                            self.sbrio_trigger_seq = int(row[0])
                            self.sbrio_trigger_idx = cnt
                    module_data = []
                    motor_AR = MotorState(float(row[self.sidx_AR_cmd_pos]), float(row[self.sidx_AR_cmd_torq]), float(
                        self.sidx_AR_cmd_kp), float(self.sidx_AR_cmd_kd), float(self.sidx_AR_rpy_pos), float(self.sidx_AR_rpy_torq), float(self.sidx_pb_AR_V), float(self.sidx_pb_AR_I))
                    motor_AL = MotorState(float(row[self.sidx_AL_cmd_pos]), float(row[self.sidx_AL_cmd_torq]), float(
                        self.sidx_AL_cmd_kp), float(self.sidx_AL_cmd_kd), float(self.sidx_AL_rpy_pos), float(self.sidx_AL_rpy_torq), float(self.sidx_pb_AL_V), float(self.sidx_pb_AL_I))
                    mod_A = ModuleState(motor_AR, motor_AL)

                    motor_BR = MotorState(float(row[self.sidx_BR_cmd_pos]), float(row[self.sidx_BR_cmd_torq]), float(
                        self.sidx_BR_cmd_kp), float(self.sidx_BR_cmd_kd), float(self.sidx_BR_rpy_pos), float(self.sidx_BR_rpy_torq), float(self.sidx_pb_BR_V), float(self.sidx_pb_BR_I))
                    motor_BL = MotorState(float(row[self.sidx_BL_cmd_pos]), float(row[self.sidx_BL_cmd_torq]), float(
                        self.sidx_BL_cmd_kp), float(self.sidx_BL_cmd_kd), float(self.sidx_BL_rpy_pos), float(self.sidx_BL_rpy_torq), float(self.sidx_pb_BL_V), float(self.sidx_pb_BL_I))
                    mod_B = ModuleState(motor_BR, motor_BL)

                    motor_CR = MotorState(float(row[self.sidx_CR_cmd_pos]), float(row[self.sidx_CR_cmd_torq]), float(
                        self.sidx_CR_cmd_kp), float(self.sidx_CR_cmd_kd), float(self.sidx_CR_rpy_pos), float(self.sidx_CR_rpy_torq), float(self.sidx_pb_CR_V), float(self.sidx_pb_CR_I))
                    motor_CL = MotorState(float(row[self.sidx_CL_cmd_pos]), float(row[self.sidx_CL_cmd_torq]), float(
                        self.sidx_CL_cmd_kp), float(self.sidx_CL_cmd_kd), float(self.sidx_CL_rpy_pos), float(self.sidx_CL_rpy_torq), float(self.sidx_pb_CL_V), float(self.sidx_pb_CL_I))
                    mod_C = ModuleState(motor_CR, motor_CL)

                    motor_DR = MotorState(float(row[self.sidx_DR_cmd_pos]), float(row[self.sidx_DR_cmd_torq]), float(
                        self.sidx_DR_cmd_kp), float(self.sidx_DR_cmd_kd), float(self.sidx_DR_rpy_pos), float(self.sidx_DR_rpy_torq), float(self.sidx_pb_DR_V), float(self.sidx_pb_DR_I))
                    motor_DL = MotorState(float(row[self.sidx_DL_cmd_pos]), float(row[self.sidx_DL_cmd_torq]), float(
                        self.sidx_DL_cmd_kp), float(self.sidx_DL_cmd_kd), float(self.sidx_DL_rpy_pos), float(self.sidx_DL_rpy_torq), float(self.sidx_pb_DL_V), float(self.sidx_pb_DL_I))
                    mod_D = ModuleState(motor_DR, motor_DL)

                    module_data.append(int(row[0])*0.001)
                    module_data.append(mod_A)
                    module_data.append(mod_B)
                    module_data.append(mod_C)
                    module_data.append(mod_D)
                    self.module_data_list.append(module_data)
                    cnt += 1
                except:
                    pass

    def importImuData(self):
        df = pd.read_csv(self.orin_path)

        tstamp_sec = df["robot_state/imu.msg.header.timestamp_sec"]
        tstamp_usec = df["robot_state/imu.msg.header.timestamp_usec"]
        orin_trigger = df["robot_state/t265.msg.header.seq"]
        acc_x = df["robot_state/imu.msg.accel.x"]
        acc_y = df["robot_state/imu.msg.accel.y"]
        acc_z = df["robot_state/imu.msg.accel.z"]

        w_x = df["robot_state/imu.msg.angular_velocity.x"]
        w_y = df["robot_state/imu.msg.angular_velocity.y"]
        w_z = df["robot_state/imu.msg.angular_velocity.z"]

        att_x = df["robot_state/imu.msg.attitude.x"]
        att_y = df["robot_state/imu.msg.attitude.y"]
        att_z = df["robot_state/imu.msg.attitude.z"]
        att_w = df["robot_state/imu.msg.attitude.w"]

        t_ = 0
        t_start = False
        t_init = 0
        trigger = False
        for i in range(tstamp_sec.shape[0]):
            if not t_start:
                if not np.isnan(tstamp_sec[i] + tstamp_usec[i] * 10e-6):
                    t_start = True
                    t_init = tstamp_sec[i] + tstamp_usec[i] * 10e-6
            else:
                t_ = tstamp_sec[i] + tstamp_usec[i] * 10e-6 - t_init

                if not trigger:
                    if orin_trigger[i] == True:
                        trigger = True
                        self.imu_trigger_tstamp = t_
                        self.imu_trigger_idx = i

                imu_ = ImuState(t_, np.array([acc_x[i], acc_y[i], acc_z[i]]), np.array(
                    [w_x[i], w_y[i], w_z[i]]), np.array([att_x[i], att_y[i], att_z[i], att_w[i]]))

                self.imu_data_list.append([t_, imu_])

    def alignData(self):
        self.trimmed_sbrio = self.module_data_list[self.sbrio_trigger_idx:][:]
        self.trimmed_imu = self.imu_data_list[self.imu_trigger_idx:][:]
        self.trimmed_vicon = self.vicon_data_list[self.vicon_trigger_idx:][:]
        # print("vt = ", self.vicon_trigger_idx)

        sbrio_t0 = self.trimmed_sbrio[0][0]
        imu_t0 = self.trimmed_imu[0][0]
        vicon_t0 = self.trimmed_vicon[0][0]
        print("--")
        print("vicon tspan: ", self.trimmed_vicon[0][0], self.trimmed_vicon[-1][0])
        print("vicon triggered data length: ", len(self.trimmed_vicon))
        print("--")
        print("sbrio tspan: ", self.trimmed_sbrio[0][0], self.trimmed_sbrio[-1][0])
        print("sbrio triggered data length: ", len(self.trimmed_sbrio))
        print("--")
        print("imu tspan: ", self.trimmed_imu[0][0], self.trimmed_imu[-1][0])
        print("imu triggered data length: ", len(self.trimmed_imu))
        print("--")

        # Align Vicon and Sbrio
        v_idx = 0
        o_idx = 0
        for i in range(len(self.trimmed_sbrio)):
            s_data = self.trimmed_sbrio[i][1:]

            if v_idx+1 < len(self.trimmed_vicon):
                t_sb = round(self.trimmed_sbrio[i][0] - sbrio_t0, 4)
                t_nvi = round(self.trimmed_vicon[v_idx+1][0] - vicon_t0, 4)

                if t_sb < t_nvi:
                    v_data = self.trimmed_vicon[v_idx][1:]
                else:
                    v_idx += 1
                    v_data = self.trimmed_vicon[v_idx][1:]
            else:
                break

            if o_idx+1 < len(self.trimmed_imu):
                t_sb = round(self.trimmed_sbrio[i][0] - sbrio_t0, 4)
                t_no = round(self.trimmed_imu[o_idx+1][0] - imu_t0, 4)
                if t_sb < t_no:
                    o_data = self.trimmed_imu[o_idx][1]
                else:
                    o_idx += 1
                    o_data = self.trimmed_imu[o_idx][1]
            else:
                break

            self.aligned_full_data.append([t_sb, s_data, o_data, v_data])
        print("Aligned Data tspan: ", self.aligned_full_data[0][0], self.aligned_full_data[-1][0])
        print("Aligned Data length: ", len(self.aligned_full_data))
        


if __name__ == '__main__':
    expdata = ExperimentData(sbrio_filepath, vicon_filepath, orin_filepath)
    expdata.importViconData()
    # print(expdata.vicon_data_list[0])
    expdata.importSbrioData()
    # print(expdata.module_data_list[10])
    expdata.importImuData()
    expdata.alignData()
    pass
