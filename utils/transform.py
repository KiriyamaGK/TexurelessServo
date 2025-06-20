from math import sin,cos,sqrt,atan2,pi
import numpy as np
from scipy.spatial.transform import Rotation as R

def is_number(var):
    return isinstance(var, (int, float, complex))

def rmat2quat(rotation_matrix: np.ndarray, eps: float = 1.0e-8) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (x, y, z，w) format.

    Args:
        rotation_matrix: the rotation matrix to convert with shape (*, 3, 3).
        eps: small value to avoid zero division.

    Return:
        the rotation in quaternion with shape (*, 4).
    """
    if not isinstance(rotation_matrix, np.ndarray):
        raise TypeError(f"Input type is not a numpy.ndarray. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) array. Got {rotation_matrix.shape}")

    def safe_zero_division(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return numerator / np.maximum(denominator, eps)

    rotation_matrix_vec = rotation_matrix.reshape(*rotation_matrix.shape[:-2], 9)
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = np.moveaxis(rotation_matrix_vec, -1, 0)

    trace = m00 + m11 + m22

    def trace_positive_cond() -> np.ndarray:
        sq = np.sqrt(trace + 1.0 + eps) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return np.stack((qx, qy, qz, qw), axis=-1)

    def cond_1() -> np.ndarray:
        sq = np.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return np.stack((qx, qy, qz, qw), axis=-1)

    def cond_2() -> np.ndarray:
        sq = np.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return np.stack((qx, qy, qz, qw), axis=-1)

    def cond_3() -> np.ndarray:
        sq = np.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return np.stack((qx, qy, qz, qw), axis=-1)

    where_2 = np.where(m11 > m22, cond_2(), cond_3())
    where_1 = np.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion = np.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion

def rotation_matrix_x(theta):
    """生成绕x轴旋转theta弧度的旋转矩阵"""
    return np.array([
        [1, 0, 0],
        [0, cos(theta), -sin(theta)],
        [0, sin(theta), cos(theta)]
    ])

def rotation_matrix_y(theta):
    """生成绕y轴旋转theta弧度的旋转矩阵"""
    return np.array([
        [cos(theta), 0, sin(theta)],
        [0, 1, 0],
        [-sin(theta), 0, cos(theta)]
    ])

def rotation_matrix_z(theta):
    """生成绕z轴旋转theta弧度的旋转矩阵,弧度"""
    return np.array([
        [cos(theta), -sin(theta), 0],
        [sin(theta), cos(theta), 0],
        [0, 0, 1]
    ])
def from_mov_vel2rots(mov_vel):
    "微小旋转量变成旋转矩阵"
    pi = np.pi
    Rx = rotation_matrix_x(mov_vel[0] * pi / 180)
    Ry = rotation_matrix_y(mov_vel[1] * pi / 180)
    Rz = rotation_matrix_z(mov_vel[2] * pi / 180)
    return Rx, Ry, Rz

def get_rot_matrix_from_delta(pose,Rx, Ry, Rz):
    mat = euler2rot([pose[3] * pi / 180, pose[4] * pi / 180, pose[5] * pi / 180])
    return Rz @ Ry @ Rx @ mat

def euler2rot(rpy):
    '''欧拉角转换为旋转矩阵
    输入为依次绕定轴x,y,z旋转
    rx,ry,rz依次记为r,p,y'''
    roll, pitch, yaw = rpy[0:3]
    r11 = cos(pitch) * cos(yaw)
    r12 = cos(yaw) * sin(roll) * sin(pitch) - cos(roll) * sin(yaw)
    r13 = sin(roll) * sin(yaw) + cos(roll) * cos(yaw) * sin(pitch)
    r21 = cos(pitch) * sin(yaw)
    r22 = cos(roll) * cos(yaw) + sin(roll) * sin(pitch) * sin(yaw)
    r23 = cos(roll) * sin(pitch) * sin(yaw) - sin(roll) * cos(yaw)
    r31 = -sin(pitch)
    r32 = cos(pitch) * sin(roll)
    r33 = cos(roll) * cos(pitch)
    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]])

def euler2Matrix(Euler):
    "末端位姿，欧拉角转齐次变换矩阵，输入角度"
    Matrix = np.zeros((4, 4))
    Matrix[0, 3] = Euler[0]
    Matrix[1, 3] = Euler[1]
    Matrix[2, 3] = Euler[2]
    Matrix[3, 0] = 0
    Matrix[3, 1] = 0
    Matrix[3, 2] = 0
    Matrix[3, 3] = 1
    Matrix[0:3, 0:3] = euler2rot(np.array([Euler[3] / 180 * np.pi, Euler[4] / 180 * np.pi, Euler[5] / 180 * np.pi]))
    return Matrix

def rmat2euler_degree(Matrix: np.array):
    '''旋转矩阵转欧拉角
    依次绕定轴x,y,z动轴z,y,x旋转/
    输出角度制'''
    pi = np.pi
    Euler = [0, 0, 0]
    r11 = Matrix[0, 0]
    r21 = Matrix[1, 0]
    r31 = Matrix[2, 0]
    if Matrix[0, 1] < 0.000001:
        r12 = 0
    else:
        r12 = Matrix[0, 1]
    r22 = Matrix[1, 1]
    r32 = Matrix[2, 1]
    r33 = Matrix[2, 2]

    if abs(r11 * r11 + r21 * r21) < 0.000001:
        if r31 > 0:
            Euler[2] = 0
            Euler[1] = -pi / 2
            Euler[0] = -atan2(r12, r22)
            # 345交换过
        else:
            Euler[2] = 0
            Euler[1] = pi / 2
            Euler[0] = atan2(r12, r22)
            # 345交换过
    else:
        cb = sqrt(r11 * r11 + r21 * r21)
        Euler[2] = atan2(r21, r11)
        Euler[1] = atan2(-r31, cb)
        Euler[0] = atan2(r32, r33)
        # 345交换过
    Euler[0] = Euler[0] / pi * 180
    Euler[1] = Euler[1] / pi * 180
    Euler[2] = Euler[2] / pi * 180
    return Euler

def rmat2euler_rz_degree(Matrix: np.array):
    '''旋转矩阵转欧拉角
    依次绕定轴x,y,z旋转
    输出rz,0-360度，且取不到360'''
    pi = np.pi
    Euler = [0, 0, 0]
    r11 = Matrix[0, 0]
    r21 = Matrix[1, 0]
    if abs(r11 * r11 + r21 * r21) < 0.000001:
        Euler[2] = 0
    else:
        Euler[2] = atan2(r21, r11)
    #postprocess
    assert -pi<=Euler[2]<=pi
    if Euler[2] < 0:
        Euler[2] += pi*2
    Euler[2] = Euler[2] / pi * 180
    return Euler[2]

def get_inverse(mat):
    '''
    求四维旋转矩阵的逆
    '''
    R = mat[0:3, 0:3]  # 提取旋转矩阵
    t = mat[0:3, 3]  # 提取平移向量

    R_t = R.T  # 旋转矩阵的转置
    new_mat = np.eye(4, 4)  # 创建一个 4x4 的单位矩阵

    new_mat[0:3, 0:3] = R_t  # 将旋转矩阵的转置赋值给新矩阵的前 3x3 部分

    # 计算 -R_t @ t，并将结果赋值给新矩阵的第 4 行的前 3 个元素
    new_mat[0:3, 3] = -R_t @ t.flatten()

    return new_mat


def project_XYZw_to_uv(intr: np.array,cwT:np.array,XYZw:np.array):
    assert XYZw.shape == (3,)
    intr=np.concatenate((intr,np.array([[0],[0],[0]])),axis=1)
    XYZw=np.concatenate((XYZw,np.array([1])),axis=0)
    XYZc=XYZw@cwT.T
    Zc=XYZc[2]
    uv=XYZc@intr.T/Zc
    uv=uv[0:2]
    uv[0]=int(uv[0])
    uv[1]=int(uv[1])
    return uv

def make_an_angle_in_180(ang,max_attempts_num=10):
    attempts_num=0
    while True:
        if attempts_num >= max_attempts_num:
            raise RuntimeError("Too many attempts")
        if abs(ang) >180:
            if ang>0:
                ang -= 360
            else:
                ang += 360
        if abs(ang) <=180:
            return ang
        attempts_num+=1

def rot_angle_normalization(ang,max_attempts_num=10): #轉換到0-360之間，-180和180之間有間斷點，不方便學習
    attempts_num = 0
    while True:
        if attempts_num >= max_attempts_num:
            raise RuntimeError("Too many attempts")
        if ang>=360:
            ang-=360
        if ang<0:
            ang+=360
        if ang>=0 and ang <360:
            return ang
        attempts_num += 1

def compute_pos_error(pos_cur,pos_tar):
    '''
    :param pos_cur:mm/m ,deg [6,]
    :param pos_tar:mm/m ,deg [6,]
    :return:
    delta_pos:mm/m,deg [6,]
    '''
    delta_pos = np.zeros(6)

    T_cur =np.eye(4)
    T_tar =np.eye(4)
    T_cur[0:3,0:3]=R.from_rotvec(pos_cur[3:]/180*np.pi).as_matrix()
    T_cur[0:3,3]=pos_cur[0:3]
    T_tar[0:3, 0:3] = R.from_rotvec(pos_tar[3:] / 180 * np.pi).as_matrix()
    T_tar[0:3, 3] = pos_tar[0:3]
    dT = np.linalg.inv(T_cur) @ T_tar

    delta_pos[3:] = np.linalg.norm(R.from_matrix(dT[:3, :3]).as_rotvec()) / np.pi * 180
    delta_pos[0:3] = np.linalg.norm(dT[:3, 3])

    return delta_pos

def error_pos_transform(error_pos):
    '''
    :param error_pos: [dx,dy,dz,theta(in axis-angle)] [6,]
    :return: [delta_xyz,delta_z,delta_theta][3,]
    '''
    return np.array([np.linalg.norm(error_pos[0:3]),abs(error_pos[2]),np.linalg.norm(error_pos[3:])])
