import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import math

# 1. 加载模型
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# 2. 获取无人机物理参数
body_id = model.body('x2').id
mass = model.body_mass[body_id]
gravity = -model.opt.gravity[2]

try:
    site1 = model.site('thrust1').pos
    site2 = model.site('thrust2').pos
    site3 = model.site('thrust3').pos
    site4 = model.site('thrust4').pos
    Lx = np.mean([abs(site1[0]), abs(site2[0]), abs(site3[0]), abs(site4[0])])
    Ly = np.mean([abs(site1[1]), abs(site2[1]), abs(site3[1]), abs(site4[1])])
    print(f"臂长 Lx: {Lx:.3f}, Ly: {Ly:.3f}")
except KeyError:
    Lx, Ly = 0.14, 0.18

# 3. 轨迹参数
trajectory_radius = 0.8
trajectory_omega = 0.4
trajectory_z_amp = 0.3
trajectory_z_omega = 0.3
trajectory_center_x = 1.0
trajectory_center_y = 1.0
trajectory_center_z = 1.0

# 4. 控制器参数
k1 = 2.0       # 位置环收敛增益
k2 = 5.0       # 速度环收敛增益
tau = 0.05     # 滤波器时间常数

Kp_att = 120.0
Kd_att = 30.0
Ki_att = 3.0
gamma_att = 0.3
max_tilt = 0.6

dt = 0.01

# 获取惯性矩阵
inertia = model.body_inertia[body_id]
Jx, Jy, Jz = inertia[0], inertia[1], inertia[2]

# --- 关键修复：初始化滤波器状态 ---
mujoco.mj_forward(model, data)
x0, y0, z0 = data.qpos[0], data.qpos[1], data.qpos[2]

# 计算初始虚拟控制量，防止启动瞬间滤波器滞后导致的掉落
init_target_x = trajectory_center_x + trajectory_radius
init_target_y = trajectory_center_y
init_target_z = trajectory_center_z

alpha_f_x = -k1 * (x0 - init_target_x)
alpha_f_y = -k1 * (y0 - init_target_y)
alpha_f_z = -k1 * (z0 - init_target_z)
# -------------------------------

# alpha_f_x = 0
# alpha_f_y = 0
# alpha_f_z = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("运行中: 动态面控制 (DSC) - 已修复初始化")
    
    while viewer.is_running():
        t = data.time
        
        # ---- 1. 计算期望轨迹 ----
        target_x = trajectory_center_x + trajectory_radius * math.cos(trajectory_omega * t)
        target_y = trajectory_center_y + trajectory_radius * math.sin(trajectory_omega * t)
        target_z = trajectory_center_z + trajectory_z_amp * math.sin(trajectory_z_omega * t)
        
        target_vx = -trajectory_radius * trajectory_omega * math.sin(trajectory_omega * t)
        target_vy = trajectory_radius * trajectory_omega * math.cos(trajectory_omega * t)
        target_vz = trajectory_z_amp * trajectory_z_omega * math.cos(trajectory_z_omega * t)
        
        # ---- 2. 读取状态 ----
        x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
        vx, vy, vz = data.qvel[0], data.qvel[1], data.qvel[2]
        
        quat = data.qpos[3:7]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        wx, wy, wz = data.qvel[3], data.qvel[4], data.qvel[5]
        
        # ---- 3. 位置外环 (DSC) ----
        z1_x = x - target_x
        z1_y = y - target_y
        z1_z = z - target_z
        
        alpha_x = target_vx - k1 * z1_x
        alpha_y = target_vy - k1 * z1_y
        alpha_z = target_vz - k1 * z1_z
        
        alpha_f_dot_x = -(alpha_f_x - alpha_x) / tau
        alpha_f_dot_y = -(alpha_f_y - alpha_y) / tau
        alpha_f_dot_z = -(alpha_f_z - alpha_z) / tau
        
        alpha_f_x += alpha_f_dot_x * dt
        alpha_f_y += alpha_f_dot_y * dt
        alpha_f_z += alpha_f_dot_z * dt
        
        z2_x = vx - alpha_f_x
        z2_y = vy - alpha_f_y
        z2_z = vz - alpha_f_z
        
        # 控制律 (移除了自适应项以保持稳定)
        ux = -k2 * z2_x + alpha_f_dot_x - z1_x
        uy = -k2 * z2_y + alpha_f_dot_y - z1_y
        uz = -k2 * z2_z + alpha_f_dot_z - z1_z
        
        # ---- 4. 推力与姿态解算 ----
        F_total = mass * (uz + gravity)
        psi = yaw  # 当前偏航角（对应公式中的ψ）

        # 计算期望姿态角（替换原公式，使用总推力F_total作为T）
        target_pitch = - (mass / F_total) * (ux * np.cos(psi) + uy * np.sin(psi))  # 期望俯仰角θ_d
        target_roll = - (mass / F_total) * (ux * np.sin(psi) - uy * np.cos(psi))   # 期望滚转角φ_d
        target_yaw = 0.0  # 偏航角保持0（对应公式中的ψ_d=0）
        
        target_pitch = np.clip(ux / gravity, -max_tilt, max_tilt)
        target_roll = np.clip(-uy / gravity, -max_tilt, max_tilt)
        # target_yaw = 0.0
        
        # ---- 5. 姿态内环 ----
        err_roll = target_roll - roll
        err_pitch = target_pitch - pitch
        err_yaw = target_yaw - yaw
        
        tau_roll = Jx * (Kp_att * err_roll - Kd_att * wx)
        tau_pitch = Jy * (Kp_att * err_pitch - Kd_att * wy)
        tau_yaw = Jz * (10.0 * err_yaw - 5.0 * wz)
        
        # ---- 6. 混控分配 ----
        motor_positions = np.array([[-0.14, -0.18], [-0.14, 0.18], [0.14, 0.18], [0.14, -0.18]])
        rotor_signs = np.array([1, -1, 1, -1])
        kappa = 0.01
        
        A = np.zeros((4, 4))
        for i, (x_m, y_m) in enumerate(motor_positions):
            A[0, i] = 1.0; A[1, i] = y_m; A[2, i] = -x_m; A[3, i] = -kappa * rotor_signs[i]
            
        mix_inv = np.linalg.inv(A)
        wrench = np.array([F_total, tau_roll, tau_pitch, tau_yaw])
        m1, m2, m3, m4 = mix_inv @ wrench
        
        motor_forces = np.clip([m1, m2, m3, m4], 0, 13)
        data.ctrl[:] = motor_forces
        
        # ---- 7. 步进 ----
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)
        
        # ---- 8. 打印跟踪误差 (参照原代码格式) ----
        if int(data.time * 10) % 10 == 0:
            print(f"目标:({target_x:.2f},{target_y:.2f},{target_z:.2f}) | "
                  f"实际:({x:.2f},{y:.2f},{z:.2f}) | "
                  f"误差:({z1_x:.2f},{z1_y:.2f},{z1_z:.2f})", end='\r')
