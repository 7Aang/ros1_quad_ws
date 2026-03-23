import numpy as np
from scipy.integrate import odeint
import rospy

# 多智能体世界
class World:
    """
    # multi-agent world
    """
    def __init__(self):
        self.dt = None
        self.args = None
        self.agent_list = []
        self.initial_centroid_y = None
        self.field_range = [-4, 4, -2.5, 2.5, -0.1, 2]

    @property
    def field_center(self):
        center_x = (self.field_range[0] + self.field_range[1]) / 2
        center_y = (self.field_range[2] + self.field_range[3]) / 2
        center_z = 0
        return center_x, center_y, center_z

    @property
    def field_half_size(self):
        length = (self.field_range[1] - self.field_range[0]) / 2
        width = (self.field_range[3] - self.field_range[2]) / 2
        height = (self.field_range[5] - self.field_range[4]) / 2
        return length, width, height
    
    def check_collision(self):
        for idx_a, agent_a in enumerate(self.agent_list):
            if agent_a.state.crash_bound or agent_a.state.crash_agent:
                agent_a.state.movable = False
                continue
            for idx_b, agent_b in enumerate(self.agent_list):
                if idx_a == idx_b:
                    continue
                pos_a = agent_a.state.pose.pose.position
                pos_b = agent_b.state.pose.pose.position
                dist = np.sqrt(
                    (pos_a.x - pos_b.x) ** 2 +
                    (pos_a.y - pos_b.y) ** 2 +
                    (pos_a.z - pos_b.z) ** 2
                )
                if dist <= agent_a.r_safe + agent_b.r_safe:
                    rospy.logwarn(f"碰撞检测: {agent_a.agent_id} 与 {agent_b.agent_id} 发生碰撞!")
                    agent_a.state.crash_agent = True
                    agent_a.state.movable = False
                    break
            
            if not agent_a.state.movable:
                continue

            pos_a = agent_a.state.pose.pose.position
            if pos_a.x <= self.field_range[0] or \
                    pos_a.x >= self.field_range[1] or \
                    pos_a.y <= self.field_range[2] or \
                    pos_a.y >= self.field_range[3] or \
                    pos_a.z <= self.field_range[4] or \
                    pos_a.z >= self.field_range[5]:
                rospy.logwarn(f"碰撞检测: {agent_a.agent_id} 撞到场地边界!")
                agent_a.state.crash_bound = True
                agent_a.state.movable = False
    
    def check_dynamics_constraints(self):
        for agent in self.agent_list:
            vel_lim = agent.twist_lim
            accel_lim = agent.accel_lim
            linear_vel = agent.state.target_twist.twist.linear
            angular_vel = agent.state.target_twist.twist.angular
            linear_acc = agent.state.target_accel.vector
            vx, vy, vz = linear_vel.x, linear_vel.y, linear_vel.z
            wx, wy, wz = angular_vel.x, angular_vel.y, angular_vel.z
            ax, ay, az = linear_acc.x, linear_acc.y, linear_acc.z
            vel = np.array([vx, vy, vz, wx, wy, wz])
            accel_vals = np.array([ax, ay, az])
            vel_clamped = np.clip(vel, -np.array(vel_lim), np.array(vel_lim))
            accel_clamped = np.clip(accel_vals, -np.array(accel_lim), np.array(accel_lim))
            linear_vel.x, linear_vel.y, linear_vel.z = vel_clamped[:3]
            angular_vel.x, angular_vel.y, angular_vel.z = vel_clamped[3:]
            linear_acc.x, linear_acc.y, linear_acc.z = accel_clamped[:3]

    def update_one_sim_step(self):
        t = [0, self.dt]
        for agent in self.agent_list:
            if agent.state.movable:
                if agent.cmd_mode == "pose":
                    initial_state = [
                        agent.state.pose.pose.position.x,
                        agent.state.pose.pose.position.y,
                        agent.state.pose.pose.position.z,
                        agent.state.twist.twist.linear.x,
                        agent.state.twist.twist.linear.y,
                        agent.state.twist.twist.linear.z,
                    ]
                    next_state = odeint(f, initial_state, t, args=(agent.action.u,))
                    agent.state.target_pose.pose.position.x = next_state[-1][0]
                    agent.state.target_pose.pose.position.y = next_state[-1][1]
                    agent.state.target_pose.pose.position.z = next_state[-1][2]
                    agent.state.target_twist.twist.linear.x = next_state[-1][3]
                    agent.state.target_twist.twist.linear.y = next_state[-1][4]
                    agent.state.target_twist.twist.linear.z = next_state[-1][5]
                    agent.state.twist.twist.linear.x = next_state[-1][3]
                    agent.state.twist.twist.linear.y = next_state[-1][4]
                    agent.state.twist.twist.linear.z = next_state[-1][5]
                if agent.cmd_mode == "twist":
                    measured_twist = np.array([
                        agent.state.twist.twist.linear.x,
                        agent.state.twist.twist.linear.y,
                        agent.state.twist.twist.linear.z,
                    ])
                    previous_target_twist = np.array([
                        agent.state.target_twist.twist.linear.x,
                        agent.state.target_twist.twist.linear.y,
                        agent.state.target_twist.twist.linear.z,
                    ])
                    if getattr(agent.controller, "has_twist_feedback", False):
                        base_twist = measured_twist
                    else:
                        base_twist = previous_target_twist

                    next_twist = base_twist + self.dt * agent.action.u
                    agent.state.target_twist.twist.linear.x = next_twist[0]
                    agent.state.target_twist.twist.linear.y = next_twist[1]
                    agent.state.target_twist.twist.linear.z = next_twist[2]

                    # When no velocity topic is available, keep an internal estimate
                    # so the velocity command can still ramp instead of restarting at zero.
                    if not getattr(agent.controller, "has_twist_feedback", False):
                        agent.state.twist.twist.linear.x = next_twist[0]
                        agent.state.twist.twist.linear.y = next_twist[1]
                        agent.state.twist.twist.linear.z = next_twist[2]
                if agent.cmd_mode == "accel":
                    agent.state.target_accel.vector.x = agent.action.u[0]
                    agent.state.target_accel.vector.y = agent.action.u[1]
                    agent.state.target_accel.vector.z = agent.action.u[2]
        self.check_dynamics_constraints()

    def update_traditional_formation_control_input(self):
        p1 = np.array([
            self.agent_list[0].state.pose.pose.position.x, 
            self.agent_list[0].state.pose.pose.position.y, 
            self.agent_list[0].state.pose.pose.position.z, 
        ])
        p2 = np.array([
            self.agent_list[1].state.pose.pose.position.x, 
            self.agent_list[1].state.pose.pose.position.y, 
            self.agent_list[1].state.pose.pose.position.z, 
        ])
        p3 = np.array([
            self.agent_list[2].state.pose.pose.position.x, 
            self.agent_list[2].state.pose.pose.position.y, 
            self.agent_list[2].state.pose.pose.position.z, 
        ])

        v1 = np.array([
            self.agent_list[0].state.twist.twist.linear.x,
            self.agent_list[0].state.twist.twist.linear.y,
            self.agent_list[0].state.twist.twist.linear.z,
        ])
        v2 = np.array([
            self.agent_list[1].state.twist.twist.linear.x,
            self.agent_list[1].state.twist.twist.linear.y,
            self.agent_list[1].state.twist.twist.linear.z,
        ])
        v3 = np.array([
            self.agent_list[2].state.twist.twist.linear.x,
            self.agent_list[2].state.twist.twist.linear.y,
            self.agent_list[2].state.twist.twist.linear.z,
        ])

        d_12 = np.linalg.norm(p1[:2] - p2[:2])
        d_13 = np.linalg.norm(p1[:2] - p3[:2])
        d_21 = np.linalg.norm(p2[:2] - p1[:2])
        d_23 = np.linalg.norm(p2[:2] - p3[:2])
        d_31 = np.linalg.norm(p3[:2] - p1[:2])
        d_32 = np.linalg.norm(p3[:2] - p2[:2])

        if any(d < 1e-6 for d in [d_12, d_13, d_21, d_23, d_31, d_32]):
            return [np.zeros(3)] * 3

        z_12 = (p2[:2] - p1[:2]) / d_21
        z_13 = (p3[:2] - p1[:2]) / d_31
        z_21 = (p1[:2] - p2[:2]) / d_12
        z_23 = (p3[:2] - p2[:2]) / d_32
        z_31 = (p1[:2] - p3[:2]) / d_13
        z_32 = (p2[:2] - p3[:2]) / d_23

        alpha_213 = np.arccos(np.clip(np.dot(z_12, z_13), -1.0, 1.0))
        alpha_123 = np.arccos(np.clip(np.dot(z_21, z_23), -1.0, 1.0))
        alpha_132 = np.arccos(np.clip(np.dot(z_31, z_32), -1.0, 1.0))

        k1 = self.args.fc_k1
        k2 = self.args.fc_k2
        k3 = self.args.fc_k3
        k4 = self.args.fc_k4

        desired_alpha_213 = self.args.fc_desired_alpha_213
        desired_alpha_123 = self.args.fc_desired_alpha_123
        desired_alpha_132 = self.args.fc_desired_alpha_132
        desired_d_13 = self.args.fc_desired_d_13
        desired_vc = np.array(self.args.fc_desired_vc)

        u1 = np.zeros(3)
        u2 = np.zeros(3)
        u3 = np.zeros(3)

        u1[:2] = -k1 * (v1[:2] - desired_vc) - k2 * (alpha_213 - desired_alpha_213) * (z_12 + z_13)
        u2[:2] = -k1 * (v2[:2] - desired_vc) - k2 * (alpha_123 - desired_alpha_123) * (z_21 + z_23)
        u3[:2] = -k1 * (v3[:2] - desired_vc) - k2 * (alpha_132 - desired_alpha_132) * (z_31 + z_32) + k3 * (d_13 - desired_d_13) * z_31

        # [修改] 将中心线对齐的目标从 Y=-0.3 调整为 Y=-0.15，以寻找平衡点
        current_centroid_y = (p1[1] + p2[1] + p3[1]) / 3
        y_error = current_centroid_y - (-0.15)
        
        u1[1] -= k4 * y_error
        u2[1] -= k4 * y_error
        u3[1] -= k4 * y_error

        u = [u1, u2, u3]
        return u

def f(state, t, u):
    ux, uy, uz = u
    x, y, z, vx, vy, vz = state
    return [vx, vy, vz, ux, uy, uz]
