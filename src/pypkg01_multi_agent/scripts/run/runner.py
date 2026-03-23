import rospy
from pypkg01_multi_agent.scripts.env.agent.controller import UAVFleetManager
import numpy as np

def make_env(args):
    from pypkg01_multi_agent.scripts.env.environment.environment import MultiAgentEnv
    from pypkg01_multi_agent.scripts.env.scenarios.sample1 import Scenario

    scenario = Scenario(args)
    
    env = MultiAgentEnv(args, scenario)

    return env


class Runner:
    def __init__(self):
        self.args = get_args()
        rospy.init_node("test01")
        self.env = make_env(self.args)
        self.sim_timer = None
        self.fleet = UAVFleetManager(self.env.scenario.world.agent_list)
        self.task_is_done = False

    def step_cb(self, event):
        if self.task_is_done:
            return
            
        done = self.env.step()
        rospy.loginfo_throttle(2, f"编队结束标志：{done}")
        rospy.loginfo_throttle(2, f"仿真时间：{self.env.scenario.total_time}")

        if done:
            rospy.loginfo(">>> 编队任务完成，自动停止控制定时器")
            self.task_is_done = True
            self.stop_step()
            
            rospy.loginfo(">>> 同步悬停 3 秒")
            self.fleet.sync_hover_all(duration=3)

            rospy.loginfo(">>> 同步降落")
            self.fleet.sync_land_all()

            rospy.loginfo(">>> 所有任务完成")

    def start_step(self):
        for agent in self.env.scenario.world.agent_list:
            agent.cmd_mode = self.args.cmd_mode_switch
        if self.sim_timer is None:
            rospy.loginfo(">>> 启动编队控制定时器")
            self.sim_timer = rospy.Timer(rospy.Duration(self.args.dt), self.step_cb)

    def stop_step(self):
        for agent in self.env.scenario.world.agent_list:
            agent.cmd_mode = self.args.cmd_mode
        if self.sim_timer is not None:
            rospy.loginfo(">>> 停止编队控制定时器")
            self.sim_timer.shutdown()
            self.sim_timer = None


    def run(self):
        rospy.loginfo(">>> 开始多无人机同步初始化")
        self.fleet.sync_init_all()

        rospy.loginfo(">>> 计算初始编队中心")
        self.env.scenario.calculate_initial_centroid()

        rospy.loginfo(">>> 开始多无人机同步起飞")
        self.fleet.sync_takeoff_all(target_z=self.args.takeoff_target_altitude)

        rospy.loginfo(">>> 同步悬停 3 秒 (稳定机身)")
        self.fleet.sync_hover_all(duration=3)
        
        if len(self.env.scenario.world.agent_list) >= 3:
            self.start_step()
            rospy.loginfo(">>> 编队任务运行中（定时器驱动）")
        else:
            rospy.loginfo(">>> 同步降落")
            self.fleet.sync_land_all()
            rospy.loginfo(">>> 所有任务完成")

        rospy.spin()

        
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--mode', type=str, default="real", help="sim用于仿真，real用于控制真实无人机")
    parser.add_argument('--cmd_mode', type=str, default="pose", help="设置无人机起降阶段的指令模式，pose/twist/accel")
    parser.add_argument('--cmd_mode_switch', type=str, default="twist", help="设置无人机编队任务阶段的指令模式，pose/twist/accel")
    parser.add_argument('--dt', type=float, default=0.1, help="设置步长")
    parser.add_argument('--field_range', type=list, default=[-3.8, 3.8, -1.8, 1.8, 0, 1.5], help="设置主场范围")
    parser.add_argument('--agent_ids', type=list, default=[4, 5, 6], help="设置无人机编号")
    parser.add_argument('--r_safe', type=float, default=0.15, help="设置无人机安全半径")
    parser.add_argument('--twist_lim', type=list, default=[2, 2, 2, 0.1, 0.1, 0.1], help="设置无人机速度上限")
    parser.add_argument('--accel_lim', type=list, default=[0.5, 0.5, 0.5], help="设置无人机加速度上限")
    parser.add_argument('--takeoff_target_altitude', type=float, default=1.0, help="设置无人机执行任务时的高度")

    # 控制器增益参数
    parser.add_argument('--fc_k1', type=float, default=0.8, help="编队速度增益")
    parser.add_argument('--fc_k2', type=float, default=0.5, help="编队角度增益")
    parser.add_argument('--fc_k3', type=float, default=1.2, help="编队距离增益")
    parser.add_argument('--fc_k4', type=float, default=1.0, help="编队中心y轴稳定增益")

    parser.add_argument('--fc_desired_alpha_213', type=float, default=np.pi / 3, help="编队期望角度")
    parser.add_argument('--fc_desired_alpha_123', type=float, default=np.pi / 3, help="编队期望角度")
    parser.add_argument('--fc_desired_alpha_132', type=float, default=np.pi / 3, help="编队期望角度")
    
    # [修改] 稍微提高前进速度
    parser.add_argument('--fc_desired_vc', type=list, default=[0.7, 0.0], help="编队期望平移速度")

    # 通道与编队距离参数
    parser.add_argument('--fc_desired_d_13', type=float, default=1.0, help="编队期望距离")
    # [修改] 将收缩距离调整为更平衡的值
    parser.add_argument('--formation_distance_narrow', type=float, default=0.4, help="通过通道时的编队期望距离")
    parser.add_argument('--channel_center_x', type=float, default=0.0, help="通道中心x坐标")
    parser.add_argument('--channel_length', type=float, default=1.2, help="通道长度")
    parser.add_argument('--channel_width', type=float, default=1.3, help="通道宽度")
    
    parser.add_argument('--anticipation_distance', type=float, default=1.0, help="编队开始缩小的提前距离")

    # 在ros中使用argparse, 必须要加
    args, _ = parser.parse_known_args()
    return args
