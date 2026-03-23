from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode, CommandBoolRequest, SetModeRequest
from std_msgs.msg import Header
import rospy
from copy import deepcopy
import numpy as np
from threading import Event, Thread
from nav_msgs.msg import Path



class Controller:
    def __init__(self, agent):
        self.agent = agent
        self.uav_name = agent.agent_id

        self.current_state = State()
        self.initial_pose = None
        self.has_initial_pose = False

        self.current_pose = None
        self.target_pose = None
        self.temp_target_pose = None

        self.current_twist = None
        self.target_twist = None
        self.temp_target_twist = None

        self.current_accel = None
        self.target_accel = None
        self.temp_target_accel = None

        self.landing_pose = None
        self.local_pose = None
        self.keep_publishing = False
        self.publisher_thread = None
        self.has_twist_feedback = False

        self.is_landing = False

        self._init_state()

        self.state_sub = rospy.Subscriber(f"/{self.uav_name}/mavros/state", State, self.state_cb)
        self.pose_sub = rospy.Subscriber(f'/{self.uav_name}/mavros/vision_pose/pose', PoseStamped, self.make_pose_cb(self.agent))
        self.local_pose_sub = rospy.Subscriber(f'/{self.uav_name}/mavros/local_position/pose', PoseStamped, self.local_pose_cb)
        self.twist_sub = rospy.Subscriber(
            f'/{self.uav_name}/mavros/local_position/velocity_local',
            TwistStamped,
            self.make_twist_cb(self.agent)
        )
        
        self.pub_rate = rospy.Rate(20)
        self.rate = rospy.Rate(20)
        self.cmd_pose_pub = rospy.Publisher(f'/{self.uav_name}/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.cmd_twist_pub = rospy.Publisher(f'/{self.uav_name}/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        self.cmd_accel_pub = rospy.Publisher(f'/{self.uav_name}/mavros/setpoint_accel/accel', Vector3Stamped, queue_size=10)
        self.path_pub = rospy.Publisher(f"/{self.uav_name}/trajectory", Path, queue_size=10)

        self.set_mode_client = rospy.ServiceProxy(f"/{self.uav_name}/mavros/set_mode", SetMode)
        self.set_mode_client.wait_for_service()
        self.arming_client = rospy.ServiceProxy(f'/{self.uav_name}/mavros/cmd/arming', CommandBool)
        self.arming_client.wait_for_service()

        self.timer = rospy.Timer(rospy.Duration(0.1), self.update_target_cmd)


    def init(self):
        self._wait_for_connection()
        self._wait_for_initial_pose()
        self._wait_for_ekf_ready()

    def make_pose_cb(self, agent):
        def cb(msg):
            self.current_pose = deepcopy(msg)
            agent.state.pose = deepcopy(msg)
            pos = self.current_pose.pose.position
            
            rospy.loginfo_throttle(
                2, 
                f"[{self.uav_name}] 当前坐标: "
                f"x={pos.x:.2f}, "
                f"y={pos.y:.2f}, "
                f"z={pos.z:.2f}"
            )
            if not self.has_initial_pose and abs(pos.z) > 0.01:
                self.initial_pose = deepcopy(msg)

                self.has_initial_pose = True
                rospy.loginfo(
                    f"[{self.uav_name}] 记录初始位置: "
                    f"x={pos.x:.2f}, "
                    f"y={pos.y:.2f}, "
                    f"z={pos.z:.2f}"
                )
        return cb
    
    def make_twist_cb(self, agent):
        def cb(msg):
            self.current_twist = deepcopy(msg)
            self.has_twist_feedback = True
            agent.state.twist = deepcopy(msg)
        return cb

    def make_accel_cb(self, agent):
        def cb(msg):
            agent.state.accel = deepcopy(msg)
        return cb
    
    def state_cb(self, msg):
        self.current_state = deepcopy(msg)
        rospy.loginfo_throttle(
            2, 
            f"[{self.uav_name}] 状态: "
            f"armed={msg.armed}, "
            f"mode={msg.mode}, "
            f"system_status={msg.system_status}"
        )

    def local_pose_cb(self, msg):
        self.local_pose = deepcopy(msg)
        pos = self.local_pose.pose.position
        rospy.loginfo_throttle(
            2, 
            f"[{self.uav_name}] EKF融合坐标: "
            f"x={pos.x:.2f}, "
            f"y={pos.y:.2f}, "
            f"z={pos.z:.2f}"
        ) 


    def start_publisher_thread(self):
        if not self.publisher_thread:
            self.keep_publishing = True
            self.publisher_thread = Thread(target=self._publisher_loop)
            self.publisher_thread.daemon = True
            self.publisher_thread.start()
    
    def _set_cmd_direct(self, new_cmd):
        if self.agent.cmd_mode == "pose":
            self.temp_target_pose = deepcopy(new_cmd)
        if self.agent.cmd_mode == "twist":
            self.temp_target_twist = deepcopy(new_cmd)
        if self.agent.cmd_mode == "accel":
            self.temp_target_accel = deepcopy(new_cmd)

    def set_target_cmd(self, new_cmd):
        if self.is_landing:
            rospy.logwarn_throttle(1.0, f"[{self.uav_name}] 正在降落，忽略外部指令")
            return
        self._set_cmd_direct(new_cmd)

    def update_target_cmd(self, event):
        if self.agent.cmd_mode == "pose":
            self.target_pose = deepcopy(self.temp_target_pose)
            
            z = self.target_pose.pose.position.z
            if z > 1.2:
                rospy.logwarn_throttle(1.0, f"[{self.uav_name}] z 高度 {z:.2f} 超出限制，已强制限制为 1.2m")
                self.target_pose.pose.position.z = 1.0
            self.target_pose.header.stamp = rospy.Time.now()
            self.agent.state.path.header.frame_id = "world"
            self.publish_path(self.agent.state.pose)
        if self.agent.cmd_mode == "twist":
            self.target_twist = deepcopy(self.temp_target_twist)
            self.target_twist.twist.linear.z = 0.0
            self.target_twist.twist.angular.x = 0.0
            self.target_twist.twist.angular.y = 0.0
            self.target_twist.twist.angular.z = 0.0
            self.target_twist.header.stamp = rospy.Time.now()
            self.agent.state.path.header.frame_id = "world"
            self.publish_path(self.agent.state.pose)

        if self.agent.cmd_mode == "accel":
            self.target_accel = deepcopy(self.temp_target_accel)
            self.target_accel.vector.z = 9.8
            self.target_accel.header.stamp = rospy.Time.now()

    def publish_path(self, current_pose):
        self.agent.state.path.poses.append(current_pose)
        self.path_pub.publish(self.agent.state.path)

    def _init_state(self):
        self.initial_pose = PoseStamped()
        self.initial_pose.header.frame_id = ""
        self.initial_pose.pose.orientation.w = 1.0
        self.current_pose = PoseStamped()
        self.current_pose.header.frame_id = ""
        self.current_pose.pose.orientation.w = 1.0
        self.current_twist = TwistStamped()
        self.current_twist.header.frame_id = ""
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = ""
        self.target_pose.pose.orientation.w = 1.0
        self.temp_target_pose = PoseStamped()
        self.temp_target_pose.header.frame_id = ""
        self.temp_target_pose.pose.orientation.w = 1.0
        self.target_twist = TwistStamped()
        self.target_twist.header.frame_id = ""
        self.temp_target_twist = TwistStamped()
        self.temp_target_twist.header.frame_id = ""
        self.landing_pose = PoseStamped()
        self.landing_pose.header.frame_id = ""
        self.landing_pose.pose.orientation.w = 1.0
        self.local_pose = None
        self.current_accel = Vector3Stamped()
        self.target_accel = Vector3Stamped()
        self.temp_target_accel = Vector3Stamped()
    
    def _wait_for_connection(self, timeout=5):
        rospy.loginfo(f"[{self.uav_name}] 等待飞控连接...")
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.current_state.connected:
                rospy.loginfo(f"[{self.uav_name}] 飞控连接成功")
                return True
            self.rate.sleep()

    def _wait_for_initial_pose(self, timeout=10):
        rospy.loginfo(f"[{self.uav_name}] 等待初始位姿数据...")
        start = rospy.Time.now()
        while not self.has_initial_pose and not rospy.is_shutdown():
            self.rate.sleep()
        rospy.loginfo(
            f"[{self.uav_name}] 获取到初始位置: "
            f"x={self.initial_pose.pose.position.x:.2f}, "
            f"y={self.initial_pose.pose.position.y:.2f}, "
            f"z={self.initial_pose.pose.position.z:.2f}"
        )   
        return True
    
    def _wait_for_ekf_ready(self, timeout=10):
        rospy.loginfo(f"[{self.uav_name}] 等待EKF状态准备就绪...")
        rospy.sleep(5)
        start_time = rospy.Time.now()
        while self.local_pose is None:
            if rospy.Time.now() - start_time > rospy.Duration(timeout):
                rospy.logerr(f"[{self.uav_name}] 等待 EKF 超时，未收到 local_pose！")
                raise TimeoutError("EKF 未就绪")
            rospy.logwarn(f"[{self.uav_name}] 尚未接收到 /mavros/local_position/pose...")
            self.rate.sleep()
        while not rospy.is_shutdown():
            current_pos = self.current_pose.pose.position
            local_pos = self.local_pose.pose.position
            dx = current_pos.x - local_pos.x
            dy = current_pos.y - local_pos.y
            if np.sqrt(dx**2 + dy**2) < 0.05:
                rospy.loginfo(f"[{self.uav_name}] EKF 状态准备就绪，位置差: "
                              f"dx={dx:.2f}, dy={dy:.2f}")
                break
            else:
                rospy.logwarn(f"[{self.uav_name}] EKF 状态不正常，位置差: "
                              f"dx={dx:.2f}, dy={dy:.2f}")
            self.rate.sleep()

    def _publisher_loop(self):
        while not rospy.is_shutdown() and self.keep_publishing:
            if self.agent.cmd_mode == "pose":
                self.cmd_pose_pub.publish(self.target_pose)
            if self.agent.cmd_mode == "twist":
                self.cmd_twist_pub.publish(self.target_twist)
            if self.agent.cmd_mode == "accel":
                self.cmd_accel_pub.publish(self.target_accel)
            self.pub_rate.sleep()

    def stop_publisher_thread(self):
        self.keep_publishing = False
        if self.publisher_thread:
            self.publisher_thread.join()
            rospy.loginfo(f"[{self.uav_name}] 后台命令发布线程已停止")
            self.publisher_thread = None

    def takeoff_and_start_task(self, target_z=1.0, timeout=15):
        pose = PoseStamped()
        pose.header.frame_id = ""
        pose.pose.position.x = self.current_pose.pose.position.x
        pose.pose.position.y = self.current_pose.pose.position.y
        pose.pose.position.z = self.current_pose.pose.position.z
        self.set_target_cmd(pose)
        self.start_publisher_thread()
        preheat_start = rospy.Time.now()
        preheat_duration = rospy.Duration(5)
        while not rospy.is_shutdown() and (rospy.Time.now() - preheat_start < preheat_duration):
            self.set_target_cmd(pose)
            self.rate.sleep()
        offb_mode = SetModeRequest()
        offb_mode.custom_mode = "OFFBOARD"
        requested_logged = False
        resp = self.set_mode_client.call(offb_mode)
        while not rospy.is_shutdown() and self.current_state.mode != "OFFBOARD":
            resp = self.set_mode_client.call(offb_mode)
            if resp.mode_sent and not requested_logged:
                rospy.loginfo(f"[{self.uav_name}] 请求切换为 OFFBOARD 模式")
                requested_logged = True
            if self.current_state.mode == "OFFBOARD":
                rospy.loginfo(f"[{self.uav_name}] 成功切换为 OFFBOARD 模式")
                break
            self.rate.sleep()
        arm_cmd = CommandBoolRequest()
        arm_cmd.value = True
        requested_logged = False
        resp = self.arming_client.call(arm_cmd)
        while not rospy.is_shutdown() and not self.current_state.armed:
            resp = self.arming_client.call(arm_cmd)
            if resp.success and not requested_logged:
                rospy.loginfo(f"[{self.uav_name}] 请求解锁电机")
                requested_logged = True
            if self.current_state.armed:
                rospy.loginfo(f"[{self.uav_name}] 成功解锁电机")
                break
            self.rate.sleep()
        rospy.sleep(2.0)
        takeoff_vel = 0.3
        t_step = 0.1
        last_time = rospy.Time.now()
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if now - last_time > rospy.Duration(t_step):
                current_z = self.current_pose.pose.position.z
                if current_z > 0.90 * target_z:
                    rospy.loginfo(f"[{self.uav_name}] 已达到目标高度: {current_z:.2f}m")
                    break
                next_z = min(current_z +  takeoff_vel, target_z)
                pose.pose.position.z = next_z
                self.set_target_cmd(pose)
                last_time = now
                self.rate.sleep()

    def hover_for(self, duration_sec=3):
        rospy.loginfo(f"[{self.uav_name}] 悬停 {duration_sec}秒")
        pose = PoseStamped()
        pose.header.frame_id = ""
        pose.pose.position.x = self.current_pose.pose.position.x
        pose.pose.position.y = self.current_pose.pose.position.y
        pose.pose.position.z = 1.0
        start = rospy.Time.now()
        while (rospy.Time.now() - start < rospy.Duration(duration_sec)) and not rospy.is_shutdown():
            self.set_target_cmd(pose)
            self.rate.sleep()
        rospy.loginfo(f"[{self.uav_name}] 悬停完成")

    def smooth_manual_land_and_disarm(self, descent_speed=0.3, min_z=0.02, timeout=20):
        self.is_landing = True
        self.landing_pose = deepcopy(self.current_pose)
        rospy.loginfo(f"[{self.uav_name}] 手动平滑降落开始")
        rospy.loginfo(f"[{self.uav_name}] 降落前位置: {self.landing_pose.pose.position.z:.2f}m")
        pose = deepcopy(self.landing_pose)
        start_time = rospy.Time.now()
        last_time = rospy.Time.now()
        land_vel = descent_speed
        t_step = 0.1
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if now - last_time > rospy.Duration(t_step):
                current_z = self.current_pose.pose.position.z
                if current_z <= min_z or (rospy.Time.now() - start_time > rospy.Duration(5)):
                    rospy.loginfo(f"[{self.uav_name}] 已达到最小高度: {current_z:.2f}m，停止降落")
                    break
                next_z = max(current_z - land_vel, min_z)
                pose.pose.position.z = next_z
                self._set_cmd_direct(pose)
                last_time = now
                self.rate.sleep()
        set_mode_cmd = SetModeRequest()
        set_mode_cmd.custom_mode = "POSCTL"
        requested_logged = False
        while not rospy.is_shutdown() and self.current_state.mode != "POSCTL":
            resp = self.set_mode_client.call(set_mode_cmd)
            if resp.mode_sent and not requested_logged:
                rospy.loginfo(f"[{self.uav_name}] 请求切换为 POSCTL 模式")
                requested_logged = True
            if self.current_state.mode == "POSCTL":
                rospy.loginfo(f"[{self.uav_name}] 已切换至 POSCTL 模式")
                break
            self.rate.sleep()
        rospy.sleep(2.0)
        rospy.loginfo(f"[{self.uav_name}] 平滑降落完成，开始锁定电机")
        disarm_cmd = CommandBoolRequest()
        disarm_cmd.value = False
        requested_logged = False
        while not rospy.is_shutdown() and self.current_state.armed:
            resp = self.arming_client.call(disarm_cmd)
            if resp.success and not requested_logged:
                rospy.loginfo("请求锁定电机")
                requested_logged = True
            if not self.current_state.armed:
                rospy.loginfo("成功锁定电机")
            self.rate.sleep()
        self.stop_publisher_thread()


class UAVFleetManager:
    def __init__(self, agents):
        self.controllers = [agent.controller for agent in agents]
        self.num_uavs = len(self.controllers)

    def _sync_execute(self, task_fn, event_name, *args, **kwargs):
        threads = []
        def task_wrapper(controller):
            rospy.loginfo(f"[{controller.uav_name}] 开始执行 {event_name}")
            getattr(controller, task_fn)(*args, **kwargs)
            rospy.loginfo(f"[{controller.uav_name}] 完成 {event_name}")
        for controller in self.controllers:
            t = Thread(target=task_wrapper, args=(controller,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    def sync_init_all(self):
        self._sync_execute("init", "init")

    def sync_takeoff_all(self, target_z=1.0):
        self._sync_execute("takeoff_and_start_task", "takeoff", target_z)

    def sync_run_task_all(self):
        self._sync_execute("run_task", "task")

    def sync_hover_all(self, duration=3):
        self._sync_execute("hover_for", "hover", duration)
        
    def sync_land_all(self):
        self._sync_execute("smooth_manual_land_and_disarm", "land")
