# RovVisionApi 类：一个用于水下机器人的视觉控制API
# 功能：将高级别的、归一化的控制指令（如 forward=0.5, yaw=-0.2）
#       转换为给每个推进器和舵机的具体PWM或角度值。
class RovVisionApi:
    def __init__(self, gain_settings=None, invert_settings=None):
        # 定义了API接受的9个控制指令的名称和顺序
        self._command_keys = [
            'forward',  # 前进/后退
            'heave',  # 上浮/下潜
            'yaw',  # 偏航（左转/右转）
            'body_pitch',  # 俯仰
            'roll',  # 翻滚
            'arm_extend',  # 机械臂伸缩
            'arm_pitch',  # 机械臂俯仰
            'gripper',  # 夹爪开合
            'grab_action'  # 抓取动作（一个逻辑信号）
        ]
        # 设置默认的运动增益（灵敏度）
        default_gains = {
            'FORWARD_GAIN': 1.0, 'HEAVE_GAIN': 1.0, 'YAW_GAIN': 1.0,
            'BODY_PITCH_GAIN': 1.0, 'ROLL_GAIN': 1.0
        }
        # 如果用户提供了自定义增益，则更新默认值
        if gain_settings:
            default_gains.update(gain_settings)
        self.gains = default_gains

        # 设置默认的电机反向标志
        default_inverts = {
            'pitch_mid': False, 'vert_left': False, 'vert_right': False,
            'yaw_left': False, 'yaw_right': False, 'arm_extend': False,
            'arm_pitch': False, 'gripper': False
        }
        # 如果用户提供了自定义反向设置，则更新默认值
        if invert_settings:
            default_inverts.update(invert_settings)
        self.inverts = invert_settings

        # 初始化时打印配置信息
        print("ROV 视觉 API 已初始化 (9指令模式)。")
        print(f"  指令顺序: {self._command_keys}")
        print(f"  增益设置: {self.gains}")
        print(f"  反向设置: {self.inverts}")

    # 主函数：接收指令元组，返回一个包含各电机输出值的字典
    def get_outputs(self, command_tuple):
        # 检查输入是否为包含9个元素的元组或列表
        if not isinstance(command_tuple, (list, tuple)) or len(command_tuple) != 9:
            raise ValueError("输入必须是包含9个元素的元组或列表。")

        # 将元组和键名打包成一个易于访问的字典
        commands = dict(zip(self._command_keys, command_tuple))

        # 1. 计算运动推进器的速度（PWM值）
        motion_pwms = self._calculate_motion_speeds(
            commands['forward'], commands['heave'], commands['yaw'],
            commands['body_pitch'], commands['roll']
        )
        # 2. 计算机械臂直流电机的速度（PWM值）
        arm_dc_pwms = {
            'arm_extend': int(commands['arm_extend'] * 255),
            'arm_pitch': int(commands['arm_pitch'] * 255)
        }
        # 合并所有直流电机的逻辑PWM值
        logical_dc_pwms = {**motion_pwms, **arm_dc_pwms}

        # 3. 应用电机反向设置
        final_dc_outputs = {}
        for name, pwm in logical_dc_pwms.items():
            if self.inverts.get(name):
                final_dc_outputs[name] = -pwm
            else:
                final_dc_outputs[name] = pwm

        # 4. 计算夹爪舵机的角度
        gripper_command = commands['gripper']  # 输入范围 [-1.0, 1.0]
        angle = (gripper_command + 1.0) * 90  # 映射到 [0, 180] 度
        if self.inverts['gripper']:
            angle = 180 - angle  # 应用反向
        final_servo_output = {'gripper': int(angle)}

        # 5. 组合所有最终输出，并包含抓取动作信号
        final_outputs = {**final_dc_outputs, **final_servo_output}
        final_outputs['grab_action'] = commands.get('grab_action', 0)

        return final_outputs

    # 私有方法：根据高级运动指令计算每个推进器的速度（运动混合算法）
    def _calculate_motion_speeds(self, forward, heave, yaw, body_pitch, roll):
        # 应用增益
        forward_force = forward * self.gains['FORWARD_GAIN']
        heave_force = heave * self.gains['HEAVE_GAIN']
        yaw_force = yaw * self.gains['YAW_GAIN']
        pitch_force = body_pitch * self.gains['BODY_PITCH_GAIN']
        roll_force = roll * self.gains['ROLL_GAIN']

        # 初始化各推进器的速度
        raw_speeds = {'pitch_mid': 0.0, 'vert_left': 0.0, 'vert_right': 0.0, 'yaw_left': 0.0, 'yaw_right': 0.0}

        # --- 运动混合核心逻辑 ---
        # 这里的混合逻辑取决于ROV的物理布局和推进器安装方式
        # 这是一个示例性的混合算法：
        raw_speeds['pitch_mid'] += heave_force + pitch_force
        raw_speeds['vert_left'] += heave_force - pitch_force - roll_force
        raw_speeds['vert_right'] += heave_force - pitch_force + roll_force
        raw_speeds['yaw_left'] += forward_force - yaw_force
        raw_speeds['yaw_right'] += forward_force + yaw_force

        # 归一化处理：防止电机输出饱和
        # 找到所有电机速度中的最大绝对值
        max_abs_speed = max(abs(s) for s in raw_speeds.values()) if raw_speeds else 0
        # 如果最大速度超过1.0，则按比例缩放所有电机的速度
        if max_abs_speed > 1.0:
            for motor in raw_speeds:
                raw_speeds[motor] /= max_abs_speed

        # 将归一化的速度 [-1.0, 1.0] 转换为PWM值 [0, 255] 或 [-255, 255]
        return {motor: int(speed * 255) for motor, speed in raw_speeds.items()}