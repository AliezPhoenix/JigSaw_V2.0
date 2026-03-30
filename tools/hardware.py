import numpy as np
import threading
import time
from ctypes import *
from tools.MvImport.MvCameraControl_class import MvCamera
from tools.MvImport.CameraParams_const import *
from tools.MvImport.MvErrorDefine_const import *
from tools.Camera import CameraController
import cv2 as cv
import modbus_tk.modbus_tcp as modbus_tcp
import modbus_tk.defines as cst


class Hardware_Manager:
    """
    硬件管理类
    
    管理多个相机实例，支持通过IP地址连接、拍照、参数设置等功能
    线程安全：不同线程可以同时操作不同的相机实例
    
    设计说明：
    - 直接存储CameraController实例，不维护额外的状态信息
    - 在初始化时创建CameraController（包含MvCamera实例）
    - 使用CameraController的b_open_device和b_start_grabbing属性判断状态
    """

    def __init__(self, hardware_id_list: list = None):
        """
        初始化硬件管理器
        
        Args:
            hardware_id_list: 硬件ID列表，格式为 [{"alias": "CAM_1", "port_ip": "192.168.1.10", "device_ip": "192.168.1.100"}, ...]
                            如果为None，则创建空的管理器
        """
        # 使用字典存储CameraController实例，key为别名
        self.hardware_dict: dict[str, CameraController] = {}
        # 线程锁，保护hardware_dict的访问（多线程环境下保护字典操作）
        self._lock = threading.Lock()
        
        # 如果提供了硬件ID列表，则初始化创建CameraController
        if hardware_id_list:
            for hw_info in hardware_id_list:
                alias = hw_info.get("alias")
                port_ip = hw_info.get("port_ip")
                device_ip = hw_info.get("device_ip")
                if alias and port_ip and device_ip:
                    self.add_camera(alias, port_ip, device_ip)
    
    def add_camera(self, alias: str, port_ip: str, device_ip: str) -> tuple:
        """
        添加相机实例（创建CameraController）
        
        Args:
            alias: 相机别名
            port_ip: 主机网口IP
            device_ip: 相机IP
            
        Returns:
            (成功标志, 消息, 结果)，成功时返回 (True, "相机添加成功", None)，失败时返回 (False, 错误消息, None)
        """
        with self._lock:
            # 检查别名是否已存在
            if alias in self.hardware_dict:
                return False, "别名已存在", None
            
            try:
                # 创建MvCamera实例
                obj_cam = MvCamera()
                
                # 创建CameraController实例
                camera_controller = CameraController(
                    obj_cam=obj_cam,
                    netIp=port_ip,
                    deviceIp=device_ip
                )
                
                self.hardware_dict[alias] = camera_controller
                return True, "相机添加成功", None
            except Exception as e:
                return False, f"创建相机失败: {str(e)}", None
     
    def connect(self, alias: str) -> tuple:
        """
        连接指定相机
        
        Args:
            alias: 相机别名
            
        Returns:
            (成功标志, 消息, 结果)，成功时返回 (True, "相机连接成功", None)，失败时返回 (False, 错误消息, None)
        """
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            
            camera_controller = self.hardware_dict[alias]
        
        # 在锁外执行连接操作，因为连接可能耗时较长
        # 如果相机已连接，先关闭
        if camera_controller.b_open_device:
            success, msg, _ = self.close_camera(alias)
            if not success:
                return False, f"关闭现有连接失败: {msg}", None
        
        try:
            # 打开设备（使用CameraController的Open_device方法）
            nRet = camera_controller.Open_device()
            if nRet != MV_OK:
                error_msg = f"开启相机失败，错误码: 0x{nRet:08x}"
                if nRet == 0x80000206:  # MV_E_NETER
                    error_msg += " (网络错误，请检查port_ip和device_ip配置及网络连接)"
                elif nRet == 0x80000221:  # MV_E_IP_CONFLICT
                    error_msg += " (IP冲突)"
                elif nRet == 0x80000203:  # MV_E_ACCESS_DENIED
                    error_msg += " (设备无访问权限)"
                elif nRet == 0x80000204:  # MV_E_BUSY
                    error_msg += " (设备忙或网络断开)"
                return False, error_msg, None
            
            # 开始取流
            camera_controller.Start_grabbing()
            
            return True, "相机连接成功", None
        except Exception as e:
            return False, f"连接异常: {str(e)}", None
    
    def disconnect(self, alias: str) -> tuple:
        """
        断开指定相机连接（关闭相机）
        
        Args:
            alias: 相机别名
            
        Returns:
            (成功标志, 消息, 结果)，成功时返回 (True, "相机断开成功", None)，失败时返回 (False, 错误消息, None)
        """
        return self.close_camera(alias)
    
    def close_camera(self, alias: str) -> tuple:
        """
        关闭指定相机（停止取流并关闭设备连接）
        
        Args:
            alias: 相机别名
            
        Returns:
            (成功标志, 消息, 结果)，成功时返回 (True, "相机关闭成功", None)，失败时返回 (False, 错误消息, None)
        """
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            
            camera_controller = self.hardware_dict[alias]
        
        # 在锁外执行关闭操作
        try:
            # 停止取流
            if camera_controller.b_start_grabbing:
                camera_controller.Stop_grabbing()
            
            # 关闭设备（使用CameraController的Close_device方法）
            if camera_controller.b_open_device:
                camera_controller.Close_device()
            
            return True, "相机关闭成功", None
        except Exception as e:
            return False, f"关闭异常: {str(e)}", None
    
    def open_camera(self, alias: str) -> tuple:
        """
        打开指定相机（打开设备连接并开始取流）
        
        Args:
            alias: 相机别名
            
        Returns:
            (成功标志, 消息, 结果)，成功时返回 (True, "相机打开成功", None)，失败时返回 (False, 错误消息, None)
        """
        return self.connect(alias)
    
    def restart_camera(self, alias: str) -> tuple:
        """
        重启指定相机（先关闭再打开）
        
        Args:
            alias: 相机别名
            
        Returns:
            (成功标志, 消息, 结果)，成功时返回 (True, "相机重启成功", None)，失败时返回 (False, 错误消息, None)
        """
        # 先关闭
        success, msg, _ = self.close_camera(alias)
        if not success:
            return False, f"关闭失败: {msg}", None
        
        # 再打开
        return self.connect(alias)
    
    def capture_image(self, alias: str, timeout: int = 1000) -> tuple:
        """
        同步拍照取图（阻塞直到获取图像）
        
        Args:
            alias: 相机别名
            timeout: 超时时间（毫秒），默认1000ms（注意：CameraController的Get_image使用固定超时）
            
        Returns:
            (成功标志, 消息, 图像数据)，成功时返回 (True, "图像获取成功", 图像数据)
            失败时返回 (False, 错误消息, None)
            图像数据为numpy数组（OpenCV格式，BGR）
        """
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            
            camera_controller = self.hardware_dict[alias]
        
        # 检查相机是否已连接并正在取流
        if not camera_controller.b_open_device or not camera_controller.b_start_grabbing:
            return False, "相机未连接或未开始取流", None
        
        # 使用CameraController的Get_image方法获取图像
        try:
            image = camera_controller.Get_image()
            if image is None:
                return False, "获取图像数据为空", None
            
            # CameraController的Color_numpy返回的是RGB格式（R,G,B通道顺序）
            # 需要转换为BGR（OpenCV格式）
            # 检查图像维度
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB格式，转换为BGR（交换R和B通道）
                image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            
            return True, "图像获取成功", image
            
        except Exception as e:
            return False, f"获取图像异常: {str(e)}", None
    
    def get_parameter(self, alias: str, param_name: str) -> tuple:
        """
        获取指定相机的指定参数
        
        Args:
            alias: 相机别名
            param_name: 参数名称，如 "ExposureTime", "Gain", "Gamma", "AcquisitionFrameRate"
            
        Returns:
            (成功标志, 消息, 参数值)，成功时返回 (True, "参数获取成功", 参数值)
            失败时返回 (False, 错误消息, None)
        """
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            
            camera_controller = self.hardware_dict[alias]
        
        # 检查相机是否已连接
        if not camera_controller.b_open_device:
            return False, "相机未连接", None
        
        try:
            # 使用CameraController的Get_parameter方法
            value = camera_controller.Get_parameter(param_name)
            if value is None:
                return False, f"参数 {param_name} 获取失败或不存在", None
            
            return True, "参数获取成功", value
                
        except Exception as e:
            return False, f"获取参数异常: {str(e)}", None
    
    def set_parameter(self, alias: str, param_name: str, value: float) -> tuple:
        """
        设置指定相机的指定参数
        
        Args:
            alias: 相机别名
            param_name: 参数名称，如 "ExposureTime", "Gain", "Gamma", "AcquisitionFrameRate"
            value: 参数值
            
        Returns:
            (成功标志, 消息, 结果)，成功时返回 (True, "参数设置成功", None)，失败时返回 (False, 错误消息, None)
        """
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            
            camera_controller = self.hardware_dict[alias]
        
        # 检查相机是否已连接
        if not camera_controller.b_open_device:
            return False, "相机未连接", None
        
        try:
            # 使用CameraController的Set_parameter方法
            ret, msg = camera_controller.Set_parameter(param_name, value)
            if ret != MV_OK:
                return False, f"参数设置失败: {msg}", None
            
            return True, "参数设置成功", None
                
        except Exception as e:
            return False, f"设置参数异常: {str(e)}", None
    
    def get_all_cameras(self) -> tuple:
        """
        获取所有相机别名列表
        
        Returns:
            (成功标志, 消息, 相机别名列表)，成功时返回 (True, "获取成功", 相机别名列表)
        """
        with self._lock:
            camera_list = list(self.hardware_dict.keys())
            return True, "获取成功", camera_list
    
    def is_connected(self, alias: str) -> tuple:
        """
        检查指定相机是否已连接
        
        Args:
            alias: 相机别名
            
        Returns:
            (成功标志, 消息, 连接状态)，成功时返回 (True, "检查成功", True/False)
            失败时返回 (False, 错误消息, None)
        """
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            camera_controller = self.hardware_dict[alias]
            is_conn = camera_controller.b_open_device
            return True, "检查成功", is_conn
    
    def is_grabbing(self, alias: str) -> tuple:
        """
        检查指定相机是否正在取流
        
        Args:
            alias: 相机别名
            
        Returns:
            (成功标志, 消息, 取流状态)，成功时返回 (True, "检查成功", True/False)
            失败时返回 (False, 错误消息, None)
        """
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            camera_controller = self.hardware_dict[alias]
            is_grab = camera_controller.b_start_grabbing
            return True, "检查成功", is_grab
    
    def save_to_userSet(self, alias: str, user_set_index: int = None) -> tuple:
        """
        保存当前相机参数到当前用户集合
        
        Args:
            alias: 相机别名
            user_set_index: 用户集索引 (0=UserSet0, 1=UserSet1, 2=UserSet2)，如果为None则使用当前用户集
            
        Returns:
            (成功标志, 消息, 结果)，成功时返回 (True, "保存成功", None)，失败时返回 (False, 错误消息, None)
        """
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            
            camera_controller = self.hardware_dict[alias]
        
        # 检查相机是否已连接
        if not camera_controller.b_open_device:
            return False, "相机未连接", None
        
        try:
            # 调用CameraController的Save_to_user_set方法
            ret, msg = camera_controller.Save_to_user_set(user_set_index)
            if ret != 0:
                return False, f"保存用户集失败: {msg}", None
            
            return True, "保存用户集成功", None
            
        except Exception as e:
            return False, f"保存用户集异常: {str(e)}", None


class ModBus_Manager:
    
    def __init__(self,modbus_info:list):
        self.modbus_dict = {}
        self.MAX_REGISTERS = 123
        for modbus_info in modbus_info:
            alias = modbus_info['alias']
            host_ip = modbus_info['host_ip']
            port = modbus_info['port']
            self.modbus_dict[alias] = modbus_tcp.TcpMaster(host_ip, port)

    def connect(self, alias: str) -> int:
        """
        连接 Modbus 连接
        
        Args:
            alias: Modbus 连接别名
        
        Returns:
            (成功标志, 错误信息)，成功时返回 (True, None)，失败时返回 (False, 错误信息)
        """
        if alias not in self.modbus_dict:
            return False ,"modbus not found"
        try:
            master : modbus_tcp.TcpMaster = self.modbus_dict[alias]
            master.open()
            return True ,"modbus connected"
        except Exception as e:
            return False ,e

    def close(self, alias: str) -> int:
        """
        关闭 Modbus 连接
        
        Args:
            alias: Modbus 连接别名
        
        Returns:
            (成功标志, 错误信息)，成功时返回 (True, None)，失败时返回 (False, 错误信息)
        """
        if alias not in self.modbus_dict:
            return False ,"modbus not found"
        try:
            master : modbus_tcp.TcpMaster = self.modbus_dict[alias]
            master.close()
            return True ,"modbus closed"
        except Exception as e:
            return False ,e

    def read(self,alias: str, address: int, count: int, function_code: int) -> tuple:
        """
        读取 Modbus 数据
        
        Args:
            alias: Modbus 连接别名
            address: 起始地址
            count: 读取数量
            function_code: 功能码，支持以下类型：
                - cst.READ_COILS: 读线圈状态（读取单个线圈时 count=1）
                - cst.READ_DISCRETE_INPUTS: 读离散输入状态
                - cst.READ_HOLDING_REGISTERS: 读保持寄存器
                - cst.READ_INPUT_REGISTERS: 读输入寄存器
        
        Returns:
            (成功标志, 响应数据或错误信息)，成功时返回 (True, 响应数据)，失败时返回 (False, 错误信息)
        """
        if alias not in self.modbus_dict:
            return False, "modbus not found"
        
        master: modbus_tcp.TcpMaster = self.modbus_dict[alias]
        
        # 支持的功能码列表
        supported_codes = [
            cst.READ_COILS,
            cst.READ_DISCRETE_INPUTS,
            cst.READ_HOLDING_REGISTERS,
            cst.READ_INPUT_REGISTERS
        ]
        
        # 检查是否支持该功能码
        if function_code not in supported_codes:
            return False, f"Unsupported function code: {function_code}"
        
        try:
            response = master.execute(1, function_code, address, count)
            return True, response
        except Exception as e:
            return False, e


    def write(self,alias: str, address: int, value_list: list, function_code: int) -> tuple:
        """
        写入 Modbus 数据
        
        Args:
            alias: Modbus 连接别名
            address: 起始地址
            value_list: 要写入的值列表
            function_code: 功能码，支持 cst.WRITE_SINGLE_COIL 或 cst.WRITE_MULTIPLE_REGISTERS
            
        Returns:
            (成功标志, 错误信息)，成功时返回 (True, None)，失败时返回 (False, 错误信息)
        """
        if alias not in self.modbus_dict:
            return False, "modbus not found"

        master: modbus_tcp.TcpMaster = self.modbus_dict[alias]
        
        # 根据功能码选择不同的写入方式
        if function_code == cst.WRITE_SINGLE_COIL:
            # 写入单个线圈，value_list 应该只包含一个值
            if len(value_list) != 1:
                return False, "WRITE_SINGLE_COIL requires exactly one value"
            
            try:
                master.execute(1, function_code, address, len(value_list),value_list[0])
                return True, None
            except Exception as e:
                return False, e
                
        elif function_code == cst.WRITE_SINGLE_REGISTER:
            if len(value_list) != 1:
                return False, "WRITE_SINGLE_REGISTER requires exactly one value"
            try:
                master.execute(1, function_code, address, output_value=value_list[0])
                return True, None
            except Exception as e:
                return False, e
        elif function_code == cst.WRITE_MULTIPLE_REGISTERS:
            # 写入多个寄存器，最大单次传输长度为123
            if len(value_list) > self.MAX_REGISTERS:
                num_batches = (len(value_list) + self.MAX_REGISTERS - 1) // self.MAX_REGISTERS
                for batch_index in range(num_batches):
                    batch_start = batch_index * self.MAX_REGISTERS
                    batch_end = min(batch_start + self.MAX_REGISTERS, len(value_list))
                    batch_value_list = value_list[batch_start:batch_end]
                    current_address = address + batch_start
                    try:
                        master.execute(
                            1,
                            function_code,
                            current_address,
                            len(batch_value_list),
                            batch_value_list,
                        )
                    except Exception as e:
                        return False, e
                    # 与旧工程 WorkThread：每批成功后 sleep（含最后一批，再交由上层写完成线圈）
                    time.sleep(0.01)
                return True, None
            else:
                # 单次写入即可
                try:
                    master.execute(1, function_code, address, len(value_list), value_list)
                    return True, None
                except Exception as e:
                    return False, e
        else:
            return False, f"Unsupported function code: {function_code}"


            