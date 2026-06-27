import numpy as np
import threading
import time
from ctypes import *
from typing import Union
from tools.MvImport.MvCameraControl_class import MvCamera
from tools.Camera import CameraController, LineScanCamera
import modbus_tk.modbus_tcp as modbus_tcp
import modbus_tk.defines as cst


CAMERA_TYPE_AREA = "Area"
CAMERA_TYPE_LINE = "Line"

CameraInstance = Union[CameraController, LineScanCamera]


class Hardware_Manager:
    """
    硬件管理类

    管理多个相机实例，支持面阵（Area）与线扫（Line）两种类型。
    线程安全：不同线程可以同时操作不同的相机实例。
    """

    def __init__(self, hardware_id_list: list = None):
        """
        初始化硬件管理器

        Args:
            hardware_id_list: 硬件配置列表，每项示例：
                面阵: {"alias": "CAM_1", "camera_type": "Area", "port_ip": "...", "device_ip": "..."}
                线扫: {"alias": "CAM_1", "camera_type": "Line", "device_index": 0}
                camera_type 缺省为 Area；Line 的 device_index 缺省为 0
        """
        self.hardware_dict: dict[str, CameraInstance] = {}
        self._camera_types: dict[str, str] = {}
        self._lock = threading.Lock()

        if hardware_id_list:
            for hw_info in hardware_id_list:
                alias = hw_info.get("alias")
                if alias:
                    self.add_camera(hw_info)

    @staticmethod
    def _normalize_camera_type(hw_info: dict) -> str:
        camera_type = hw_info.get("camera_type", CAMERA_TYPE_AREA)
        if camera_type not in (CAMERA_TYPE_AREA, CAMERA_TYPE_LINE):
            raise ValueError(f"不支持的 camera_type: {camera_type}")
        return camera_type

    def _create_camera(self, hw_info: dict) -> CameraInstance:
        camera_type = self._normalize_camera_type(hw_info)
        if camera_type == CAMERA_TYPE_LINE:
            return LineScanCamera(
                obj_cam=MvCamera(),
                netIp=hw_info.get("port_ip", ""),
                deviceIp=hw_info.get("device_ip", ""),
                device_index=hw_info.get("device_index", 0),
            )
        return CameraController(
            obj_cam=MvCamera(),
            netIp=hw_info["port_ip"],
            deviceIp=hw_info["device_ip"],
        )

    def _camera_type(self, alias: str) -> str:
        return self._camera_types.get(alias, CAMERA_TYPE_AREA)

    def _open_device(self, camera: CameraInstance) -> tuple:
        ret, msg = camera.Open_device()
        if ret != 0:
            return False, msg or f"开启相机失败，错误码: 0x{ret:08x}", ret
        return True, None, ret

    def _start_grabbing(self, camera: CameraInstance, camera_type: str) -> tuple:
        # 线扫相机由调用方按需 Start/Stop，连接阶段无需持续取流
        if camera_type == CAMERA_TYPE_LINE:
            return True, None
        ret, msg = camera.Start_grabbing()
        if ret != 0:
            return False, msg
        return True, None

    def _stop_grabbing(self, camera: CameraInstance) -> None:
        if not camera.b_start_grabbing:
            return
        ret, msg = camera.Stop_grabbing()
        if ret != 0:
            print(f"停止取流失败: {msg}")

    def _close_device(self, camera: CameraInstance) -> None:
        if not camera.b_open_device:
            return
        ret, msg = camera.Close_device()
        if ret != 0:
            print(f"关闭设备失败: {msg}")

    def _get_image(self, camera: CameraInstance):
        return camera.Get_image()

    def add_camera(self, hw_info: dict) -> tuple:
        """
        添加相机实例

        Args:
            hw_info: 相机配置字典，需包含 alias；面阵需 port_ip/device_ip，线扫需 device_index（可选，默认0）

        Returns:
            (成功标志, 消息, 结果)
        """
        alias = hw_info.get("alias")
        if not alias:
            return False, "缺少 alias", None

        try:
            camera_type = self._normalize_camera_type(hw_info)
        except ValueError as e:
            return False, str(e), None

        if camera_type == CAMERA_TYPE_AREA:
            if not hw_info.get("port_ip") or not hw_info.get("device_ip"):
                return False, "面阵相机需要 port_ip 和 device_ip", None

        with self._lock:
            if alias in self.hardware_dict:
                return False, "别名已存在", None

            try:
                camera = self._create_camera(hw_info)
                self.hardware_dict[alias] = camera
                self._camera_types[alias] = camera_type
                return True, "相机添加成功", None
            except Exception as e:
                return False, f"创建相机失败: {str(e)}", None

    def connect(self, alias: str) -> tuple:
        """
        连接指定相机

        Args:
            alias: 相机别名

        Returns:
            (成功标志, 消息, 结果)
        """
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            camera = self.hardware_dict[alias]
            camera_type = self._camera_type(alias)

        if camera.b_open_device:
            success, msg, _ = self.close_camera(alias)
            if not success:
                return False, f"关闭现有连接失败: {msg}", None

        try:
            success, error_msg, _ = self._open_device(camera)
            if not success:
                return False, error_msg, None

            success, error_msg = self._start_grabbing(camera, camera_type)
            if not success:
                self._close_device(camera)
                return False, error_msg, None

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
            camera = self.hardware_dict[alias]

        try:
            self._stop_grabbing(camera)
            self._close_device(camera)
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
            camera = self.hardware_dict[alias]

        try:
            image = self._get_image(camera)
            if image is None:
                return False, "获取图像数据为空", None
            return True, "图像获取成功", image
        except Exception as e:
            return False, f"获取图像异常: {str(e)}", None

    def start_grabbing(self, alias: str) -> tuple:
        """开始指定相机取流。"""
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            camera = self.hardware_dict[alias]

        if not camera.b_open_device:
            return False, "相机未连接", None

        try:
            ret, msg = camera.Start_grabbing()
            if ret != 0:
                return False, msg, None
            return True, "开始取流成功", None
        except Exception as e:
            return False, f"开始取流异常: {str(e)}", None

    def stop_grabbing(self, alias: str) -> tuple:
        """停止指定相机取流。"""
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            camera = self.hardware_dict[alias]

        try:
            ret, msg = camera.Stop_grabbing()
            if ret != 0:
                return False, msg, None
            return True, "停止取流成功", None
        except Exception as e:
            return False, f"停止取流异常: {str(e)}", None

    def get_parameter(self, alias: str, param_name: str, parameter_type: str = None) -> tuple:
        """
        获取指定相机的指定参数
        
        Args:
            alias: 相机别名
            param_name: 参数名称，如 "ExposureTime", "Gain", "Gamma", "AcquisitionFrameRate"
            parameter_type: 参数类型，如 "FloatValue", "IntValue", "BoolValue", "StringValue", "EnumValue"
        Returns:
            (成功标志, 消息, 参数值)，成功时返回 (True, "参数获取成功", 参数值)
            失败时返回 (False, 错误消息, None)
        """
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            camera = self.hardware_dict[alias]

        if not camera.b_open_device:
            return False, "相机未连接", None

        try:
            value = camera.Get_parameter(param_name, parameter_type)
            if value is None:
                return False, f"参数 {param_name} 获取失败或不存在", None
            return True, "参数获取成功", value
        except Exception as e:
            return False, f"获取参数异常: {str(e)}", None
    
    def set_parameter(self, alias: str, param_name: str, value, parameter_type: str = None) -> tuple:
        """
        设置指定相机的指定参数

        Args:
            alias: 相机别名
            param_name: 参数名称，如 "ExposureTime", "Gain", "Gamma", "AcquisitionFrameRate"
            value: 参数值
            parameter_type: 参数类型，如 "FloatValue", "IntValue", "BoolValue", "StringValue", "EnumValue"

        Returns:
            (成功标志, 消息, 结果)，成功时返回 (True, "参数设置成功", None)，失败时返回 (False, 错误消息, None)
        """
        with self._lock:
            if alias not in self.hardware_dict:
                return False, "相机不存在", None
            camera = self.hardware_dict[alias]

        if not camera.b_open_device:
            return False, "相机未连接", None

        try:
            if parameter_type is not None:
                ret, msg = camera.Set_parameter(param_name, parameter_type, value)
            else:
                ret, msg = camera.Set_parameter(param_name, value)
            if ret != 0:
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
            camera = self.hardware_dict[alias]
            is_conn = camera.b_open_device
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
            camera = self.hardware_dict[alias]
            is_grab = camera.b_start_grabbing
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
            camera = self.hardware_dict[alias]

        if not camera.b_open_device:
            return False, "相机未连接", None

        try:
            ret, msg = camera.Save_to_user_set(user_set_index)
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


            