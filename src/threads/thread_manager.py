from PyQt5.QtCore import QThread
from src.threads.dry_thread import DryThread
from src.threads.transfer_thread import TransferThread
from src.threads.suckerthread_1 import SuckerThread1
from src.threads.suckerthread_2 import SuckerThread2
from src.threads.fulltray_thread import FulltrayThread
from tools.hardware import Hardware_Manager, ModBus_Manager
from src.config.config_manager import ConfigManager

# 类型检查导入（避免循环导入）
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main_window import MainWindow


class ThreadManager:
    # 线程类映射
    THREAD_CLASS_MAP = {
        'dry_thread': DryThread,
        'transfer_thread': TransferThread,
        'sucker_thread_1': SuckerThread1,
        'sucker_thread_2': SuckerThread2,
        'fulltray_thread': FulltrayThread
    }

    # 参数 section 映射
    PARAMS_SECTION_MAP = {
        'dry_thread': 'work_dry_params',
        'transfer_thread': 'work_transfer_params',
        'sucker_thread_1': 'work_sucker1_params',
        'sucker_thread_2': 'work_sucker2_params',
        'fulltray_thread': 'work_fulltray_params'
    }

    def __init__(self, thread_list: list, hardware_manager:Hardware_Manager, modbus_manager:ModBus_Manager, config_manager:ConfigManager, ui:'MainWindow' =None):
        """
        初始化线程管理器
        
        Args:
            thread_list: 线程别名列表，例如 ['dry_thread', 'transfer_thread', ...]
            hardware_manager: 硬件管理器实例（所有线程共用）
            modbus_manager: ModBus 管理器实例（所有线程共用）
            config_manager: 配置管理器实例（所有线程共用，用于读取参数）
            ui: MainWindow 实例，用于传递给线程
        """
        self.threads: dict[str, QThread] = {}
        self.ui = ui
        self.hardware_manager = hardware_manager
        self.modbus_manager = modbus_manager
        self.config_manager = config_manager
        
        for alias in thread_list:
            if not alias:
                continue
            
            # 获取线程类和参数 section
            thread_class = self.THREAD_CLASS_MAP.get(alias)
            params_section = self.PARAMS_SECTION_MAP.get(alias)
            
            if not thread_class:
                print(f"警告: 未找到线程类映射，alias={alias}")
                continue
            
            # 通过 ConfigManager 读取参数
            try:
                if params_section:
                    params = self.config_manager.get_section(params_section)
                else:
                    params = {}
            except KeyError:
                # 如果参数 section 不存在，使用空字典
                print(f"警告: 参数 section '{params_section}' 不存在，使用空字典，alias={alias}")
                params = {}
            
            # 创建线程实例
            try:
                thread_instance = thread_class(
                    params=params, 
                    HM=self.hardware_manager, 
                    MM=self.modbus_manager, 
                    ui=self.ui
                )
                self.threads[alias] = thread_instance
            except Exception as e:
                print(f"错误: 创建线程失败，alias={alias}, 错误={str(e)}")

    def start_all_threads(self):
        """启动所有线程"""
        for alias, thread_instance in self.threads.items():
            if not thread_instance.isRunning():
                thread_instance.start()
                print(f"启动线程: {alias}")

    def stop_all_threads(self):
        """停止所有线程"""
        for alias, thread_instance in self.threads.items():
            if thread_instance.isRunning():
                thread_instance.requestInterruption()
                thread_instance.wait()
                print(f"停止线程: {alias}")

    def get_thread_obj(self, station):
        """
        获取指定站点的线程对象
        
        Args:
            station: 线程别名
            
        Returns:
            线程对象，如果不存在则返回 None
        """
        if station in self.threads.keys():
            return self.threads[station]
        return None

    def get_thread_status(self, station=None):
        """
        获取线程状态信息
        
        Args:
            station: 线程别名，如果为 None 则返回所有线程的状态
            
        Returns:
            如果 station 为 None，返回字典，key 为线程别名，value 为状态字典
            如果 station 指定，返回该线程的状态字典
            状态字典包含：
            - 'exists': 线程是否存在
            - 'is_running': 线程是否正在运行
            - 'is_paused': 线程是否暂停（如果线程有 is_paused 属性）
            - 'is_finished': 线程是否已完成
        """
        if station is None:
            # 返回所有线程的状态
            all_status = {}
            for alias, thread_instance in self.threads.items():
                all_status[alias] = self._get_single_thread_status(thread_instance)
            return all_status
        else:
            # 返回指定线程的状态
            if station not in self.threads:
                return {
                    'exists': False,
                    'is_running': False,
                    'is_paused': False,
                    'is_finished': False
                }
            return self._get_single_thread_status(self.threads[station])
    
    def _get_single_thread_status(self, thread_instance):
        """
        获取单个线程的详细状态
        
        Args:
            thread_instance: 线程实例
            
        Returns:
            状态字典，包含：
            - 'exists': 线程是否存在（始终为 True）
            - 'is_running': 线程是否正在运行
            - 'is_paused': 线程是否暂停
            - 'is_finished': 线程是否已完成
        """
        return {
            'exists': True,
            'is_running': thread_instance.isRunning(),
            'is_paused': thread_instance.is_paused,
            'is_finished': thread_instance.isFinished()
        }

    def pause_thread(self, station):
        """
        暂停指定线程
        
        Args:
            station: 线程别名
            
        Returns:
            bool: 成功返回 True，失败返回 False
        """
        if station not in self.threads:
            print(f"警告: 线程不存在，station={station}")
            return False
        
        thread_instance = self.threads[station]
        thread_instance.pause()
        print(f"暂停线程: {station}")
        return True
    
    def resume_thread(self, station):
        """
        恢复指定线程
        
        Args:
            station: 线程别名
            
        Returns:
            bool: 成功返回 True，失败返回 False
        """
        if station not in self.threads:
            print(f"警告: 线程不存在，station={station}")
            return False
        
        thread_instance = self.threads[station]
        thread_instance.resume()
        print(f"恢复线程: {station}")
        return True
    
    def restart_thread(self, station):
        """
        重启指定线程（停止 -> 重新读取参数 -> 更新参数 -> 启动）
        
        Args:
            station: 线程别名
            
        Returns:
            bool: 成功返回 True，失败返回 False
        """
        if station not in self.threads:
            print(f"警告: 线程不存在，station={station}")
            return False
        
        thread_instance = self.threads[station]
        was_running = thread_instance.isRunning()
        
        # 如果线程正在运行，先停止
        if was_running:
            thread_instance.quit()
            thread_instance.wait(3000)  # 等待最多3秒
            if thread_instance.isRunning():
                print(f"警告: 线程 {station} 停止超时，强制终止")
                thread_instance.terminate()
                thread_instance.wait(1000)
        
        # 重新读取配置参数
        params_section = self.PARAMS_SECTION_MAP.get(station)
        if not params_section:
            print(f"警告: 未找到参数 section 映射，station={station}")
            return False
        
        try:
            params = self.config_manager.get_section(params_section)
        except KeyError:
            print(f"警告: 参数 section '{params_section}' 不存在，station={station}")
            return False
        
        # 更新线程参数
        thread_instance.update_params(params)
        print(f"更新线程参数: {station}")
        
        # 如果之前正在运行，重新启动
        if was_running:
            if not thread_instance.isRunning():
                thread_instance.start()
                print(f"重启线程: {station}")
            else:
                print(f"警告: 线程 {station} 仍在运行，无法重启")
                return False
        
        return True
    
    def pause_all_threads(self):
        """暂停所有线程"""
        success_count = 0
        for alias in self.threads.keys():
            if self.pause_thread(alias):
                success_count += 1
        print(f"暂停线程完成: {success_count}/{len(self.threads)}")
        return success_count
    
    def resume_all_threads(self):
        """恢复所有线程"""
        success_count = 0
        for alias in self.threads.keys():
            if self.resume_thread(alias):
                success_count += 1
        print(f"恢复线程完成: {success_count}/{len(self.threads)}")
        return success_count
    
    def restart_all_threads(self):
        """重启所有线程"""
        success_count = 0
        for alias in self.threads.keys():
            if self.restart_thread(alias):
                success_count += 1
        print(f"重启线程完成: {success_count}/{len(self.threads)}")
        return success_count
    
    def update_params(self, station):
        """
        更新指定线程的参数（不重启线程）
        
        Args:
            station: 线程别名
            
        Returns:
            bool: 成功返回 True，失败返回 False
        """
        if station not in self.threads:
            print(f"警告: 线程不存在，station={station}")
            return False
        
        # 获取参数 section
        params_section = self.PARAMS_SECTION_MAP.get(station)
        if not params_section:
            print(f"警告: 未找到参数 section 映射，station={station}")
            return False
        
        # 从配置管理器读取参数
        try:
            params = self.config_manager.get_section(params_section)
        except KeyError:
            print(f"警告: 参数 section '{params_section}' 不存在，station={station}")
            return False
        
        # 更新线程参数
        thread_instance = self.threads[station]
        thread_instance.update_params(params)
        print(f"更新线程参数: {station}")
        return True

