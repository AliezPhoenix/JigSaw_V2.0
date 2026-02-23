import json
import os
import threading
from typing import Any, Optional


class ConfigManager:
    """
    线程安全的配置管理器
    
    支持保存、修改、读取section和key的功能
    使用JSON格式存储配置，采用嵌套字典的方式表示section和key
    """
    
    def __init__(self, config_file_path: str = "./config/default.json") -> None:
        """
        初始化配置管理器
        
        Args:
            config_file_path: 配置文件路径，默认为 "./config/default.json"
        """
        self.config_file_path = config_file_path
        self.config_dict: dict = {}
        self._lock = threading.Lock()  # 线程锁，保证线程安全
        
        # 如果配置文件存在，则加载；不存在则创建空配置
        if os.path.exists(config_file_path):
            self.load()
        else:
            # 确保目录存在
            os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
            self.config_dict = {}
    
    def load(self,file_path:str) -> None:
        """
        从文件加载配置
        
        Returns:
            bool: 是否成功加载配置
            str: 错误信息，如果失败则返回错误信息
        """
        with self._lock:
            try:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Config file not found: {file_path}")
                
                with open(file_path, "r", encoding="utf-8") as f:
                    self.config_dict = json.load(f)
            except json.JSONDecodeError as e:
                return False,f"Failed to parse JSON file: {file_path}. Error: {str(e)}"
            except IOError as e:
                return False,f"Failed to read config file: {file_path}. Error: {str(e)}"
            return True,None
    
    def save(self, file_path: str) -> None:
        """
        保存配置到文件
        
        直接更新整个JSON文件。如果文件不存在则创建新文件，存在则替换。
        
        Returns:
            bool: 是否成功保存配置
            str: 错误信息，如果失败则返回错误信息
        """
        with self._lock:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # 写入文件，使用缩进格式化JSON
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self.config_dict, f, indent=2, ensure_ascii=False)
            except IOError as e:
                return False,f"Failed to write config file: {file_path}. Error: {str(e)}"
            return True,None
    
    def get_section(self, section: str) -> dict:
        """
        获取整个section
        
        Args:
            section: section名称
            
        Returns:
            section对应的字典
            
        Raises:
            KeyError: section不存在时抛出异常
        """
        with self._lock:
            if section not in self.config_dict:
                raise KeyError(f"Section not found: {section}")
            return self.config_dict[section].copy()  # 返回副本，避免外部修改
    
    def get_key(self, section: str, key: str, default: Any = None) -> Any:
        """
        获取指定section下的key值
        
        Args:
            section: section名称
            key: key名称
            default: 如果key不存在时返回的默认值，默认为None
            
        Returns:
            key对应的值，如果不存在则返回default
        """
        with self._lock:
            try:
                return self.config_dict[section][key]
            except KeyError:
                if default is not None:
                    return default
                raise KeyError(f"Key '{key}' not found in section '{section}'")
    
    def set_section(self, section: str, value: dict) -> None:
        """
        设置指定section的值
        
        Args:
            section: section名称
            value: 要设置的值
        """
        with self._lock:
            if section not in self.config_dict:
                self.config_dict[section] = value
            self.config_dict[section].update(value)
    def set_key(self, section: str, key: str, value: Any) -> None:
        """
        设置指定section下的key值
        
        如果section或key不存在，则自动创建。
        
        Args:
            section: section名称
            key: key名称
            value: 要设置的值
        """
        with self._lock:
            # 如果section不存在，则创建
            if section not in self.config_dict:
                self.config_dict[section] = {}
            
            # 设置key值
            self.config_dict[section][key] = value
    
    def create_section(self, section: str) -> None:
        """
        创建新的section
        
        如果section已存在，则不进行任何操作。
        
        Args:
            section: section名称
        """
        with self._lock:
            if section not in self.config_dict:
                self.config_dict[section] = {}
    
    def delete_section(self, section: str) -> None:
        """
        删除指定的section
        
        Args:
            section: section名称
            
        Raises:
            KeyError: section不存在时抛出异常
        """
        with self._lock:
            if section not in self.config_dict:
                raise KeyError(f"Section not found: {section}")
            del self.config_dict[section]
    
    def delete_key(self, section: str, key: str) -> None:
        """
        删除指定section下的key
        
        Args:
            section: section名称
            key: key名称
            
        Raises:
            KeyError: section或key不存在时抛出异常
        """
        with self._lock:
            if section not in self.config_dict:
                raise KeyError(f"Section not found: {section}")
            if key not in self.config_dict[section]:
                raise KeyError(f"Key '{key}' not found in section '{section}'")
            del self.config_dict[section][key]
    
    def has_section(self, section: str) -> bool:
        """
        检查section是否存在
        
        Args:
            section: section名称
            
        Returns:
            如果section存在返回True，否则返回False
        """
        with self._lock:
            return section in self.config_dict
    
    def has_key(self, section: str, key: str) -> bool:
        """
        检查指定section下的key是否存在
        
        Args:
            section: section名称
            key: key名称
            
        Returns:
            如果key存在返回True，否则返回False
        """
        with self._lock:
            return section in self.config_dict and key in self.config_dict[section]
    
    def get_all_sections(self) -> list:
        """
        获取所有section名称列表
        
        Returns:
            section名称列表
        """
        with self._lock:
            return list(self.config_dict.keys())
    
    def get_all_keys(self, section: str) -> list:
        """
        获取指定section下的所有key名称列表
        
        Args:
            section: section名称
            
        Returns:
            key名称列表
            
        Raises:
            KeyError: section不存在时抛出异常
        """
        with self._lock:
            if section not in self.config_dict:
                raise KeyError(f"Section not found: {section}")
            return list(self.config_dict[section].keys())

