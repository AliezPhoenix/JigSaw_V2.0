# 从共享导入文件导入所有需要的模块
from src.threads.thread_imports import *


class SuckerThread1(QThread):
    _update_image_signal = pyqtSignal(np.ndarray)
    _update_statistics_signal = pyqtSignal(dict)  # 统计更新信号
    _update_message_signal = pyqtSignal(str)

    # ———————————————————————————————初始化————————————————————————————————————————————————————————————————
    def __init__(self, params: dict, HM: Hardware_Manager, MM: ModBus_Manager, ui: 'MainWindow'):
        super().__init__()
        self.HM = HM    # 硬件管理器
        self.MM = MM    # Modbus管理器
        self.ui = ui
        self.ball_detector = BallDetector()  # 球检测器
        self.size_detector = SizeDetector()  # 尺寸检测器
        self.shift_detector = ShiftDetector()
        self.mark_detector = MarkDetector()  # 标记检测器
        self.template_detector = TemplateDetector()  # 模板检测器
        self.bga_strip = Bga_Strip(strip_side="", strip_lot="", strip_sn="", strip_create_time="", params=params)
        self.update_params(params)
        
        # 图像异步保存
        self.image_save_queue = Queue()
        self._init_image_save_thread()
        
        # 垃圾回收计数器
        self.gc_counter = 0
        self.gc_interval = 10  # 每10次循环执行一次垃圾回收
        
        # 暂停/恢复标志
        self.is_paused = False

    # ——————————————————————————————参数更新函数————————————————————————————————————————————————————————————————————
    def update_params(self, params: dict):
        pass

    def _init_image_save_thread(self):
        pass

    def run(self):
        pass
