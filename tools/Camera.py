import numpy as np
import cv2 as cv
import time
from PIL import Image
from ctypes import *
from tools.MvImport.MvCameraControl_class import *
from tools.MvImport.CameraParams_header import MVCC_ENUMVALUE

class _CameraCommon:

    @staticmethod
    def to_hex_str(num):
        chaDic = {10: 'a', 11: 'b', 12: 'c', 13: 'd', 14: 'e', 15: 'f'}
        hexStr = ""
        if num < 0:
            num = num + 2**32
        while num >= 16:
            digit = num % 16
            hexStr = chaDic.get(digit, str(digit)) + hexStr
            num //= 16
        hexStr = chaDic.get(num, str(num)) + hexStr
        return hexStr
    
    @staticmethod
    def frame_to_opencv_image(frame_out_info, src_buf):
        width = int(frame_out_info.stFrameInfo.nWidth)
        height = int(frame_out_info.stFrameInfo.nHeight)
        pixel_type = int(frame_out_info.stFrameInfo.enPixelType)
        frame_len = int(frame_out_info.stFrameInfo.nFrameLen)

        np_buffer = np.ctypeslib.as_array(src_buf, shape=(frame_len,)).copy()

        if pixel_type == PixelType_Gvsp_Mono8:
            return np_buffer.reshape(height, width)

        if pixel_type == PixelType_Gvsp_RGB8_Packed:
            rgb_img = np_buffer.reshape(height, width, 3)
            return cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)

        if pixel_type == PixelType_Gvsp_BGR8_Packed:
            return np_buffer.reshape(height, width, 3)

        if pixel_type in (
            PixelType_Gvsp_BayerRG8,
            PixelType_Gvsp_BayerGB8,
            PixelType_Gvsp_BayerGR8,
            PixelType_Gvsp_BayerBG8,
        ):
            bayer_img = np_buffer.reshape(height, width)
            bayer_code_map = {
                PixelType_Gvsp_BayerRG8: cv.COLOR_BAYER_RG2BGR,
                PixelType_Gvsp_BayerGB8: cv.COLOR_BAYER_GB2BGR,
                PixelType_Gvsp_BayerGR8: cv.COLOR_BAYER_GR2BGR,
                PixelType_Gvsp_BayerBG8: cv.COLOR_BAYER_BG2BGR,
            }
            return cv.cvtColor(bayer_img, bayer_code_map[pixel_type])

        return None



class CameraController(_CameraCommon):

    def __init__(self,obj_cam,netIp,deviceIp,b_open_device=False,b_start_grabbing = False,h_thread_handle=None,\
                b_thread_closed=False,st_frame_info=None,b_exit=False,b_save_bmp=False,b_save_jpg=False,buf_save_image=None,\
                n_save_image_size=0,n_win_gui_id=0,frame_rate=0,exposure_time=0,gain=0):

        self.obj_cam:MvCamera= obj_cam
        self.b_open_device = b_open_device
        self.netIp = netIp
        self.deviceIp = deviceIp
        self.b_start_grabbing = b_start_grabbing 
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.h_thread_handle = h_thread_handle
        self.n_win_gui_id = n_win_gui_id
        self.n_save_image_size = n_save_image_size
        self.b_thread_closed
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain
        # 飞拍优化：缓存缓冲区，避免重复分配
        self._buf_cache = None
        self._img_buff = None
        self._cached_frame_size = 0

    def To_hex_str(self,num):
        return self.to_hex_str(num)

    def Open_device(self):
        if False == self.b_open_device:
            stDevInfo = MV_CC_DEVICE_INFO()
            stGigEDev = MV_GIGE_DEVICE_INFO()
            deviceIpList = self.deviceIp.split('.')
            stGigEDev.nCurrentIp = (int(deviceIpList[0]) << 24) | (int(deviceIpList[1]) << 16) | (int(deviceIpList[2]) << 8) | int(deviceIpList[3])
            netIpList = self.netIp.split('.')
            stGigEDev.nNetExport =  (int(netIpList[0]) << 24) | (int(netIpList[1]) << 16) | (int(netIpList[2]) << 8) | int(netIpList[3])
            stDevInfo.nTLayerType = MV_GIGE_DEVICE
            stDevInfo.SpecialInfo.stGigEInfo = stGigEDev
            self.cam = MvCamera()
            g_bExit = False
            ret = self.obj_cam.MV_CC_CreateHandle(stDevInfo)
            if ret != 0:
                self.obj_cam.MV_CC_DestroyHandle()
                print('show error','create handle fail! ret = '+ self.To_hex_str(ret))
                return ret

            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                print('show error','open device fail! ret = '+ self.To_hex_str(ret))
                return ret
            print ("open device successfully!")
            self.b_open_device = True
            self.b_thread_closed = False

            # ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
            if stDevInfo.nTLayerType == MV_GIGE_DEVICE:
                nPacketSize = self.obj_cam.MV_CC_GetOptimalPacketSize()
                if int(nPacketSize) > 0:
                    ret = self.obj_cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
                    if ret != 0:
                        print ("warning: set packet size fail! ret[0x%x]" % ret)
                else:
                    print ("warning: set packet size fail! ret[0x%x]" % nPacketSize)

            stBool = c_bool(False)
            ret =self.obj_cam.MV_CC_GetBoolValue("AcquisitionFrameRateEnable", stBool)
            if ret != 0:
                print ("get acquisition frame rate enable fail! ret[0x%x]" % ret)

            # ch:设置触发模式为off | en:Set trigger mode as off
            ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            if ret != 0:
                print ("set trigger mode fail! ret[0x%x]" % ret)
            return 0

    def Start_grabbing(self):
        if False == self.b_start_grabbing and True == self.b_open_device:
            self.b_exit = False
            ret = self.obj_cam.MV_CC_StartGrabbing()
            if ret != 0:
                print('show error','start grabbing fail! ret = '+ self.To_hex_str(ret))
                return
            self.b_start_grabbing = True
            print ("start grabbing successfully!")

    def Stop_grabbing(self):
        if True == self.b_start_grabbing and self.b_open_device == True:
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                print('show error','stop grabbing fail! ret = '+self.To_hex_str(ret))
                return
            print ("stop grabbing successfully!")
            self.b_start_grabbing = False
            self.b_exit  = True      

    def Close_device(self):
        if True == self.b_open_device:
            ret = self.obj_cam.MV_CC_CloseDevice()
            if ret != 0:
                print('show error','close deivce fail! ret = '+self.To_hex_str(ret))
                return
                
        # ch:销毁句柄 | Destroy handle
        self.obj_cam.MV_CC_DestroyHandle()
        self.b_open_device = False
        self.b_start_grabbing = False
        self.b_exit  = True
        print ("close device successfully!")

    def Set_trigger_mode(self,strMode):
        if True == self.b_open_device:
            if "continuous" == strMode: 
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode",0)
                if ret != 0:
                    print('show error','set triggermode fail! ret = '+self.To_hex_str(ret))
            if "triggermode" == strMode:
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerMode",1)
                if ret != 0:
                    print('show error','set triggermode fail! ret = '+self.To_hex_str(ret))
                ret = self.obj_cam.MV_CC_SetEnumValue("TriggerSource",7)
                if ret != 0:
                    print('show error','set triggersource fail! ret = '+self.To_hex_str(ret))

    def Trigger_once(self,nCommand):
        if True == self.b_open_device:
            if 1 == nCommand: 
                ret = self.obj_cam.MV_CC_SetCommandValue("TriggerSoftware")
                if ret != 0:
                    print('show error','set triggersoftware fail! ret = '+self.To_hex_str(ret))

    def Get_parameter(self,parameter_type = None):
        st_float_param = MVCC_FLOATVALUE()
        memset(byref(st_float_param), 0, sizeof(MVCC_FLOATVALUE))

        if parameter_type is None or not self.b_open_device:
            return None  # 直接返回 None 表示失败

        ret = self.obj_cam.MV_CC_GetFloatValue(parameter_type, st_float_param)
        if ret != 0:
            print(f"获取 {parameter_type} 失败，ret = {ret}")
            return None
        return st_float_param.fCurValue
      
    def Set_parameter(self ,parameter_type = None,value = None):
        ret = 0
        if parameter_type is None or value is None:
            return 1, "Please Enter a Parameter Type"
        if self.b_open_device == False:
            return 1, "Please Open Device First"
        else:
            if parameter_type == 'AcquisitionFrameRate':
                ret = self.obj_cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(value))
                if ret != 0:
                    return ret,"set parameter fail! ret = "+self.To_hex_str(ret)
            if parameter_type == 'ExposureTime':
                ret = self.obj_cam.MV_CC_SetFloatValue("ExposureTime",float(value))
                if ret != 0:
                    return ret,"set exposure time fail! ret = "+self.To_hex_str(ret)
            if parameter_type == 'Gain':
                ret = self.obj_cam.MV_CC_SetFloatValue("Gain",float(value))
                if ret != 0:
                    return ret,"set gain fail! ret = "+self.To_hex_str(ret)
            if parameter_type == "Gamma":
                ret = self.obj_cam.MV_CC_SetFloatValue("Gamma",float(value))
                if ret != 0:
                    return ret,"set gamma fail! ret = "+self.To_hex_str(ret)
 
        return ret,"Success"
    
    def Get_current_user_set(self):
        """获取当前使用的用户集
        
        Returns:
            (ret, user_set_index): ret=0表示成功，非0表示失败；user_set_index为用户集索引(0/1/2)，失败时返回0
        """
        if not self.b_open_device:
            return 1, 0
        
        try:
            st_enum_value = MVCC_ENUMVALUE()
            memset(byref(st_enum_value), 0, sizeof(MVCC_ENUMVALUE))
            
            # 尝试获取UserSetSelector（当前选择的用户集）
            ret = self.obj_cam.MV_CC_GetEnumValue("UserSetSelector", st_enum_value)
            if ret == 0:
                return 0, st_enum_value.nCurValue
            else:
                # 如果获取失败，尝试获取UserSetDefault（默认用户集）
                ret = self.obj_cam.MV_CC_GetEnumValue("UserSetDefault", st_enum_value)
                if ret == 0:
                    return 0, st_enum_value.nCurValue
                else:
                    # 如果都失败，返回默认值0
                    print(f'获取用户集失败，使用默认UserSet0，ret = {self.To_hex_str(ret)}')
                    return 0, 0
        except Exception as e:
            print(f'获取用户集异常：{str(e)}，使用默认UserSet0')
            return 0, 0
    
    def Save_to_user_set(self, user_set_index=None):
        """保存当前参数到相机用户集
        
        Args:
            user_set_index: 用户集索引 (0=UserSet0, 1=UserSet1, 2=UserSet2)，如果为None则使用当前用户集
        
        Returns:
            (ret, msg): ret=0表示成功，非0表示失败；msg为消息描述
        """
        if not self.b_open_device:
            return 1, "设备未打开"
        
        try:
            # 如果没有指定用户集，则获取当前使用的用户集
            if user_set_index is None:
                ret_get, user_set_index = self.Get_current_user_set()
                if ret_get != 0:
                    user_set_index = 0  # 默认使用UserSet0
            
            # 步骤1: 选择要保存到的用户集
            ret = self.obj_cam.MV_CC_SetEnumValue("UserSetSelector", user_set_index)
            if ret != 0:
                return ret, f"设置UserSetSelector失败，ret = {self.To_hex_str(ret)}"
            
            # 步骤2: 执行保存命令
            ret = self.obj_cam.MV_CC_SetCommandValue("UserSetSave")
            if ret != 0:
                return ret, f"执行UserSetSave失败，ret = {self.To_hex_str(ret)}"
            
            return 0, f"参数已成功保存到UserSet{user_set_index}"
        except Exception as e:
            return 1, f"保存到用户集失败：{str(e)}"
        
    def Get_image(self):
        """获取图像（快速模式，优化采图速度）
        
        Returns:
            numpy.ndarray: 图像数组，失败返回None
        """
        stOutFrame = MV_FRAME_OUT()
        retry_counter = 0
        max_retries = 2  # 快速模式：减少重试次数
        timeout_ms = 100  # 快速模式：减少超时时间
        
        while True:
            ret = self.obj_cam.MV_CC_GetImageBuffer(stOutFrame, timeout_ms)
            if 0 == ret:
                # 获取帧信息
                self.st_frame_info = stOutFrame.stFrameInfo
                frame_len = self.st_frame_info.nFrameLen
                
                # 缓存缓冲区：只在大小变化时重新分配
                if self._buf_cache is None or self._cached_frame_size != frame_len:
                    self._buf_cache = (c_ubyte * frame_len)()
                    self._cached_frame_size = frame_len
                
                buf_cache = self._buf_cache
                
                # 复制图像数据
                cdll.msvcrt.memcpy(byref(buf_cache), stOutFrame.pBufAddr, frame_len)
                
                # 跳过同步保存操作（应在异步线程中处理）
                # 注意：如需保存图像，请在异步线程中调用Save_jpg或Save_Bmp
                
                # 像素格式处理
                pixel_type = self.st_frame_info.enPixelType
                width = self.st_frame_info.nWidth
                height = self.st_frame_info.nHeight
                
                # 直接处理常见格式，避免不必要的转换
                if PixelType_Gvsp_RGB8_Packed == pixel_type:
                    # RGB8直接转换
                    numArray = CameraController.Color_numpy(self, buf_cache, width, height)
                elif PixelType_Gvsp_Mono8 == pixel_type:
                    # Mono8直接转换
                    numArray = CameraController.Mono_numpy(self, buf_cache, width, height)
                else:
                    # 需要格式转换
                    # 预分配转换缓冲区（仅在需要时）
                    if self.Is_color_data(pixel_type):
                        nConvertSize = width * height * 3
                        if self._img_buff is None or len(self._img_buff) < nConvertSize:
                            self._img_buff = (c_ubyte * nConvertSize)()
                        
                        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
                        memset(byref(stConvertParam), 0, sizeof(stConvertParam))
                        stConvertParam.nWidth = width
                        stConvertParam.nHeight = height
                        stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
                        stConvertParam.nSrcDataLen = frame_len
                        stConvertParam.enSrcPixelType = pixel_type
                        stConvertParam.enDstPixelType = PixelType_Gvsp_RGB8_Packed
                        stConvertParam.pDstBuffer = self._img_buff
                        stConvertParam.nDstBufferSize = nConvertSize
                        
                        ret = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
                        if ret != 0:
                            self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
                            retry_counter += 1
                            if retry_counter >= max_retries:
                                return None
                            continue
                        
                        numArray = CameraController.Color_numpy(self, self._img_buff, width, height)
                    elif self.Is_mono_data(pixel_type):
                        nConvertSize = width * height
                        if self._img_buff is None or len(self._img_buff) < nConvertSize:
                            self._img_buff = (c_ubyte * nConvertSize)()
                        
                        stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
                        memset(byref(stConvertParam), 0, sizeof(stConvertParam))
                        stConvertParam.nWidth = width
                        stConvertParam.nHeight = height
                        stConvertParam.pSrcData = cast(buf_cache, POINTER(c_ubyte))
                        stConvertParam.nSrcDataLen = frame_len
                        stConvertParam.enSrcPixelType = pixel_type
                        stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
                        stConvertParam.pDstBuffer = self._img_buff
                        stConvertParam.nDstBufferSize = nConvertSize
                        
                        ret = self.obj_cam.MV_CC_ConvertPixelType(stConvertParam)
                        if ret != 0:
                            self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
                            retry_counter += 1
                            if retry_counter >= max_retries:
                                return None
                            continue
                        
                        numArray = CameraController.Mono_numpy(self, self._img_buff, width, height)
                    else:
                        # 未知格式
                        self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
                        return None
                
                # 释放缓冲区
                self.obj_cam.MV_CC_FreeImageBuffer(stOutFrame)
                
                # 直接返回numpy数组
                return numArray
            else:
                retry_counter += 1
                if retry_counter >= max_retries:
                    return None
                continue

    def Save_jpg(self,buf_cache):
        if(None == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".jpg"
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Jpeg;                                        # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType                               # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth      = self.st_frame_info.nWidth                                    # ch:相机对应的宽 | en:Width
        stParam.nHeight     = self.st_frame_info.nHeight                                   # ch:相机对应的高 | en:Height
        stParam.nDataLen    = self.st_frame_info.nFrameLen
        stParam.pData       = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer=  cast(byref(self.buf_save_image), POINTER(c_ubyte)) 
        stParam.nBufferSize = self.n_save_image_size                                 # ch:存储节点的大小 | en:Buffer node size
        stParam.nJpgQuality = 80;                                                    # ch:jpg编码，仅在保存Jpg图像时有效。保存BMP时SDK内忽略该参数
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)            

        if return_code != 0:
            print('show error','save jpg fail! ret = '+self.To_hex_str(return_code))
            self.b_save_jpg = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_jpg = False
            print('show info','save jpg success!')
        except Exception as e:
            self.b_save_jpg = False
            raise Exception("get one frame failed:%s" % str(e))
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image

    def Save_Bmp(self,buf_cache):
        if(0 == buf_cache):
            return
        self.buf_save_image = None
        file_path = str(self.st_frame_info.nFrameNum) + ".bmp"    
        self.n_save_image_size = self.st_frame_info.nWidth * self.st_frame_info.nHeight * 3 + 2048
        if self.buf_save_image is None:
            self.buf_save_image = (c_ubyte * self.n_save_image_size)()

        stParam = MV_SAVE_IMAGE_PARAM_EX()
        stParam.enImageType = MV_Image_Bmp;                                        # ch:需要保存的图像类型 | en:Image format to save
        stParam.enPixelType = self.st_frame_info.enPixelType                               # ch:相机对应的像素格式 | en:Camera pixel type
        stParam.nWidth      = self.st_frame_info.nWidth                                    # ch:相机对应的宽 | en:Width
        stParam.nHeight     = self.st_frame_info.nHeight                                   # ch:相机对应的高 | en:Height
        stParam.nDataLen    = self.st_frame_info.nFrameLen
        stParam.pData       = cast(buf_cache, POINTER(c_ubyte))
        stParam.pImageBuffer=  cast(byref(self.buf_save_image), POINTER(c_ubyte)) 
        stParam.nBufferSize = self.n_save_image_size                                 # ch:存储节点的大小 | en:Buffer node size
        return_code = self.obj_cam.MV_CC_SaveImageEx2(stParam)            
        if return_code != 0:
            print('show error','save bmp fail! ret = '+self.To_hex_str(return_code))
            self.b_save_bmp = False
            return
        file_open = open(file_path.encode('ascii'), 'wb+')
        img_buff = (c_ubyte * stParam.nImageLen)()
        try:
            cdll.msvcrt.memcpy(byref(img_buff), stParam.pImageBuffer, stParam.nImageLen)
            file_open.write(img_buff)
            self.b_save_bmp = False
            print('show info','save bmp success!')
        except Exception as e:
            self.b_save_bmp = False
            raise Exception("get one frame failed:%s" % str(e))
        if None != img_buff:
            del img_buff
        if None != self.buf_save_image:
            del self.buf_save_image



class LineScanCamera(_CameraCommon):
    def __init__(self,obj_cam,netIp,deviceIp,device_index=0,b_open_device=False,b_start_grabbing = False,h_thread_handle=None,\
                b_thread_closed=False,st_frame_info=None,b_exit=False,b_save_bmp=False,b_save_jpg=False,buf_save_image=None,\
                n_save_image_size=0,n_win_gui_id=0,frame_rate=0,exposure_time=0,gain=0):

        self.obj_cam:MvCamera= obj_cam
        self.b_open_device = b_open_device
        self.device_index = device_index
        self.netIp = netIp
        self.deviceIp = deviceIp
        self.b_start_grabbing = b_start_grabbing 
        self.b_thread_closed = b_thread_closed
        self.st_frame_info = st_frame_info
        self.b_exit = b_exit
        self.b_save_bmp = b_save_bmp
        self.b_save_jpg = b_save_jpg
        self.buf_save_image = buf_save_image
        self.h_thread_handle = h_thread_handle
        self.n_win_gui_id = n_win_gui_id
        self.n_save_image_size = n_save_image_size
        self.b_thread_closed
        self.frame_rate = frame_rate
        self.exposure_time = exposure_time
        self.gain = gain
        # 飞拍优化：缓存缓冲区，避免重复分配
        self._buf_cache = None
        self._img_buff = None
        self._cached_frame_size = 0
        MvCamera.MV_CC_Initialize()

    def Open_device(self):
        if False == self.b_open_device:
            device_list = MV_CC_DEVICE_INFO_LIST()
            t_layer_type = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                            | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)
            ret = MvCamera.MV_CC_EnumDevices(t_layer_type, device_list)
            if ret != 0:
         
                return ret,"enum devices fail! ret[0x%x]" % ret
            if device_list.nDeviceNum == 0:
                return ret,"find no device!"
            self.mvcc_dev_info = cast(device_list.pDeviceInfo[self.device_index], POINTER(MV_CC_DEVICE_INFO)).contents
            self.obj_cam = MvCamera()
            ret = self.obj_cam.MV_CC_CreateHandle(self.mvcc_dev_info)
            if ret != 0:
                return ret,"create handle fail! ret[0x%x]" % ret
            ret = self.obj_cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                return ret,"open device fail! ret[0x%x]" % ret
            print("open device successfully!")
            self.b_open_device = True
            self.b_thread_closed = False
            return 0,"Success"
    def Close_device(self):
        if False == self.b_open_device:
            return 1,"device not open!"
        ret = self.obj_cam.MV_CC_CloseDevice()
        if ret != 0:
            return ret,"close device fail! ret[0x%x]" % ret
        self.b_open_device = False
        self.b_thread_closed = True
        return 0,"Success"

    def Start_grabbing(self):
        if False == self.b_open_device:
            return 1,"device not open!"
        ret = self.obj_cam.MV_CC_StartGrabbing()
        if ret != 0:
            return ret,"start grabbing fail! ret[0x%x]" % ret
        self.b_start_grabbing = True
        print("start grabbing successfully!")
        return 0,"Success"

    def Stop_grabbing(self):
        if True == self.b_start_grabbing and self.b_open_device == True:
            ret = self.obj_cam.MV_CC_StopGrabbing()
            if ret != 0:
                return ret,"stop grabbing fail! ret[0x%x]" % ret                       

    def Get_image(self):
        self.st_frame_info = MV_FRAME_OUT()
        memset(byref(self.st_frame_info), 0, sizeof(self.st_frame_info))
        ret =self.obj_cam.MV_CC_GetImageBuffer(self.st_frame_info, 1000)
        if ret != 0:
            return ret,"get image buffer fail! ret[0x%x]" % ret
        
        opencv_image =self.frame_to_opencv_image(self.st_frame_info, self.st_frame_info.pBufAddr)
        self.obj_cam.MV_CC_FreeImageBuffer(self.st_frame_info)
        return opencv_image

    def Set_parameter(self,parameter_name,parameter_type,value):
        if parameter_type == "EnumValue":
            ret = self.obj_cam.MV_CC_SetEnumValue(parameter_name,int(value))
            if ret != 0:
                return ret,"set enum value fail! ret[0x%x]" % ret
        elif parameter_type == "FloatValue":
            ret = self.obj_cam.MV_CC_SetFloatValue(parameter_name,float(value))
            if ret != 0:
                return ret,"set float value fail! ret[0x%x]" % ret
        elif parameter_type == "IntValue":
            ret = self.obj_cam.MV_CC_SetIntValueEx(parameter_name,int(value))
            if ret != 0:
                return ret,"set int value fail! ret[0x%x]" % ret
        elif parameter_type == "BoolValue":
            ret = self.obj_cam.MV_CC_SetBoolValue(parameter_name,c_bool(value))
            if ret != 0:
                return ret,"set bool value fail! ret[0x%x]" % ret
        elif parameter_type == "StringValue":
            ret = self.obj_cam.MV_CC_SetStringValue(parameter_name,value)
            if ret != 0:
                return ret,"set string value fail! ret[0x%x]" % ret
        return 0,"Success"