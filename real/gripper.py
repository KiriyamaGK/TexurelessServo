import minimalmodbus
import threading

class Gripper:
    def __init__(self):
        self.TARGET_POSITION_REGISTER = 0x0101
        self.TARGET_SPEED_REGISTER = 0x0102
        self.TARGET_POWER_REGISTER = 0x0103
        self.TARGET_ENABLE_REGISTER = 0x0104
        self.POWER_REGISTER = 0x0401
        self.POSITION_ARRIVE_REGISTER = 0x0402
        self.CURRENT_POSITION_HIGH_REGISTER = 0x0501
        self.CURRENT_POSITION_LOW_REGISTER = 0x0502

        # 串口设置
        self.STOPBITS = 1
        self.PORT = '/dev/Gripper'
        self.BAUD = 115200
        self.instrument = minimalmodbus.Instrument(self.PORT, 1)
        self.instrument.serial.baudrate = self.BAUD
        self.instrument.serial.timeout = 1
        self.instrument.serial.stopbits = self.STOPBITS
        # 线程锁
        self.lock = threading.Lock()

    def read_power(self):
        with self.lock:
            return self.instrument.read_register(self.POWER_REGISTER)
    def read_current_low_position(self):
        with self.lock:
            return self.instrument.read_register(self.CURRENT_POSITION_LOW_REGISTER)

    def write_target_position(self,position):
        with self.lock:
            self.instrument.write_register(self.TARGET_POSITION_REGISTER, position)

    # 写入目标速度
    def write_target_speed(self, speed):
        with self.lock:
            self.instrument.write_register(self.TARGET_SPEED_REGISTER, speed)

    # 写入目标力矩
    def write_target_power(self, power):
        with self.lock:
            self.instrument.write_register(self.TARGET_POWER_REGISTER, power)

    # 写入目标使能
    def write_target_enable(self, enable):
        with self.lock:
            self.instrument.write_register(self.TARGET_ENABLE_REGISTER, enable)

    def move_gripper(self,position,speed,power):
        '''
        :param position: absolute position of the gripper:0 means fully open,3050 means fully closed
        :param speed: 0-100,in pencent
        :param power: 0-100,in pencent
        :return:
        '''
        self.write_target_position(position)
        self.write_target_speed(speed)
        self.write_target_power(power)
        self.write_target_enable(1)

if __name__=='__main__':
    gripper=Gripper()
    # a=gripper.read_current_low_position()
    # print(a)
    gripper.move_gripper(0,60,60)
    a = gripper.read_current_low_position()
    print(a)