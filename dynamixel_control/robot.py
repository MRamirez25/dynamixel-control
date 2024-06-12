import numpy as np
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, DXL_LOWORD, DXL_HIWORD, DXL_LOBYTE, DXL_HIBYTE, GroupSyncRead
from config import Config

class Robot:

    def __init__(self, config: Config, ids: list) -> None:
        super().__init__()

        self.config = config
        self.ids = ids
        self.port_handler = PortHandler(config.DEVICENAME)
        self.packet_handler = PacketHandler(config.PROTOCOL_VERSION)
        self.initial_positions = {}

    def start(self, op_mode):
        if self.port_handler.openPort():
            print(f"Succeeded to open the port")

        if self.port_handler.setBaudRate(self.config.BAUDRATE):
            print(f"Succeeded to set baud rate")

        for i, id in enumerate(self.ids):
            # disable torque
            self.packet_handler.write1ByteTxRx(self.port_handler, id, self.config.ADDR_TORQUE_ENABLE,
                                               self.config.TORQUE_DISABLE)
            # update operating mode
            self.packet_handler.write1ByteTxRx(self.port_handler, id, self.config.ADDR_OPERATING_MODE,op_mode)
            self.group_sync_write_pos = GroupSyncWrite(self.port_handler, self.packet_handler, self.config.ADDR_GOAL_POSITION, 4)
            self.group_sync_read_pos = GroupSyncRead(self.port_handler, self.packet_handler, self.config.ADDR_PRESENT_POSITION, 4)

        for id in self.ids:
            self.packet_handler.write1ByteTxRx(self.port_handler, id, self.config.ADDR_TORQUE_ENABLE,
                                               self.config.TORQUE_ENABLE)
            present_position, _, _ = self.packet_handler \
                .read4ByteTxRx(self.port_handler, id, self.config.ADDR_PRESENT_POSITION)
            self.initial_positions[id] = present_position
            print(f"initial position {id}: {present_position}")

    def set_homing_offsets(self, homing_offsets: dict):
        for id in self.ids:
            self.packet_handler.write1ByteTxRx(self.port_handler, id, self.config.ADDR_TORQUE_ENABLE,
                                               self.config.TORQUE_DISABLE)
            self.packet_handler.write4ByteTxRx(self.port_handler, id, self.config.ADDR_HOMING_OFFSET, homing_offsets[id])
            self.packet_handler.write1ByteTxRx(self.port_handler, id, self.config.ADDR_TORQUE_ENABLE,
                                               self.config.TORQUE_ENABLE)

    def move_pos_sync(self, positions):
        for id in self.ids:
            goal = int(self.initial_positions[id]) + positions[id]
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(goal)), DXL_HIBYTE(DXL_LOWORD(goal)), DXL_LOBYTE(DXL_HIWORD(goal)), DXL_HIBYTE(DXL_HIWORD(goal))]
            self.group_sync_write_pos.addParam(id, param_goal_position)
        self.group_sync_write_pos.txPacket()
        self.group_sync_write_pos.clearParam()
    
    def get_positions_sync(self):
        positions = dict.fromkeys(self.ids)
        self.group_sync_read_pos.txRxPacket()
        for id in self.ids:
            positions[id] = self.group_sync_read_pos.getData(id, self.config.ADDR_PRESENT_POSITION, 4)
        return positions