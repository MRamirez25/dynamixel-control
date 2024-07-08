import numpy as np
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, DXL_LOWORD, DXL_HIWORD, DXL_LOBYTE, DXL_HIBYTE, GroupSyncRead
from .config import Config

class Robot:

    def __init__(self, config: Config, ids: list) -> None:
        super().__init__()

        self.config = config
        self.ids = ids
        self.port_handler = PortHandler(config.DEVICENAME)
        self.packet_handler = PacketHandler(config.PROTOCOL_VERSION)
        self.initial_positions = {}

    def start(self, op_mode, current_limit=None):
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
            self.packet_handler.write2ByteTxRx(self.port_handler, id, self.config.ADDR_CURRENT_LIMIT, int(current_limit))
            self.group_sync_write_pos = GroupSyncWrite(self.port_handler, self.packet_handler, self.config.ADDR_GOAL_POSITION, self.config.LEN_GOAL_POSITION)
            self.group_sync_write_current = GroupSyncWrite(self.port_handler, self.packet_handler, self.config.ADDR_GOAL_CURRENT, self.config.LEN_GOAL_CURRENT)
            self.group_sync_read_pos = GroupSyncRead(self.port_handler, self.packet_handler, self.config.ADDR_PRESENT_POSITION, self.config.LEN_PRESENT_POSITION)
            self.group_sync_read_current = GroupSyncRead(self.port_handler, self.packet_handler, self.config.ADDR_PRESENT_CURRENT, self.config.LEN_PRESENT_CURRENT)

            self.group_sync_read_pos.addParam(id)
            self.group_sync_read_current.addParam(id)

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

    def move_pos_sync(self, positions, relative_to_init=True):
        for id in self.ids:
            if relative_to_init:
                goal = int(self.initial_positions[id] + positions[id])
            else:
                goal = int(positions[id])
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(goal)), DXL_HIBYTE(DXL_LOWORD(goal)), DXL_LOBYTE(DXL_HIWORD(goal)), DXL_HIBYTE(DXL_HIWORD(goal))]
            self.group_sync_write_pos.addParam(id, param_goal_position)
        self.group_sync_write_pos.txPacket()
        self.group_sync_write_pos.clearParam()
    
    def get_positions_sync(self):
        positions = dict.fromkeys(self.ids)
        self.group_sync_read_pos.txRxPacket()
        for id in self.ids:
            positions[id] = self.group_sync_read_pos.getData(id, self.config.ADDR_PRESENT_POSITION, self.config.LEN_PRESENT_POSITION)
        return positions
    
    def move_current_sync(self, currents):
        for id in self.ids:
            goal = int(currents)
            param_goal_current = [DXL_LOBYTE(goal), DXL_HIBYTE(goal)]
            self.group_sync_write_current.addParam(id, param_goal_current)
        self.group_sync_write_current.txPacket()
        self.group_sync_write_current.clearParam()
    
    def get_currents_sync(self):
        currents = dict.fromkeys(self.ids)
        self.group_sync_read_current.txRxPacket()
        for id in self.ids:
            currents[id] = self.group_sync_read_current.getData(id, self.config.ADDR_PRESENT_CURRENT, self.config.LEN_PRESENT_CURRENT)