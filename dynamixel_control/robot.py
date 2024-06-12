import numpy as np
from dynamixel_sdk import PortHandler, PacketHandler, GroupSyncWrite, DXL_LOWORD, DXL_HIWORD, DXL_LOBYTE, DXL_HIBYTE, GroupSyncRead


class Robot:
    mask = np.array([1, -1])

    def __init__(self, config, tentacles) -> None:
        super().__init__()

        self.config = config
        self.tentacles = tentacles
        self.port_handler = PortHandler(config.DEVICENAME)
        self.packet_handler = PacketHandler(config.PROTOCOL_VERSION)
        self.initial_positions = {}

    def start(self, op_mode):
        if self.port_handler.openPort():
            print(f"Succeeded to open the port")

        if self.port_handler.setBaudRate(self.config.BAUDRATE):
            print(f"Succeeded to set baud rate")

        for i, tentacle in enumerate(self.tentacles):
            # disable torque
            self.packet_handler.write1ByteTxRx(self.port_handler, tentacle, self.config.ADDR_TORQUE_ENABLE,
                                               self.config.TORQUE_DISABLE)
            # update operating mode
            self.packet_handler.write1ByteTxRx(self.port_handler, tentacle, self.config.ADDR_OPERATING_MODE,op_mode)
            self.group_sync_write_pos = GroupSyncWrite(self.port_handler, self.packet_handler, self.config.ADDR_GOAL_POSITION, 4)
            self.group_sync_read_pos = GroupSyncRead(self.port_handler, self.packet_handler, self.config.ADDR_PRESENT_POSITION, 4)

            # self.packet_handler.write2ByteTxRx(self.port_handler, tentacle, self.config.ADDR_POSITION_D_GAIN,
            #                                    int(0))
            # self.packet_handler.write2ByteTxRx(self.port_handler, tentacle, self.config.ADDR_POSITION_P_GAIN,
            #                                    int(1000))
            # # enable torque

        for tentacle in self.tentacles:
            self.packet_handler.write1ByteTxRx(self.port_handler, tentacle, self.config.ADDR_TORQUE_ENABLE,
                                               self.config.TORQUE_ENABLE)
            present_position, _, _ = self.packet_handler \
                .read4ByteTxRx(self.port_handler, tentacle, self.config.ADDR_PRESENT_POSITION)
            self.initial_positions[tentacle] = present_position
            print(f"initial position {tentacle}: {present_position}")

    def set_homing_offsets(self, homing_offsets: dict):
        for tentacle in self.tentacles:
            self.packet_handler.write1ByteTxRx(self.port_handler, tentacle, self.config.ADDR_TORQUE_ENABLE,
                                               self.config.TORQUE_DISABLE)
            self.packet_handler.write4ByteTxRx(self.port_handler, tentacle, self.config.ADDR_HOMING_OFFSET, homing_offsets[tentacle])
            self.packet_handler.write1ByteTxRx(self.port_handler, tentacle, self.config.ADDR_TORQUE_ENABLE,
                                               self.config.TORQUE_ENABLE)
                                               
    
    def stop(self):
        for tentacle in self.tentacles:
            self.packet_handler \
                .write2ByteTxRx(self.port_handler, tentacle, self.config.ADDR_GOAL_CURRENT, 0)
        self.port_handler.closePort()

    def move_cur(self, current):
        positions = self.get_positions()
        for tentacle in self.tentacles:
            self.packet_handler \
                .write2ByteTxRx(self.port_handler, tentacle, self.config.ADDR_GOAL_CURRENT, current[tentacle])
            
    def move_vel(self, velocity):
        for tentacle in self.tentacles:
            self.packet_handler \
                .write2ByteTxRx(self.port_handler, tentacle, self.config.ADDR_GOAL_VELOCITY, velocity[tentacle])
            
    def move_pos(self, id, position):
        goal_position = int(self.initial_positions[id]) + position
        self.packet_handler \
            .write4ByteTxRx(self.port_handler, id, self.config.ADDR_GOAL_POSITION, goal_position)
    
    def move_pos_sync(self, positions):

        for tentacle in self.tentacles:
            goal = int(self.initial_positions[tentacle]) + positions[tentacle]
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(goal)), DXL_HIBYTE(DXL_LOWORD(goal)), DXL_LOBYTE(DXL_HIWORD(goal)), DXL_HIBYTE(DXL_HIWORD(goal))]
            self.group_sync_write_pos.addParam(tentacle, param_goal_position)
        self.group_sync_write_pos.txPacket()
        self.group_sync_write_pos.clearParam()

    def move_pos_cur(self, motion_params):
        for tentacle in self.tentacles:
            present_position, _, _ = self.packet_handler \
                .read4ByteTxRx(self.port_handler, tentacle, self.config.ADDR_PRESENT_POSITION)
            # present_current, _, _ = self.packet_handler \
            #     .read2ByteTxRx(self.port_handler, tentacle, self.config.ADDR_PRESENT_CURRENT)
            present_goal_position, _, _ = self.packet_handler \
                .read4ByteTxRx(self.port_handler, tentacle, self.config.ADDR_GOAL_POSITION)

            limits = motion_params[tentacle][:2]
            desired_current = motion_params[tentacle][2]

            is_past_limit = present_position * self.mask <= limits * self.mask + self.config.POSITION_THRESHOLD

            if is_past_limit[0]:
                goal_position = limits[1]
                goal_current = desired_current
            elif is_past_limit[1]:
                goal_position = limits[0]
                goal_current = -desired_current
            else:
                goal_position = present_goal_position if np.any(np.isin(limits, present_goal_position)) else limits[0]
                sign = np.sign(self.config.DXL_NEUTRAL_POSITION - present_goal_position)
                goal_current = sign * desired_current

            self.packet_handler \
                .write4ByteTxRx(self.port_handler, tentacle, self.config.ADDR_GOAL_POSITION, goal_position)
            self.packet_handler \
                .write2ByteTxRx(self.port_handler, tentacle, self.config.ADDR_GOAL_CURRENT, goal_current)
    
    def move_pos_cur_2(self, goal_pos, goal_curr):
        for tentacle in self.tentacles:
            goal_position = int(self.initial_positions[tentacle]) + goal_pos[tentacle]
            goal_current = int(self.initial_positions[tentacle]) + goal_curr[tentacle]

            self.packet_handler \
                    .write4ByteTxRx(self.port_handler, tentacle, self.config.ADDR_GOAL_POSITION, goal_position)
            self.packet_handler \
                    .write2ByteTxRx(self.port_handler, tentacle, self.config.ADDR_GOAL_CURRENT, goal_current)

    def get_positions(self):
        positions = dict.fromkeys(self.tentacles)
        for tentacle in self.tentacles:
            present_position, _, _ = self.packet_handler \
                .read4ByteTxRx(self.port_handler, tentacle, self.config.ADDR_PRESENT_POSITION)
            positions[tentacle] = present_position
        return positions

    def get_homing_offsets(self):
        offsets = dict.fromkeys(self.tentacles)
        for tentacle in self.tentacles:
            offset, _, _ = self.packet_handler \
                .read4ByteTxRx(self.port_handler, tentacle, self.config.ADDR_HOMING_OFFSET)
            offsets[tentacle] = offset
        return offsets
    
    def get_positions_sync(self):
        positions = dict.fromkeys(self.tentacles)
        self.group_sync_read_pos.txRxPacket()
        for tentacle in self.tentacles:
            positions[tentacle] = self.group_sync_read_pos.getData(tentacle, self.config.ADDR_PRESENT_POSITION, 4)
        return positions

    
    def get_currents(self):
        currents = dict.fromkeys(self.tentacles)
        for tentacle in self.tentacles:
            present_current, _, _ = self.packet_handler \
                .read2ByteTxRx(self.port_handler, tentacle, self.config.ADDR_PRESENT_CURRENT)
            currents[tentacle] = present_current
        return currents

    def go_to_neutral(self):
        # TODO: change current direction!
        for tentacle in self.tentacles:
            # disable torque
            self.packet_handler.write1ByteTxRx(self.port_handler, tentacle, self.config.ADDR_TORQUE_ENABLE,
                                               self.config.TORQUE_DISABLE)
            # update operating mode
            self.packet_handler.write1ByteTxRx(self.port_handler, tentacle, self.config.ADDR_OPERATING_MODE,
                                               self.config.OPERATING_MODE_POS_CURRENT)
            # enable torque
            self.packet_handler.write1ByteTxRx(self.port_handler, tentacle, self.config.ADDR_TORQUE_ENABLE,
                                               self.config.TORQUE_ENABLE)

            present_position, _, _ = self.packet_handler \
                .read4ByteTxRx(self.port_handler, tentacle, self.config.ADDR_PRESENT_POSITION)

            initial_position = self.initial_positions[tentacle]

            goal_current = 25 if initial_position - present_position < 0 else -25

            # self.packet_handler \
            #     .write4ByteTxRx(self.port_handler, tentacle, self.config.ADDR_GOAL_POSITION, self.initial_positions[tentacle])

            # self.packet_handler \
            #     .write2ByteTxRx(self.port_handler, tentacle, self.config.ADDR_GOAL_CURRENT, goal_current)
