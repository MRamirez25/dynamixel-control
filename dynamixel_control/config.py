class Config:

    # Control table address
    ADDR_BAUD_RATE              = 8
    ADDR_OPERATING_MODE         = 11
    ADDR_HOMING_OFFSET          = 20
    ADDR_TORQUE_ENABLE          = 64
    ADDR_POSITION_D_GAIN        = 80
    ADDR_POSITION_I_GAIN        = 82
    ADDR_POSITION_P_GAIN        = 84
    ADDR_GOAL_CURRENT           = 102
    ADDR_GOAL_VELOCITY          = 104
    ADDR_GOAL_POSITION          = 116
    ADDR_PRESENT_CURRENT        = 126
    ADDR_PRESENT_POSITION       = 132
    DXL_MINIMUM_POSITION_VALUE  = 0         # CCW: -180 deg
    DXL_MAXIMUM_POSITION_VALUE  = 4095      # CW: 180 deg
    DXL_NEUTRAL_POSITION        = 0
    DXL_MINIMUM_CURRENT         = -1193     # CCW: -theta_dot
    DXL_MAXIMUM_CURRENT         = 1193      # CW: theta_dot
    BAUDRATE                    = 1_000_000
    OPERATING_MODE_CURRENT      = 0
    OPERATING_MODE_VELOCITY     = 1
    OPERATING_MODE_POSITION     = 3
    OPERATING_MODE_POS_CURRENT  = 5
    OPERATING_MODE_EXTENDED_POSITION = 4

    TORQUE_ENABLE               = 1
    TORQUE_DISABLE              = 0
    PROTOCOL_VERSION            = 2.0

    LEN_GOAL_POSITION           = 4
    LEN_PRESENT_POSITION        = 4
    LEN_GOAL_CURRENT            = 2
    LEN_PRESENT_CURRENT         = 2

    def __init__(self, device_name='/dev/ttyUSB0', position_threshold=20) -> None:
        super().__init__()
        self.DEVICENAME = device_name
        self.POSITION_THRESHOLD = position_threshold
