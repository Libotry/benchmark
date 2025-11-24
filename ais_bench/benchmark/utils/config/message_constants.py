"""Shared constants for communication between tasks and inferencers."""
import struct
from collections import OrderedDict

# Message queue format for communication with subprocesses: 6 integers.
# The 6 integers represent status, post, recv, fail, finish, and data_index respectively.
# Using signed integers to support -1 for data_index
FMT = "7I1i"
MESSAGE_TYPE_NUM = 5
MESSAGE_SIZE = struct.calcsize(FMT)
STATUS_REPORT_INTERVAL = 1
WAIT_FLAG = 2
SYNC_MAIN_PROCESS_INTERVAL = 0.1


class _MessageInfo:
    STATUS = None
    POST = None
    RECV = None
    FAIL = None
    FINISH = None
    CASE_FINISH = None
    DATA_SYNC_FLAG = None
    DATA_INDEX = None


MESSAGE_INFO = _MessageInfo()

FIELDS = OrderedDict(
    [
        ("STATUS", "I"),
        ("POST", "I"),
        ("RECV", "I"),
        ("FAIL", "I"),
        ("FINISH", "I"),
        ("CASE_FINISH", "I"),
        ("DATA_SYNC_FLAG", "I"),
        ("DATA_INDEX", "i"),
    ]
)

# Calculate offsets for each field
offset = 0
for name, fmt in FIELDS.items():
    size = struct.calcsize(fmt)
    setattr(MESSAGE_INFO, name, (offset, offset + size))
    offset += size

