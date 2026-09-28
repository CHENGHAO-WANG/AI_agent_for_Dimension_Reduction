"""The machine's memory, and the share of it a Run's candidates may use.

The limit is half the physical memory, the same for every budget. It depends on the
machine, so a plan valid on one could be refused on another. To keep a Run's verdicts
stable, the limit is measured once, at the Run's first registration, recorded in the
registration record, and reused by every later registration of that Run.

Standard library only: `psutil` would be one more dependency for one number.
"""

from __future__ import annotations

import ctypes
import os
import sys

#: The share of physical memory a candidate's estimated peak may reach.
MEMORY_SHARE = 0.5


def physical_memory_bytes() -> int:
    """Total physical memory, as the operating system reports it."""
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("GlobalMemoryStatusEx failed")
        return int(status.ullTotalPhys)
    return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))


def memory_limit() -> dict[str, int]:
    """The limit for a Run registering for the first time, with what it was taken from."""
    physical = physical_memory_bytes()
    return {"physical_bytes": physical, "limit_bytes": int(physical * MEMORY_SHARE)}
