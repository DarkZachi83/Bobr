"""CPU i RAM – psutil działa tak samo na Linuksie i Windowsie (SPEC §13)."""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)


def ram_info() -> tuple[Optional[int], Optional[int]]:
    """(całkowity, zajęty) w bajtach; (None, None), gdy psutil nie działa."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.total - vm.available)
    except Exception as e:  # brak psutil albo nietypowe środowisko
        log.warning("psutil unavailable: %s", e)
        return None, None
