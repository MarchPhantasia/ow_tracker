from __future__ import annotations

import ctypes


class X2Hotkey:
    VK_XBUTTON1 = 0x05
    VK_XBUTTON2 = 0x06

    def __init__(self, button: str = "x2"):
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("hotkey polling is only available on Windows")
        normalized = button.strip().lower()
        if normalized not in {"x1", "x2"}:
            raise ValueError("hotkey button must be x1 or x2")
        self._vk = self.VK_XBUTTON2 if normalized == "x2" else self.VK_XBUTTON1
        self._user32 = ctypes.windll.user32

    def is_pressed(self) -> bool:
        return bool(self._user32.GetAsyncKeyState(self._vk) & 0x8000)
