from __future__ import annotations

from typing import Any


def safe_state_dump(state: Any) -> dict:
    """
    คืน state เป็น dict แบบปลอดภัย
    รองรับ:
    - Pydantic model ที่มี model_dump()
    - dict
    - object ทั่วไปที่มี __dict__
    """
    if state is None:
        return {}

    if hasattr(state, "model_dump") and callable(state.model_dump):
        try:
            return state.model_dump()
        except Exception as e:
            return {"_dump_error": str(e), "_raw_type": str(type(state))}

    if isinstance(state, dict):
        return state

    if hasattr(state, "__dict__"):
        try:
            return dict(vars(state))
        except Exception as e:
            return {"_dump_error": str(e), "_raw_type": str(type(state))}

    return {
        "_raw_type": str(type(state)),
        "_raw_value": str(state),
    }


def print_state(label: str, state: Any) -> None:
    dumped = safe_state_dump(state)
    print(f"==== {label} ===========================================================================================================================")
    print(dumped)
    print(f"========================================================================================================================================")


def print_debug(label: str, value: Any) -> None:
    print(f"DEBUG {label} =", value)