from typing import Optional
from app.modules.ai_custom.schema import ChatState_aicustom


class ChatStateStoreAICustom:
    def __init__(self):
        self.store: dict[str, ChatState_aicustom] = {}

    def _make_key(self, room_id: Optional[int]) -> str:
        if room_id is None:
            raise ValueError("room_id is required for ChatStateStoreAICustom")

        return f"room:{room_id}"

    def get_state(self, room_id: Optional[int]) -> ChatState_aicustom:
        key = self._make_key(room_id)

        if key not in self.store:
            self.store[key] = ChatState_aicustom()

        return self.store[key]

    def set_state(
        self,
        room_id: Optional[int],
        state: ChatState_aicustom,
    ) -> ChatState_aicustom:
        key = self._make_key(room_id)
        self.store[key] = state
        return state

    def reset_state(
        self,
        room_id: Optional[int],
        web_no: Optional[int] = None,
        member_no: Optional[int] = None,
        course_use: Optional[list] = None,
    ) -> ChatState_aicustom:
        key = self._make_key(room_id)

        state = ChatState_aicustom(
            web_no=web_no,
            member_no=member_no,
            course_use=course_use or [],
        )

        self.store[key] = state
        return state


chat_state_store_aicustom = ChatStateStoreAICustom()