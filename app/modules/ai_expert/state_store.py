from typing import Optional
from app.modules.ai_expert.schema import ChatState_aiexpert


class ChatStateStoreAIExpert:
    def __init__(self):
        self.store: dict[str, ChatState_aiexpert] = {}

    def _make_key(self, room_id: Optional[int]) -> str:
        if room_id is None:
            raise ValueError("room_id is required for ChatStateStoreAIExpert")

        return f"room:{room_id}"

    def get_state(self, room_id: Optional[int]) -> ChatState_aiexpert:
        key = self._make_key(room_id)

        if key not in self.store:
            self.store[key] = ChatState_aiexpert()

        return self.store[key]

    def set_state(
        self,
        room_id: Optional[int],
        state: ChatState_aiexpert,
    ) -> ChatState_aiexpert:
        key = self._make_key(room_id)
        self.store[key] = state
        return state

    def reset_state(
        self,
        room_id: Optional[int],
        web_no: Optional[int] = None,
        member_no: Optional[int] = None,
        course_use: Optional[list] = None,
    ) -> ChatState_aiexpert:
        key = self._make_key(room_id)

        state = ChatState_aiexpert(
            web_no=web_no,
            member_no=member_no,
            course_use=course_use or [],
        )

        self.store[key] = state
        return state


chat_state_store_aicustom = ChatStateStoreAIExpert()