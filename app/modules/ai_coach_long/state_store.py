from app.modules.ai_coach.schema import ChatState

class ChatStateStore:
    def __init__(self):
        self._store = {}

    def _make_key(self, web_no: int | None, member_no: int | None) -> str:
        web_no = web_no or 0
        member_no = member_no or 0
        return f"web_{web_no}_member_{member_no}"

    def get_state(self, web_no: int | None, member_no: int | None) -> ChatState:
        key = self._make_key(web_no, member_no)
        return self._store.get(key, ChatState())

    def set_state(self, web_no: int | None, member_no: int | None, state: ChatState):
        key = self._make_key(web_no, member_no)
        self._store[key] = state

    def reset_state(self, web_no: int | None, member_no: int | None) -> ChatState:
        key = self._make_key(web_no, member_no)
        self._store[key] = ChatState()
        return self._store[key]

chat_state_store_aicoach = ChatStateStore()