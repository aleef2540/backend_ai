from app.modules.ai_custom.schema import ChatState_aicustom


class ChatStateStoreAICustom:
    def __init__(self):
        self.store = {}

    def _make_key(self, web_no: int | None, member_no: int | None):
        return f"{web_no}:{member_no}"

    def get_state(self, web_no: int | None, member_no: int | None) -> ChatState_aicustom:
        key = self._make_key(web_no, member_no)

        return self.store.get(
            key,
            ChatState_aicustom(
                web_no=web_no,
                member_no=member_no,
            )
        )

    def set_state(self, web_no: int | None, member_no: int | None, state: ChatState_aicustom):
        key = self._make_key(web_no, member_no)
        self.store[key] = state

    def reset_state(self, web_no: int | None, member_no: int | None):
        key = self._make_key(web_no, member_no)

        state = ChatState_aicustom(
            web_no=web_no,
            member_no=member_no,
        )

        self.store[key] = state
        return state


chat_state_store_aicustom = ChatStateStoreAICustom()