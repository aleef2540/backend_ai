from app.modules.ai_self_learning.schema import ChatState_aiselflearning


class ChatStateStoreAISelfLearning:
    def __init__(self):
        self.store = {}

    def get_state(self, chat_id: str) -> ChatState_aiselflearning:
        return self.store.get(
            chat_id,
            ChatState_aiselflearning(chat_id=chat_id)
        )

    def set_state(self, chat_id: str, state: ChatState_aiselflearning):
        self.store[chat_id] = state

    def reset_state(self, chat_id: str) -> ChatState_aiselflearning:
        state = ChatState_aiselflearning(chat_id=chat_id)
        self.store[chat_id] = state
        return state


chat_state_store_aiselflearning = ChatStateStoreAISelfLearning()