from app.modules.ai_sale.schema import AISaleState


class AISaleStateStore:
    def __init__(self):
        self.store = {}

    def get_state(self, chat_id: str):
        return self.store.get(
            chat_id,
            AISaleState(chat_id=chat_id)
        )

    def set_state(self, chat_id: str, state):
        self.store[chat_id] = state

    def reset_state(self, chat_id: str):
        state = AISaleState(chat_id=chat_id)
        self.store[chat_id] = state
        return state


ai_sale_state_store = AISaleStateStore()