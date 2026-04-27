from app.modules.ai_sale.schema import AISaleState


class AISaleStateStore:
    def __init__(self):
        self.store = {}

    def _make_key(self, web_no, member_no):
        return f"{web_no}:{member_no}"

    def get_state(self, web_no, member_no):
        key = self._make_key(web_no, member_no)
        return self.store.get(
            key,
            AISaleState(web_no=web_no, member_no=member_no)
        )

    def set_state(self, web_no, member_no, state):
        key = self._make_key(web_no, member_no)
        self.store[key] = state

    def reset_state(self, web_no, member_no):
        key = self._make_key(web_no, member_no)
        state = AISaleState(web_no=web_no, member_no=member_no)
        self.store[key] = state
        return state


ai_sale_state_store = AISaleStateStore()