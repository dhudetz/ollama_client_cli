import curses
import requests
import json
import time
import signal
import re
import threading
from typing import Generator, Optional


class OllamaClient:
    def __init__(self, model: str = "llama3.3", host: str = "http://localhost:11434", stream: bool = True):
        self.model = model
        self.host = host.rstrip("/")
        self.stream = stream
        self.chat_history = []

    def _post(self, endpoint: str, data: dict, stream: bool = False) -> Optional[Generator[dict, None, None]]:
        url = f"{self.host}{endpoint}"
        try:
            response = requests.post(url, json=data, stream=stream)
            response.raise_for_status()
            if stream:
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)
            else:
                return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Request to {url} failed: {e}")

    def chat(self, msg: str) -> Generator[str, None, None] | str:
        self.chat_history.append({"role": "user", "content": msg})
        payload = {
            "model": self.model,
            "messages": self.chat_history,
            "stream": self.stream
        }

        if self.stream:
            return self._streaming_chat_response(payload)
        else:
            result = self._post("/api/chat", payload)
            response = result.get("message", {}).get("content", "[No response]")
            self.chat_history.append({"role": "assistant", "content": response})
            return response

    def _streaming_chat_response(self, payload: dict) -> Generator[str, None, None]:
        response_text = ""
        for chunk in self._post("/api/chat", payload, stream=True):
            content = chunk.get("message", {}).get("content", "")
            response_text += content
            yield content
        self.chat_history.append({"role": "assistant", "content": response_text})


solarized_colors = [
    (131, 148, 150), (147, 161, 161), (108, 113, 196),
    (42, 161, 152), (38, 139, 210), (211, 54, 130), (133, 153, 0)
]


def init_rainbow_colors() -> None:
    if not curses.can_change_color() or curses.COLORS < 16:
        return
    for idx, (r, g, b) in enumerate(solarized_colors, start=10):
        curses.init_color(idx, int(r / 255 * 1000), int(g / 255 * 1000), int(b / 255 * 1000))
        curses.init_pair(idx, idx, -1)


def draw_rainbow_name(win, y: int, x: int, name: str, frame: int) -> None:
    for i, ch in enumerate(name):
        color_idx = 10 + (i + frame) % len(solarized_colors)
        try:
            win.addstr(y, x + i, ch, curses.color_pair(color_idx))
        except curses.error:
            pass


def draw_header(win, width: int) -> None:
    safe_width = max(10, width - 1)
    try:
        win.clear()
        win.attron(curses.color_pair(3))
        win.addstr(0, 0, "╭" + "─" * (safe_width - 2) + "╮")
        title = " Chatting with Ollama 🧠 "
        win.addstr(1, 0, "│" + title.center(safe_width - 2) + "│")
        win.addstr(2, 0, "╰" + "─" * (safe_width - 2) + "╯")
        win.attroff(curses.color_pair(3))
        win.refresh()
    except curses.error:
        pass


class ChatInterface:
    def __init__(self, stdscr, client: OllamaClient):
        self.stdscr = stdscr
        self.client = client
        self.assistant_name = re.sub(r"\d", "", client.model).capitalize()
        self.messages = []
        self.abort_stream = threading.Event()

    def run(self):
        curses.curs_set(1)
        curses.start_color()
        curses.use_default_colors()

        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_GREEN, -1)
        curses.init_pair(4, curses.COLOR_MAGENTA, -1)
        init_rainbow_colors()

        while True:
            self.draw_layout()
            user_input = self.get_input()

            if user_input.strip().lower() in {"exit", "quit", ":q", ":wq"}:
                self.show_bye()
                return

            self.messages.append(("user", user_input))
            self.abort_stream.set()  # cancel any prior stream
            self.abort_stream = threading.Event()  # reset abort event
            self.stream_response(user_input)

    def draw_layout(self):
        self.stdscr.clear()
        self.max_y, self.max_x = self.stdscr.getmaxyx()
        self.input_height = 3
        self.header_height = 3

        self.input_win = curses.newwin(self.input_height, self.max_x, self.max_y - self.input_height, 0)
        self.history_win = curses.newwin(self.max_y - self.input_height - self.header_height, self.max_x, self.header_height, 0)
        self.header_win = curses.newwin(self.header_height, self.max_x, 0, 0)

        draw_header(self.header_win, self.max_x)
        self.redraw_history()

    def redraw_history(self, frame=0):
        self.history_win.clear()
        y = 0
        for role, text in self.messages[-(self.max_y - self.input_height - self.header_height - 1):]:
            lines = text.splitlines()
            if role == "assistant":
                draw_rainbow_name(self.history_win, y, 0, f"{self.assistant_name}:", frame)
                y += 1
                for line in lines:
                    if y >= self.max_y - self.input_height - self.header_height - 1:
                        break
                    self.history_win.addstr(y, 2, line + "\n", curses.color_pair(2))
                    y += 1
            else:
                for line in lines:
                    if y >= self.max_y - self.input_height - self.header_height - 1:
                        break
                    self.history_win.addstr(y, 0, "You: ", curses.color_pair(1))
                    self.history_win.addstr(line + "\n")
                    y += 1
        self.history_win.refresh()

    def get_input(self):
        self.input_win.clear()
        self.input_win.addstr(0, 0, "You: ", curses.color_pair(1))
        self.input_win.refresh()
        curses.echo()
        try:
            user_input = self.input_win.getstr(1, 0, self.max_x - 2).decode("utf-8")
        except KeyboardInterrupt:
            return ""
        curses.noecho()
        return user_input

    def stream_response(self, user_input):
        self.messages.append(("assistant", ""))
        stream = self.client.chat(user_input)
        frame = 0
        bot_response = ""
        self.history_win.nodelay(True)

        if hasattr(stream, '__iter__') and not isinstance(stream, str):
            for chunk in stream:
                if self.abort_stream.is_set():
                    return
                try:
                    ch = self.history_win.getch()
                    if ch != -1:
                        self.abort_stream.set()
                        curses.ungetch(ch)
                        self.messages[-1] = ("assistant", bot_response + "\n[interrupted]")
                        return
                except curses.error:
                    pass
                bot_response += chunk
                self.messages[-1] = ("assistant", bot_response)
                self.redraw_history(frame)
                frame += 1
        else:
            self.messages[-1] = ("assistant", stream)
            self.redraw_history()

    def show_bye(self):
        self.stdscr.clear()
        msg = "bye"
        y, x = self.stdscr.getmaxyx()
        self.stdscr.addstr(y // 2, (x - len(msg)) // 2, msg, curses.color_pair(3) | curses.A_BOLD)
        self.stdscr.refresh()
        time.sleep(1)


def start_chat_interface():
    client = OllamaClient(stream=True)

    def draw_screen(stdscr):
        interface = ChatInterface(stdscr, client)
        interface.run()

    curses.wrapper(draw_screen)


if __name__ == "__main__":
    signal.signal(signal.SIGWINCH, lambda n, f: None)  # prevent resize from crashing
    start_chat_interface()
