"""Interactive REPL for the Abaqus Analysis Agent."""

import sys
import os
import json
import time
import logging
import threading

# Ensure project root is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dotenv import load_dotenv
load_dotenv(os.path.join(_project_root, ".env"))

from agent import AbaqusAgent

# Setup logging
_log_dir = os.path.join(_project_root, "logs")
os.makedirs(_log_dir, exist_ok=True)

log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_log_dir, "agent.log"), encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Session history save path
_session_dir = os.path.join(_project_root, "logs")
_session_path = os.path.join(_session_dir, "session_history.json")


def _save_session(history):
    """Save conversation history to JSON file."""
    try:
        with open(_session_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Failed to save session: %s", e)


def _load_session():
    """Load previous session history from JSON file."""
    if os.path.exists(_session_path):
        try:
            with open(_session_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


class Spinner:
    """Simple terminal spinner for long-running operations."""

    def __init__(self, message="处理中"):
        self._message = message
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        # Clear spinner line
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()

    def _spin(self):
        chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while self._running:
            sys.stdout.write(f"\r{chars[i % len(chars)]} {self._message}...")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)


HELP_TEXT = """\
可用命令:
  /help     — 显示此帮助
  /history  — 查看对话历史
  /save     — 保存当前会话
  /clear    — 清空对话历史
  /export   — 导出对话历史到 JSON 文件
  quit/exit — 退出程序
"""


def main():
    print("Abaqus Analysis Agent")
    print("输入分析需求开始，输入 /help 查看帮助\n")
    logger.info("CLI session started")

    agent = AbaqusAgent()
    session_log = []

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            _save_session(session_log)
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        # Commands
        if stripped.lower() in ("quit", "exit"):
            _save_session(session_log)
            print("会话已保存。再见!")
            break

        if stripped == "/help":
            print(HELP_TEXT)
            continue

        if stripped == "/history":
            if not session_log:
                print("(暂无对话历史)")
            else:
                for entry in session_log:
                    role = "用户" if entry["role"] == "user" else "助手"
                    text = entry["content"][:100]
                    print(f"  [{role}] {text}{'...' if len(entry['content']) > 100 else ''}")
            continue

        if stripped == "/save":
            _save_session(session_log)
            print(f"会话已保存到 {_session_path}")
            continue

        if stripped == "/export":
            export_path = os.path.join(_session_dir,
                                       f"session_{time.strftime('%Y%m%d_%H%M%S')}.json")
            try:
                with open(export_path, "w", encoding="utf-8") as f:
                    json.dump(session_log, f, ensure_ascii=False, indent=2)
                print(f"已导出到 {export_path}")
            except Exception as e:
                print(f"导出失败: {e}")
            continue

        if stripped == "/clear":
            agent.history.clear()
            session_log.clear()
            print("对话历史已清空")
            continue

        if stripped.startswith("/"):
            print(f"未知命令: {stripped}  输入 /help 查看可用命令")
            continue

        # Normal chat
        session_log.append({"role": "user", "content": user_input})

        spinner = Spinner("Kimi 生成中")
        spinner.start()
        try:
            response = agent.chat(user_input)
        finally:
            spinner.stop()

        session_log.append({"role": "assistant", "content": response})

        print()
        print(response)
        print()

    logger.info("CLI session ended")


if __name__ == "__main__":
    main()
