from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# number of recent user turns to remember (user+assistant pairs)
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "6"))


client = OpenAI(api_key=API_KEY)


def _trim_history(history, max_user_turns: int):
    """Trim history to the last `max_user_turns` user turns (and their assistant replies)."""
    max_entries = max_user_turns * 2  # user + assistant per turn
    if len(history) <= max_entries:
        return history
    return history[-max_entries:]


def _format_history_prompt(history):
    """Format the memory into a single prompt string that precedes the current user input.

    Example output:
    Conversation so far:
    User: Hi
    Bot: Hello!
    User: How are you?
    Bot:
    """
    if not history:
        return ""
    lines = ["Conversation so far:"]
    for role, text in history:
        prefix = "User" if role == "user" else "Bot"
        lines.append(f"{prefix}: {text}")
    lines.append("Bot:")
    return "\n".join(lines)


def main():
    print("Simple Chatbot (Responses API). Type 'exit' or 'quit' to stop.")
    print("Type '/reset' to clear short-term memory.")

    history = []  # list of (role, text) tuples where role is 'user' or 'assistant'

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if user_input.strip() == "/reset":
            history = []
            print("Memory cleared.")
            continue

        # Add user input to history, trim to configured size
        history.append(("user", user_input))
        history = _trim_history(history, MEMORY_TURNS)

        # Build prompt: memory (if any) + current user text. Keep simple: memory formatted, then user input.
        memory_prompt = _format_history_prompt(history)
        if memory_prompt:
            prompt = memory_prompt + "\n" + user_input
        else:
            prompt = user_input

        try:
            resp = client.responses.create(model=MODEL, input=prompt)
        except Exception as e:
            print("API error:", e)
            continue

        # Prefer the convenience property if available
        text = getattr(resp, "output_text", None)
        if text:
            print("Bot:", text)
            # store assistant reply in history
            history.append(("assistant", text))
            history = _trim_history(history, MEMORY_TURNS)
            continue

        print("Bot: (no text returned)")


if __name__ == "__main__":
    main()