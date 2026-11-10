"""
STEP 2 — A streaming chat loop with conversation memory.

    python step2_chat.py

Commands inside the chat:
    /reset    forget the conversation
    /system   change the system prompt
    /tokens   rough estimate of how much context you are using
    /quit     exit

What to notice:
  - "Memory" is just a Python list. The model is stateless; every turn you
    resend the whole history. This is the single most important thing to
    understand about how chat with an LLM actually works.
  - Watch /tokens grow. When it approaches the model's context window,
    the oldest turns have to be dropped or summarised. That is all a
    "conversation memory" library does.
"""

import sys

import ollama_client as llm

SYSTEM = "You are a concise technical assistant. Answer in at most 4 sentences."


def rough_tokens(messages):
    """~4 characters per token is close enough to reason about budget."""
    return sum(len(m["content"]) for m in messages) // 4


def main():
    llm.require(llm.DEFAULT_MODEL)

    system = SYSTEM
    history = [{"role": "system", "content": system}]

    print(f"Chatting with {llm.DEFAULT_MODEL}. /quit to exit.\n")

    while True:
        try:
            user = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history = [{"role": "system", "content": system}]
            print("[history cleared]\n")
            continue
        if user == "/tokens":
            print(f"[~{rough_tokens(history)} tokens in context, "
                  f"{len(history)} messages]\n")
            continue
        if user.startswith("/system"):
            system = user[len("/system"):].strip() or SYSTEM
            history = [{"role": "system", "content": system}]
            print(f"[system prompt set, history cleared]\n")
            continue

        history.append({"role": "user", "content": user})

        print("bot > ", end="", flush=True)
        reply = []
        try:
            for piece in llm.chat_stream(history):
                sys.stdout.write(piece)
                sys.stdout.flush()
                reply.append(piece)
        except llm.OllamaError as e:
            print(f"\n[error] {e}")
            history.pop()
            continue
        print("\n")

        history.append({"role": "assistant", "content": "".join(reply)})


if __name__ == "__main__":
    main()
