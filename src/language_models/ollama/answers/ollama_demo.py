"""
By:             Lu Hou Yang
Last updated:   19th Feb 2025

Ollama demo
1. Download and install Ollama from website and pip
2. Change settings to allow network access
3. Sign in to account
4. Open command prompt, enter: ollama pull smollm2
5. Confirm model list with: ollama list
"""

# from ollama import chat
# response = chat(
#     model='smollm2',
#     messages=[{'role': 'user', 'content': 'What are the specialty foods of Malaysia?'}],
# )
# print(response.message.content)

import os
import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv
from ollama import chat

load_dotenv(".env")
TOKEN = os.getenv("TELEGRAM_TOKEN")

# start session with a system prompt
SYSTEM_PROMPT = {
    'role':
    'system',
    'content':
    'You are a helpful, witty assistant. Keep your answers concise and friendly.'
}
session_context = [SYSTEM_PROMPT]

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO)


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text

    if user_text:
        if user_text.strip() == "💣":
            session_context.clear()
            session_context.append(SYSTEM_PROMPT)
            await update.message.reply_text(
                "Context cleared! 💥 How can I help you now?")
            return

        session_context.append({'role': 'user', 'content': user_text})

        response = chat(
            model='smollm2',
            messages=session_context,
        )

        ai_text = response.message.content

        session_context.append({'role': 'assistant', 'content': ai_text})

        if len(session_context) > 20:
            session_context.pop(1)

        await update.message.reply_text(ai_text)
    else:
        # Handle cases where the message might not be text (e.g., a sticker, photo, etc.)
        await update.message.reply_text("I received a non-text message!")


def main() -> None:
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # Add a message handler that filters for text messages and uses the 'echo' function
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND,
                       echo))

    print("Bot is running... Press Ctrl+C to stop.")
    application.run_polling(poll_interval=3)


if __name__ == "__main__":
    main()
