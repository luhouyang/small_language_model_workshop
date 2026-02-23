# Ollama + Telegram

This section explores the use of locally hosted language models, and interacting with the model through Telegram. While the example is simple, agentic orchestration software (*e.g.* [local n8n with docker](https://www.digitalocean.com/community/tutorials/how-to-setup-n8n)) can be used to extend its capability and integration with databases and communication tools (*e.g.* gmail, whatsapp).

## Setup Ollama

1. Install the program [Ollama](https://ollama.com/download/windows)
1. Open your command prompt, and pull the **smollm2** model (~1.8GB)
    ```bash
    ollama pull smollm2
    ```

1. Try to chat with the model
    ```python
    from ollama import chat

    response = chat(
        model='smollm2', # put the name of model here search on ollama site for all models
        message=[{'role': 'user', 'content': ''}], # roles = ['user', 'system', 'assistant']
    )

    print(response.message.content)
    ```

    The roles usually available are:

    - user: User prompt
    - system: System message, the instructions given at initialization (personality, task domain)
    - assistant: Reply of model, usually appended as context

## Connect Telegram

### Telegram

1. Create a Telegram bot and get your token. Search for **BotFather** in the app.
1. Type **/newbot**
1. Type the name of the bot (must end with _bot, *i.e.* ai_bot)
1. Copy the token (*i.e.* 81______:______VrlW4YYOQEoGZxEJZC-tMop_7i7cQ)

### VS Code

1. Create a new **.env** file at the root of the project
1. Add the token to the file
    ```
    TELEGRAM_TOKEN="81______:______VrlW4YYOQEoGZxEJZC-tMop_7i7cQ"
    ```

1. Go to [ollama_demo.py](src/language_models/ollama/ollama_demo.py). Import dependencies:
    ```python
    import os
    import logging
    from telegram import Update
    from telegram.ext import Application, MessageHandler, filters, ContextTypes
    from dotenv import load_dotenv
    from ollama import chat
    ```

1. Write the basic structure of a Python program (main function). 

1. Setup logging to monitor Telegram responses
    ```python
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO)
    ```

1. Create a function that gets called when we get a text message. For now this function just sends the same message back.
    ```python
    async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user_text = update.message.text

        if user_text:
            await update.message.reply_text(user_text)
        else:
            # Handle cases where the message might not be text (e.g., a sticker, photo, etc.)
            await update.message.reply_text("I received a non-text message!")
    ```

1. In the main function write the main logic.
    ```python
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # Add a message handler that filters for text messages and not command (starting with /), 
    # then uses the 'echo' function
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, echo)
        )

    print("Bot is running... Press Ctrl+C to stop.")
    application.run_polling(poll_interval=3)
    ```

1. Send a message to your Telegram bot, you should get the same message as reply.

## Add SmolLM2 model in the reply logic

1. In the `if user_text:` scope add the model call, return the response
    ```python
    .
    .
    .
    if user_text:
        response = chat(
            model='smollm2',
            messages=[{'role': 'user', 'content': user_text}],
        )

        ai_text = response.message.content

        await update.message.reply_text(ai_text)
    ```

1. Test the response.

1. The model currently doesn't have context of previous chat, add a list to keep track of context. Also add system prompt to change model behaviour.
    ```python
    SYSTEM_PROMPT = {
        'role':
        'system',
        'content':
        'You are a helpful, witty assistant. Keep your answers concise and friendly.'
    }
    session_context = [SYSTEM_PROMPT]
    .
    .
    .
    async def echo(...):
        user_text = update.message.text

        if user_text:
            session_context.append({'role': 'user', 'content': user_text})

            response = chat(
            model='smollm2',
            messages=session_context,
        )

        ai_text = response.message.content

        session_context.append({'role': 'assistant', 'content': ai_text})

        await update.message.reply_text(ai_text)
    ```

1. Test the context of the model.

1. Currently, the model has unlimited context, but [SmolLM2](https://ollama.com/library/smollm2) only has 8k token context.
    ```python
    .
    .
    .
    if user_text:
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
    ```

1. We also want the user to be able to bomb the context, add an if statement to check for 💣 and clear context.
    ```python
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
    ```

1. Check if everything is working. Take 5 minutes to play with the settings / system prompt.
