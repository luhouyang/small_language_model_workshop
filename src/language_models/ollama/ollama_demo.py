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

from ollama import chat

response = chat(
    model='smollm2',
    messages=[{'role': 'user', 'content': 'Hello!'}],
)

print(response.message.content)
