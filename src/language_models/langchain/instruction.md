# LangChain + Ollama: The Infinite Story Generator

In the [previous section](src/language_models/ollama/instruction.md), we connected a simple model to Telegram. Now, we’re going to level up using **LangChain**. AI slop maxxing.

## Get the Model

Using a bigger model [llama3.2:1b](https://ollama.com/library/llama3.2:1b). In the terminal:

```bash
ollama pull llama3.2:1b
```

## Setup LangChain

Go to [auto_novel.py](src/language_models/langchain/auto_novel.py). Unlike the raw `ollama` library we used earlier, `langchain_ollama` gives us "**Chains**" "**|**" to link different prompts together.

```python
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the LLM
llm = ChatOllama(model="llama3.2:1b", temperature=0.7)
parser = StrOutputParser()
```

## Define the Prompts

We need two distinct "personalities" for our AI: the **Novelist** and the **Editor**. The variables in the string will later be replaced when calling the model.

```python
# The Novelist: Writes the zesty details
story_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional novelist. Write 1000 words for the requested chapter."),
    ("user", "Theme: {theme}\nSummary so far: {summary}\nPrevious end: {context}\n\nWrite Chapter {chapter_num}:")
])

# The Editor: Keeps the story on track (Summary)
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "Update the story summary concisely."),
    ("user", "Current Summary: {current_summary}\nRecent Chapter: {new_content}\n\nNew Summary:")
])

```

## Loop Writing

After generating each chapter, we summarize the previous and new chapter. The summary + last 1200 characters (~200 words) are used to make the next chapter.

```python
def write_story(theme):
    summary, context, chapters = "The story is just beginning.", "Once upon a time...", []

    for i in range(1, 6):
        print(f"Writing Chapter {i}...")
        
        # Chain it together: Prompt -> Model -> Clean Text
        chapter = (story_prompt | llm | parser).invoke({
            "theme": theme, "summary": summary, "context": context, "chapter_num": i
        })
        
        chapters.append(f"## Chapter {i}\n\n{chapter}")
        
        # Update context for the next chapter
        context = chapter[-1200:] # Grab the end of the chapter for continuity
        summary = (summary_prompt | llm | parser).invoke({
            "current_summary": summary, "new_content": chapter
        })

    return "\n\n---\n\n".join(chapters)

```

## Run & Save

```python
story = write_story("Novel idea here")

with open("./story.md", "w+") as f:
    f.write(story)

print("\nStory Complete! Check story.md to read your creation.")

```
