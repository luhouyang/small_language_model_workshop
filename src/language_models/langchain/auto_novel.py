# ollama pull llama3.2:1b
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.2:1b", temperature=0.7)
parser = StrOutputParser()

story_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional novelist. Write 1000 words for the requested chapter. Only include the story related text."
     ),
    ("user",
     "Theme: {theme}\nSummary so far: {summary}\nPrevious end: {context}\n\nWrite Chapter {chapter_num}:"
     )
])

summary_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Update the story summary concisely."),
    ("user",
     "Current Summary: {current_summary}\nRecent Chapter: {new_content}\n\nNew Summary:"
     )
])


def write_story(theme):
    summary, context, chapters = "The story is just beginning.", "Once upon a time...", []

    for i in range(1, 6):
        print(f"Writing Chapter {i}...")
        chapter = (story_prompt | llm | parser).invoke({
            "theme": theme,
            "summary": summary,
            "context": context,
            "chapter_num": i
        })
        chapters.append(f"## Chapter {i}\n\n{chapter}")
        context = chapter[-1200:]
        summary = (summary_prompt | llm | parser).invoke({
            "current_summary": summary,
            "new_content": chapter
        })

    return "\n\n---\n\n".join(chapters)


story = write_story("A funny bird that grows 1cm bigger everyday.")
print("\nStory Complete!")

with open("./src/language_models/langchain/story.md", "w+") as f:
    f.write(story)
