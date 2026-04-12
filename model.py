from langchain_openai import ChatOpenAI

# Connects to LLaMA-Factory API server (start it with start-api-server.bat)
# The API is OpenAI-compatible, running on localhost:8000

def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy",
        model="Qwen2.5-1.5B-Instruct",
        temperature=temperature,
    )
