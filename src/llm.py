import os

from dotenv import load_dotenv

load_dotenv()

# Stops the model from inventing a follow-up question or fake sources after its answer
STOP_SEQUENCES = ["\nQuestion", "\nType: Source", "\nSources:"]


def get_llm(ollama_model="phi3"):
    """Return the chat model selected by LLM_PROVIDER in .env ("groq", "mistral" or "ollama")."""
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "LLM_PROVIDER=groq mais GROQ_API_KEY est absent du fichier .env"
            )

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0.2,
            # gpt-oss reasons before answering; its reasoning counts against this limit
            max_tokens=2000,
        )

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        if not os.getenv("MISTRAL_API_KEY"):
            raise RuntimeError(
                "LLM_PROVIDER=mistral mais MISTRAL_API_KEY est absent du fichier .env"
            )

        return ChatMistralAI(
            model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
            temperature=0.2,
            max_tokens=800,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", ollama_model),
            mirostat=2,
            mirostat_tau=5,
            mirostat_eta=0.1,
            num_predict=600,
            stop=STOP_SEQUENCES,
        )

    raise ValueError(f"LLM_PROVIDER inconnu : {provider!r} (attendu : groq, mistral ou ollama)")
