import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_classic import hub
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import requests


load_dotenv()


# Wikipedia Wrapper
api_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wiki)


# Arxiv Wrapper
api_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=2300)
arxiv = ArxivQueryRun(api_wrapper=api_arxiv)


# Safe DuckDuckGo Search
@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def safe_search(query: str) -> str:
    """Safe web search tool. Use for factual public info only."""
    if len(query.split()) < 2 or "site:" in query.lower() or "conversation" in query.lower():
        return "Skipped invalid search query."
    try:
        requests.get("https://duckduckgo.com/?q=test", timeout=5)
    except:
        return "Network error - search unavailable."
    return DuckDuckGoSearchRun().run(query)


# ---- Streamlit UI ----
st.title("Langchain-Chat with Search")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your GROQ key", type="password")

# ✅ Clear chat history button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me anything technical! 🚀"}
    ]
    st.rerun()


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! Ask me anything technical! 🚀"}
    ]


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input("Enter your query"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)


    if not api_key:
        st.chat_message("assistant").write("⚠️ Please enter your GROQ API key!")
        st.stop()


    llm = ChatGroq(
        groq_api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0.1,
        streaming=True
    )


    tools = [wiki, arxiv, safe_search]
    prompt_template = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt_template)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=5,
        max_execution_time=60,
        return_intermediate_steps=False
    )

    # ✅ Build chat history as plain text string (reliable with LLaMA 3 + Groq)
    chat_history_text = ""
    for msg in st.session_state.messages[1:-1]:  # skip greeting and latest user prompt
        role = "User" if msg["role"] == "user" else "Assistant"
        chat_history_text += f"{role}: {msg['content']}\n"

    # ✅ Inject history directly into the input prompt
    augmented_prompt = prompt
    if chat_history_text:
        augmented_prompt = f"""Previous conversation:
{chat_history_text}
Current question: {prompt}"""


    with st.chat_message("assistant"):
        parent_container = st.container()
        st_cb = StreamlitCallbackHandler(parent_container, expand_new_thoughts=False)


        try:
            result = agent_executor.invoke(
                {
                    "input": augmented_prompt,   # ✅ history-aware prompt
                    "chat_history": []           # ✅ empty list avoids template errors
                },
                callbacks=[st_cb]
            )
            response = result["output"]


            if "iteration limit" in response.lower() or "time limit" in response.lower():
                fallback_llm = ChatGroq(groq_api_key=api_key, model="llama-3.1-8b-instant")
                response = fallback_llm.invoke(augmented_prompt).content  # ✅ use augmented


        except Exception as e:
            fallback_llm = ChatGroq(groq_api_key=api_key, model="llama-3.1-8b-instant")
            response = fallback_llm.invoke(augmented_prompt).content      # ✅ use augmented


        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


    st.rerun()
