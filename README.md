LangChain Chat with Search
A Streamlit-based AI chat application that combines Groq-hosted LLaMA 3.1 with external knowledge tools including Wikipedia, Arxiv, and DuckDuckGo search to answer technical questions in a more grounded and interactive way.

The app uses a ReAct-style LangChain agent, keeps chat history inside Streamlit session state, supports streaming responses, and includes a retry-enabled safe search tool to make web search more reliable during runtime.

Features
Streamlit chat interface with persistent session-based conversation history.

Groq-powered LLM integration using llama-3.1-8b-instant through ChatGroq.

Multi-tool agent workflow using Wikipedia, Arxiv, and DuckDuckGo search.

Custom safe_search tool with retry logic using tenacity and a lightweight network availability check with requests.

ReAct agent execution with LangChain's create_react_agent and AgentExecutor.

Streamlit callback handler to display tool reasoning steps live in the UI.

Clear chat history option from the sidebar.

Fallback direct LLM response when the agent hits iteration or time limits, or raises an exception.

Basic guardrails to skip weak or invalid search queries.

Overview
This project is designed for users who want a chat assistant that can answer technical questions using both an LLM and external search tools instead of relying only on model memory.

When the user enters a query, the application creates a Groq-backed LangChain ReAct agent, allows that agent to call supporting tools, and displays the final response in a Streamlit chat layout.

Tech Stack
Category	Tools / Libraries
Frontend	Streamlit 
LLM Provider	Groq 
Model	llama-3.1-8b-instant 
Agent Framework	LangChain Classic ReAct Agent 
Search / Knowledge Tools	Wikipedia, Arxiv, DuckDuckGo 
Reliability / Retry	Tenacity, Requests 
Configuration	python-dotenv, environment variables 
How It Works
The app initializes a Wikipedia wrapper with top_k_results=1 and doc_content_chars_max=200, and an Arxiv wrapper with top_k_results=1 and doc_content_chars_max=2300, then exposes both as LangChain tools.

It also defines a custom safe_search tool that rejects weak queries, avoids problematic patterns such as site: and conversation, checks basic network availability, and retries failed attempts with exponential backoff up to three times.

At runtime, the app creates a ReAct agent using the hwchase17/react prompt from LangChain Hub, executes it with a maximum of 5 iterations and 60 seconds, and falls back to a direct Groq model invocation if the agent exceeds time or parsing limits.

Chat History Handling
Instead of passing a structured conversation object into the agent, the app converts previous messages into a plain-text chat_history_text string and injects that history directly into the current prompt.

This design is explicitly used to improve reliability with the selected LLaMA 3 model on Groq and to avoid template issues by sending an empty chat_history list to the agent executor.

Search Safety Logic
The safe_search tool is intentionally constrained to reduce noisy or unsafe search behavior.

It skips queries that contain fewer than two words, contain site: filters, or include the word conversation, and it returns a graceful error string when a network check fails before trying DuckDuckGo search.

User Interface
The Streamlit UI contains a main chat area and a sidebar for configuration.

The sidebar accepts the Groq API key through a password field and also provides a button to clear chat history, which resets the session state and reruns the app.

The main view replays prior messages from st.session_state, collects new prompts with st.chat_input, and writes both user and assistant messages in chat-style containers.

Project Flow
text
User Query
   ↓
Streamlit Chat UI
   ↓
Groq LLM Initialization
   ↓
LangChain ReAct Agent
   ↓
Wikipedia / Arxiv / Safe Search Tools
   ↓
Agent Response
   ↓
Fallback Direct LLM Response (if needed)
   ↓
Chat History Saved in Session State
Project Structure
A minimal structure for this project would look like this:

text
.
├── app.py
├── .env
├── requirements.txt
└── README.md
If you are keeping the exact code you shared in a single file, app.py is a practical filename for the Streamlit entry point.

Setup
Prerequisites
Before running the project, make sure you have the following:

Python 3.10 or newer.

A valid Groq API key.

Internet connectivity for Wikipedia, Arxiv, and DuckDuckGo search access.

Environment Variables
Create a .env file in the project root and store your credentials there if needed.

A simple version can look like this:

text
GROQ_API_KEY=your_groq_api_key
Even though the app takes the API key from the Streamlit sidebar at runtime, using .env remains useful for local development and future extension.

Install Dependencies
Install the required packages with:

bash
pip install streamlit langchain-groq langchain-community langchain-classic langchain-core python-dotenv tenacity requests wikipedia arxiv duckduckgo-search
If you prefer, you can also save them into a requirements.txt file for reproducible setup.

Run the App
Start the Streamlit application with:

bash
streamlit run app.py
Once the app opens in the browser:

Enter your Groq API key in the sidebar.

Ask a technical question in the chat box.

Watch the agent use tools when needed and stream back a response.

Example Queries
This project is especially suited for prompts such as:

Explain retrieval-augmented generation.

Find recent Arxiv papers on diffusion models.

Who is Alan Turing?

Search for the latest updates on vector databases.

Compare transformers and RNNs.

Key Components
safe_search
This is a custom LangChain tool wrapped with @tool and @retry, and it acts as a safer DuckDuckGo search layer with query validation and retry behavior.

ChatGroq
The app uses ChatGroq with temperature=0.1 and streaming=True, which helps keep answers more stable and allows token streaming into the UI.

AgentExecutor
The agent executor runs the ReAct agent with handle_parsing_errors=True, max_iterations=5, max_execution_time=60, and return_intermediate_steps=False.

StreamlitCallbackHandler
The callback handler streams intermediate reasoning activity into the app interface through a parent container, which improves transparency while the agent is working.

Fallback Behavior
If the agent response indicates an iteration limit or time limit issue, or if an exception occurs during execution, the app falls back to a direct model call using the same augmented prompt.

This makes the system more robust by ensuring users still get an answer even when tool-driven reasoning fails.

Strengths
Easy to run locally with a single Streamlit file.

Combines search tools with LLM reasoning for more grounded answers.

Includes practical retry and fallback behavior.

Preserves conversational continuity through session-managed chat history.

Limitations
The app depends on external APIs and internet access for search-based answers.

Chat history is injected as plain text, which is simple but may become harder to scale for long conversations.

Search validation rules are basic and may reject some useful advanced queries.

The fallback direct LLM call does not use tool outputs when the agent fails mid-run.

Future Improvements
Add support for more tools such as PubMed, Stack Overflow, or custom documentation search.

Move API key handling to optional environment-based defaults with secure UI override.

Add message persistence using a database instead of temporary session state.

Add source citation formatting in the final answer.

Add prompt templates specialized for technical tutoring or research workflows.

Add rate-limit handling and better exception reporting in the UI.

Suggested Filename
If you want the repository to look clean, a good project name would be LangChain Chat with Search and the main file can be named app.py or streamlit_app.py.

License
Add your preferred license here, such as MIT, Apache-2.0, or another open-source license of your choice.
