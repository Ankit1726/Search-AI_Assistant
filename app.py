import os
import requests
import streamlit as st
import warnings

from pydantic.warnings import PydanticDeprecatedSince20

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_react_agent, AgentExecutor

# ---------------------------
# LOAD ENV VARIABLES
# ---------------------------
from dotenv import load_dotenv
load_dotenv()

GROQ = os.getenv("GROQ_API_KEY")
WEATHER = os.getenv("WEATHERSTACK_API_KEY")
TAVILY = os.getenv("TAVILY_API_KEY")

# Hide warnings
warnings.filterwarnings(
    "ignore",
    category=PydanticDeprecatedSince20
)

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Search Assistant",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------
# CUSTOM DARK UI
# ---------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color:white;
}

.main-title {
    text-align:center;
    font-size:3rem;
    font-weight:800;
    color:#38bdf8;
    margin-top:20px;
}

.sub-title {
    text-align:center;
    color:#94a3b8;
    margin-bottom:30px;
}

.custom-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding:25px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.1);
}

.stTextInput > div > div > input {
    background:#1e293b;
    color:white;
    border-radius:12px;
    border:1px solid #38bdf8;
    padding:12px;
}

.stButton > button {
    width:100%;
    height:52px;
    border:none;
    border-radius:12px;
    color:white;
    font-size:18px;
    font-weight:bold;
    background:linear-gradient(
        135deg,
        #06b6d4,
        #3b82f6
    );
    transition:0.3s;
}

.stButton > button:hover {
    transform:scale(1.02);
}

.response-box {
    background:rgba(255,255,255,0.05);
    border-left:5px solid #38bdf8;
    padding:20px;
    border-radius:15px;
    margin-top:20px;
}

footer {
    visibility:hidden;
}

header {
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# HEADER
# ---------------------------
st.markdown("""
<div class="main-title">
🤖 Search AI Assistant
</div>

<div class="sub-title">
Search • Weather • AI Agent Powered by LangChain & Groq
</div>
""", unsafe_allow_html=True)

# ---------------------------
# TOOLS
# ---------------------------
search_tool = TavilySearchResults(
    max_results=2,
    tavily_api_key=TAVILY
)

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"http://api.weatherstack.com/current?"
        f"access_key={WEATHER}&query={city}"
    )

    response = requests.get(url)
    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )

# ---------------------------
# LLM
# ---------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ
)

# ---------------------------
# PROMPT
# ---------------------------
prompt = hub.pull("hwchase17/react")

# ---------------------------
# AGENT
# ---------------------------
tools = [
    search_tool,
    get_weather_data
]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors=True
)

# ---------------------------
# INPUT SECTION
# ---------------------------
st.markdown('<div class="custom-card">', unsafe_allow_html=True)

user_query = st.text_input(
    "Enter your query",
    placeholder="Find the capital of India and current weather..."
)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# RUN AGENT
# ---------------------------
if st.button("🚀 Run Agent"):

    if user_query:

        with st.spinner("🤖 Agent is thinking..."):

            try:

                response = agent_executor.invoke(
                    {"input": user_query}
                )

                st.success("Response Generated Successfully")

                st.markdown(
                    f"""
                    <div class="response-box">
                        <h3>✨ Final Response</h3>
                        <p>{response['output']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )

    else:

        st.warning(
            "Please enter a query."
        )
