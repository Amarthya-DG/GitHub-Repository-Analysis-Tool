import os
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from github import Github
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from litellm import completion

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_ACCESS_TOKEN not found in environment variables")

try:
    g = Github(GITHUB_TOKEN)
    # Test the connection
    user = g.get_user()
    print(f"Successfully authenticated as: {user.login}")
except Exception as e:
    if "401" in str(e):
        raise Exception(
            "Invalid GitHub token. Please check your GITHUB_ACCESS_TOKEN in .env file"
        )
    else:
        raise Exception(f"Failed to initialize GitHub client: {str(e)}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


class LiteLLMWrapper(BaseLLM):
    """Wrapper for LiteLLM to make it compatible with LangChain"""

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stop=stop,
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "custom"


def get_llm(model_name: str = "gpt-3.5-turbo") -> LiteLLMWrapper:
    """Get LLM based on model name"""
    supported_models = {
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-4": "gpt-4",
        "claude-2": "claude-2",
        "gemini-pro": "gemini-pro",
    }

    if model_name not in supported_models:
        raise ValueError(
            f"Model {model_name} not supported. Choose from {list(supported_models.keys())}"
        )

    return LiteLLMWrapper(model_name)


def fetch_all_files_with_content_and_metadata(
    owner: str, repo: str, path: str = ""
) -> List[Dict[str, Any]]:
    """Recursively fetch all files, their content, and metadata."""
    try:
        repo_obj = g.get_repo(f"{owner}/{repo}")
        contents = repo_obj.get_contents(path)
        all_files = []

        for content_file in contents:
            if content_file.type == "file":
                file_content = content_file.decoded_content.decode("utf-8")
                commits = repo_obj.get_commits(path=content_file.path)
                latest_commit = commits[0]

                metadata = {
                    "last_committed_by": latest_commit.commit.author.name,
                    "last_commit_date": latest_commit.commit.author.date.isoformat(),
                    "last_commit_message": latest_commit.commit.message,
                    "github_url": f"https://github.com/{owner}/{repo}/blob/main/{content_file.path}",
                }

                all_files.append(
                    {
                        "name": content_file.name,
                        "path": content_file.path,
                        "content": file_content,
                        "metadata": metadata,
                    }
                )
            elif content_file.type == "dir":
                all_files.extend(
                    fetch_all_files_with_content_and_metadata(
                        owner, repo, content_file.path
                    )
                )

        return all_files
    except Exception as e:
        return {"error": str(e)}


def create_repo_tools(vector_store, llm) -> List[Tool]:
    """Create tools for repository analysis with web search capabilities"""

    search_prompt = PromptTemplate(
        input_variables=["query"],
        template="Search the repository for information about: {query}",
    )

    search_chain = LLMChain(llm=llm, prompt=search_prompt)

    web_search = DuckDuckGoSearchRun()
    wikipedia = WikipediaAPIWrapper()

    tools = [
        Tool(
            name="Search Repository",
            func=lambda q: vector_store.similarity_search(q),
            description="Search for information in the repository's codebase",
        ),
        Tool(
            name="Analyze Code",
            func=lambda q: search_chain.run(q),
            description="Analyze code patterns and structure in the repository",
        ),
        Tool(
            name="Web Search",
            func=web_search.run,
            description="Search the internet for general programming concepts",
        ),
        Tool(
            name="Wikipedia",
            func=wikipedia.run,
            description="Search Wikipedia for technical concepts",
        ),
    ]

    return tools


def setup_agent(vector_store, llm) -> AgentExecutor:
    """Set up an agent with enhanced search capabilities"""
    tools = create_repo_tools(vector_store, llm)

    prompt = PromptTemplate.from_template(
        """You are a helpful AI assistant for analyzing GitHub repositories.
        Use the following tools to help answer questions:
        {tools}
        
        Follow these steps:
        1. First search the repository for relevant information
        2. If repository search doesn't yield complete answers, use web search or Wikipedia
        3. Combine information from all sources to provide a comprehensive answer
        
        Question: {input}
        """
    )

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        early_stopping_method="generate",
    )

    return agent_executor


def store_in_vector_store(files: List[Dict], readme: Dict) -> FAISS:
    """Store files and README content into a vector store"""
    try:
        documents = []

        for file in files:
            documents.append(
                {
                    "content": file["content"],
                    "metadata": {
                        **file["metadata"],
                        "name": file["name"],
                        "path": file["path"],
                    },
                }
            )

        if "content" in readme:
            documents.append(
                {
                    "content": readme["content"],
                    "metadata": {"name": readme["file_name"], "path": readme["path"]},
                }
            )

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = []
        metadata = []

        for doc in documents:
            splits = text_splitter.split_text(doc["content"])
            chunks.extend(splits)
            metadata.extend([doc["metadata"]] * len(splits))

        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.from_texts(chunks, embeddings, metadatas=metadata)

        return vector_store
    except Exception as e:
        raise Exception(f"Error storing in vector store: {str(e)}")


def main():
    st.title("Enhanced GitHub Repository Analysis")

    # Model selection
    model_name = st.selectbox(
        "Select LLM Model", ["gpt-3.5-turbo", "gpt-4", "claude-2", "gemini-pro"]
    )

    owner = st.text_input("Enter GitHub Owner/Organization:")
    repo = st.text_input("Enter Repository Name:")

    if st.button("Fetch and Analyze Repository"):
        if not owner or not repo:
            st.error("Please provide both owner and repository name.")
            return

        # Validate repository exists
        try:
            repo_obj = g.get_repo(f"{owner}/{repo}")
            # Test if repo is accessible
            repo_obj.get_contents("")
        except Exception as e:
            st.error(f"Repository not found or not accessible: {owner}/{repo}")
            st.error(f"Error: {str(e)}")
            return

        with st.spinner("Fetching repository data..."):
            all_files = fetch_all_files_with_content_and_metadata(owner, repo)
            if isinstance(all_files, dict) and "error" in all_files:
                st.error(f"Error fetching files: {all_files['error']}")
                return

            readme = {"content": "", "file_name": "README.md", "path": "README.md"}
            try:
                repo_obj = g.get_repo(f"{owner}/{repo}")
                readme_content = repo_obj.get_readme()
                readme["content"] = readme_content.decoded_content.decode("utf-8")
            except Exception as e:
                st.warning("README not found, continuing without it.")
                st.error(f"Error fetching README: {str(e)}")

        with st.spinner("Setting up analysis tools..."):
            try:
                vector_store = store_in_vector_store(all_files, readme)
                llm = get_llm(model_name)
                agent_executor = setup_agent(vector_store, llm)

                st.session_state["vector_store"] = vector_store
                st.session_state["agent_executor"] = agent_executor
                st.success("Repository analysis tools ready!")
            except Exception as e:
                st.error(f"Error setting up analysis tools: {str(e)}")
                return

    if "agent_executor" in st.session_state:
        st.subheader("Query the Repository")
        question = st.text_input("Ask a question about the repository:")

        if st.button("Analyze"):
            if not question:
                st.error("Please enter a question.")
            else:
                try:
                    with st.spinner("Analyzing..."):
                        response = st.session_state["agent_executor"].run(question)
                        st.write("### Analysis Result")
                        st.write(response)
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")


if __name__ == "__main__":
    main()

# This is your main application file with all the functionality
