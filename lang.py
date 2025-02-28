import os

import streamlit as st
from dotenv import load_dotenv
from github import Github
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize GitHub client
g = Github(GITHUB_TOKEN)


def fetch_all_files_with_content_and_metadata(owner, repo, path=""):
    """Recursively fetch all files, their content, and metadata."""
    try:
        repo_obj = g.get_repo(f"{owner}/{repo}")
        contents = repo_obj.get_contents(path)
        all_files = []

        for content_file in contents:
            if content_file.type == "file":
                # Fetch the content of the file
                file_content = content_file.decoded_content.decode("utf-8")

                # Fetch metadata of the file (latest commit details)
                commits = repo_obj.get_commits(path=content_file.path)
                latest_commit = commits[0]  # Get the latest commit
                metadata = {
                    "last_committed_by": latest_commit.commit.author.name,
                    "last_commit_date": latest_commit.commit.author.date.isoformat(),
                    "last_commit_message": latest_commit.commit.message,
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
                # Recursively fetch contents of the directory
                all_files.extend(
                    fetch_all_files_with_content_and_metadata(
                        owner, repo, content_file.path
                    )
                )

        return all_files
    except Exception as e:
        return {"error": str(e)}


def fetch_readme(owner, repo):
    """Fetch the README file of a repository."""
    try:
        repo_obj = g.get_repo(f"{owner}/{repo}")
        readme_content = repo_obj.get_readme()
        return {
            "file_name": readme_content.name,
            "path": readme_content.path,
            "content": readme_content.decoded_content.decode("utf-8"),
        }
    except Exception as e:
        return {"error": str(e)}


def store_in_vector_store(files, readme):
    """Store files and README content into a vector store for querying."""
    try:
        documents = []

        # Add file contents
        for file in files:
            documents.append(
                {
                    "content": file["content"],
                    "metadata": {"name": file["name"], "path": file["path"]},
                }
            )

        # Add README content
        if "content" in readme:
            documents.append(
                {
                    "content": readme["content"],
                    "metadata": {"name": readme["file_name"], "path": readme["path"]},
                }
            )

        # Text splitting and embeddings
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = []
        metadata = []

        for doc in documents:
            splits = text_splitter.split_text(doc["content"])
            chunks.extend(splits)
            metadata.extend([doc["metadata"]] * len(splits))

        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = FAISS.from_texts(chunks, embeddings, metadatas=metadata)

        return vector_store
    except Exception as e:
        raise Exception(f"Error storing in vector store: {str(e)}")


def query_with_gemini(question, vector_store):
    """Use Google Gemini to answer the question based on the repository content."""
    # Perform a similarity search to find relevant documents
    results = vector_store.similarity_search(question, k=3)

    # Extract content from the results
    content = "\n\n".join([result.page_content for result in results])

    # Generate the query for Gemini
    query = f"Answer the following question based on the repository content:\n\n{content}\n\nQuestion: {question}"

    # Use Gemini to generate an answer
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    response = llm.generate(prompts=[query], max_output_tokens=512)
    return response.generations[0][0].text.strip()


def main():
    st.title("GitHub Repository Query App")

    owner = st.text_input("Enter GitHub Owner/Organization:")
    repo = st.text_input("Enter Repository Name:")

    if st.button("Fetch and Store Data"):
        if not owner or not repo:
            st.error("Please provide both owner and repository name.")
            return

        # Fetch all files and their content
        all_files = fetch_all_files_with_content_and_metadata(owner, repo)
        if isinstance(all_files, dict) and "error" in all_files:
            st.error(f"Error fetching files: {all_files['error']}")
            return

        # Fetch README
        readme = fetch_readme(owner, repo)
        if isinstance(readme, dict) and "error" in readme:
            st.error(f"Error fetching README: {readme['error']}")
            return

        # Store data into vector store
        try:
            vector_store = store_in_vector_store(all_files, readme)
            st.session_state["vector_store"] = (
                vector_store  # Save vector store in session state
            )
            st.success("Data successfully stored in vector store. Ready for querying!")
        except Exception as e:
            st.error(f"Error storing data: {str(e)}")

    if "vector_store" in st.session_state:
        st.subheader("Query the Repository")
        question = st.text_input("Ask a question about the repository:")
        if st.button("Submit Query"):
            if not question:
                st.error("Please enter a question.")
            else:
                try:
                    vector_store = st.session_state["vector_store"]
                    answer = query_with_gemini(question, vector_store)
                    st.write(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Error querying vector store: {str(e)}")


if __name__ == "__main__":
    main()
