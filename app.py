#!/usr/bin/env python
import os

import chromadb
import click
from bs4 import BeautifulSoup
from chromadb import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

SRC_DIR_HTML = "/data/pharm/html/"
SRC_DIR_TEXT = "/data/pharm/text/"

CHROMA_PATH = "./chroma"

COLLECTION_NAME = "pharm"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(allow_reset=False))

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)


@click.group()
def cli():
    """Pharm-GPT v2.0.0"""
    pass


@click.command(name="convert-html-to-text-files")
def generate_text_files():
    """Generates text files from /data/pharm/*/*.html files"""
    for (root, dirs, files) in os.walk(SRC_DIR_HTML, topdown=True):
        with click.progressbar(files, label=f"{root} ({len(files or [])} files)") as folder_files:
            for file in folder_files:
                filename = str(os.path.join(root, file) or None).strip()

                if not filename.endswith(".html"):
                    continue

                dir_path = os.path.dirname(filename)
                save_dir = dir_path.replace(SRC_DIR_HTML, SRC_DIR_TEXT)
                save_filename = os.path.join(save_dir, f"{os.path.basename(filename)}.txt".replace(".html", ""))

                os.makedirs(save_dir, exist_ok=True)

                with open(filename, mode="r") as f:
                    html = f.read()

                    soup = BeautifulSoup(html, features="html.parser")

                    with open(save_filename, "w") as f2:
                        f2.write(soup.get_text().strip())


@click.command(name="index-docs")
@click.option('--drop/--no-drop', default=True, is_flag=True, help="Drop the collection first before indexing")
def index_documents(drop: bool):
    """Stores text files in a ChromaDb vector database"""
    collection_name = COLLECTION_NAME
    if drop:
        try:
            click.echo(f"Resetting ChromaDB collection {collection_name}..")
            vectorstore.reset_collection()
            click.echo(f"Successfully reset ChromaDB collection {collection_name}.")
        except Exception as e:
            click.echo(f"ERROR: {e}")

    for (root, dirs, files) in os.walk(SRC_DIR_TEXT):
        if "/index" in root:  # ignore index directory
            continue

        with click.progressbar(files,
                               label=root, show_eta=True, show_percent=True,
                               show_pos=True) as _files:
            for file in _files:
                try:
                    filename = os.path.join(root, file)

                    loader = TextLoader(file_path=filename)

                    docs = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter()

                    all_chunks = text_splitter.split_documents(documents=docs)

                    vectorstore.add_documents(all_chunks)

                    # vectorstore.add_documents(
                    #     ids=UUIDGenerator(ids_len=len(docs)),
                    #     documents=list(map(lambda x: x.page_content, docs)),
                    #     metadatas=list(map(lambda _: {
                    #         "source": filename.replace(SRC_DIR_TEXT, ""),
                    #         "business": os.path.dirname(filename).replace(SRC_DIR_TEXT, ""),
                    #         "generic": os.path.basename(filename).replace(".txt", ""),
                    #     }, docs))
                    # )
                except Exception as e:
                    click.echo(f"E: {e}")


@click.command(name="ask")
@click.argument("text", type=str)
def query(text: str):
    try:
        if not str(text).strip():
            raise Exception("No text provided.")
        #
        # collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_func)
        #
        # # rag_context = collection.query(query_texts=[text], n_results=5)
        # #
        # # print(rag_context)
        #
        # llm = ChatOllama(model="llama3.1", temperature=0)
        #
        # print(response)

        retriever = vectorstore.as_retriever()

        response = retriever.invoke(text)

        print(response)
    except Exception as e:
        click.echo(e)


cli.add_command(generate_text_files)
cli.add_command(index_documents)
cli.add_command(query)

if __name__ == "__main__":
    cli()
