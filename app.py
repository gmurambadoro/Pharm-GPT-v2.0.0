#!/usr/bin/env python
import os

import chromadb
import click
from bs4 import BeautifulSoup
from chromadb import Settings

SRC_DIR_HTML = "/data/pharm/html"
SRC_DIR_TEXT = "/data/pharm/text"

CHROMA_PATH = "./chroma"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings())


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
@click.option('--drop', default=False, is_flag=True, help="Drop the collection first before indexing")
@click.argument("collection_name", default="pharm", type=str)
def index_documents(drop: bool, collection_name: str):
    """Stores text files in a ChromaDb vector database"""
    if drop:
        try:
            chroma_client.delete_collection(collection_name)
        except Exception as e:
            click.echo(e)

    collection = chroma_client.get_or_create_collection(name=collection_name)

    print(collection.id, drop)


@click.command()
@click.argument("text", type=str)
def query(text: str):
    click.echo(text)


cli.add_command(generate_text_files)
cli.add_command(index_documents)
cli.add_command(query)



if __name__ == "__main__":
    cli()
