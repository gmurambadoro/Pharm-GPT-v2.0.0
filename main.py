#!/usr/bin/env python
import os

import click
from bs4 import BeautifulSoup


@click.group()
def cli():
    """Prints "Hello, world!" to the screen"""
    pass

@click.command(name="generate-text-files")
def generate_text_files():
    """Generates text files from /data/pharm/*/*.html files"""
    for (root, dirs, files) in os.walk("/data/pharm/", topdown=True):
        with click.progressbar(files, label=f"{root} ({len(files or [])} files)") as folder_files:
            for file in folder_files:
                filename = str(os.path.join(root, file) or None).strip()

                if not filename.endswith(".html"):
                    continue

                dir_path = os.path.dirname(filename)
                save_dir = dir_path.replace("/data/pharm", "/data/pharm_text")
                save_filename = os.path.join(save_dir, f"{os.path.basename(filename)}.txt".replace(".html", ""))

                os.makedirs(save_dir, exist_ok=True)

                with open(filename, mode="r") as f:
                    html = f.read()

                    soup = BeautifulSoup(html, features="html.parser")

                    with open(save_filename, "w") as f2:
                        f2.write(soup.get_text().strip())

cli.add_command(generate_text_files)

if __name__ == "__main__":
    cli()
