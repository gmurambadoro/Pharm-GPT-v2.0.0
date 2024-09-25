#!/usr/bin/env python
import click


@click.group()
def cli():
    """Prints "Hello, world!" to the screen"""
    pass

@click.command()
def test():
    click.echo("Test")

cli.add_command(test)

if __name__ == "__main__":
    cli()
