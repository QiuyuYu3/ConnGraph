import argparse
import os
import pathlib
import sys

import conngraph
from conngraph.cli.main import build_parser

sys.path.insert(0, str(pathlib.Path(__file__).parent / "_ext"))
from gallery_scraper import GalleryScraper

os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")

project = "ConnGraph"
author = "Qiuyu Yu"
copyright = "2026, Qiuyu Yu"
release = conngraph.__version__

extensions = ["myst_parser", "sphinx_design", "sphinx_gallery.gen_gallery"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "cli/_generated"]
myst_enable_extensions = ["colon_fence"]
myst_heading_anchors = 3
# keep the double hyphen of option names in prose and tables
smartquotes_action = "qe"

html_theme = "pydata_sphinx_theme"
html_title = "ConnGraph"
html_theme_options = {
    "github_url": "https://github.com/QiuyuYu3/brainnet3d",
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc", "sg_download_links", "sg_launcher_links"],
}
html_sidebars = {"index": [], "getting-started": [], "auto_examples/index": []}

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "subsection_order": ["../examples/graph_metrics", "../examples/figures", "../examples/group_comparisons"],
    "image_scrapers": (GalleryScraper(),),
    "within_subsection_order": "FileNameSortKey",
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_signature": False,
}

# argparse group title -> table file included by the CLI reference pages
CLI_TABLES = {
    "positional arguments": "general",
    "options": "general",
    "input": "input",
    "both levels": "both-levels",
    "participant level: graph metrics": "participant",
    "group level: comparing two groups": "group",
}


def _cli_text(text: str) -> str:
    text = " ".join(text.split())
    for char in "\\|<>*":
        text = text.replace(char, "\\" + char)
    return text


def write_cli_tables() -> None:
    parser = build_parser()
    formatter = parser._get_formatter()
    tables: dict[str, list[str]] = {}
    for group in parser._action_groups:
        rows = tables.setdefault(CLI_TABLES[group.title], ["| Option | Values | Description |", "|---|---|---|"])
        for action in group._group_actions:
            if isinstance(action, argparse._HelpAction):
                continue
            names = ", ".join(f"`{s}`" for s in action.option_strings) or f"`{action.metavar or action.dest}`"
            if action.choices:
                values = ", ".join(f"`{c}`" for c in action.choices)
            elif action.nargs == 0 or not action.option_strings:
                values = ""
            else:
                values = f"`{formatter._format_args(action, action.dest.upper())}`"
            values = values.replace("|", "\\|")
            rows.append(f"| {names} | {values} | {_cli_text(formatter._expand_help(action))} |")
    out = pathlib.Path(__file__).parent / "cli" / "_generated"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        path, text = out / f"{name}.md", "\n".join(rows) + "\n"
        # rewriting an unchanged table would make Sphinx reread the pages that include it
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")


write_cli_tables()
