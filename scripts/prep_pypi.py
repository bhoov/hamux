#!/usr/bin/env python3
"""Bump version and sync from settings.ini to pyproject.toml for PyPI release"""

import subprocess
import sys
import re
import configparser
from pathlib import Path
from fastcore.script import *

ROOT = Path(__file__).parent.parent
version_part = sys.argv[1] if len(sys.argv) > 1 else '2'

@call_parse
def main(
    version_part:int=2, # 0: major, 1: minor, 2: patch
):
    print("Syncing dependencies from pyproject.toml to settings.ini...")
    subprocess.run([sys.executable, ROOT / 'scripts' / 'sync_dependencies.py'], check=True)

    # Bump version
    print("Bumping version...")
    subprocess.run(['uv', 'run', 'nbdev_bump_version', '--part', str(version_part)], check=True)

    # Read version from settings.ini
    config = configparser.ConfigParser()
    config.read(ROOT / 'settings.ini')
    version = config['DEFAULT']['version']

    # Update pyproject.toml
    content = open(ROOT / 'pyproject.toml', 'r').read()
    content = re.sub(r'^version\s*=\s*["\'].*["\']', f'version = "{version}"', content, flags=re.MULTILINE)
    open(ROOT / 'pyproject.toml', 'w').write(content)

    print(f"Complete. Now run `nbdev_pypi` to push package")
