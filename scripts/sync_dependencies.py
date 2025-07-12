#%%
"""Sync dependencies from pyproject.toml to settings.ini"""

import tomli as tomllib
import re
from pathlib import Path
import shutil
    
ROOT = Path(__file__).parent.parent
pyproject_f = ROOT / 'pyproject.toml'
settings_f = ROOT / 'settings.ini'

data = tomllib.load(open(pyproject_f, 'rb'))

# Extract shared fields
main_deps = ' '.join(data.get('project', {}).get('dependencies', []))
dev_deps = ' '.join(data.get('dependency-groups', {}).get('dev', []))
python_version = data.get('project', {}).get('requires-python', '').lstrip('>=').strip('"')

# Create backup of settings.ini
backup_path = settings_f.with_suffix('.ini.backup')
shutil.copy2(settings_f, backup_path)

ini_content = open(settings_f, 'r').read()
lines = ini_content.split('\n')

# Replace existing lines or add if missing
for pattern, replacement in [
    (r'^\s*#?\s*requirements\s*=.*$', f'requirements = {main_deps}'),
    (r'^\s*#?\s*dev_requirements\s*=.*$', f'dev_requirements = {dev_deps}'),
    (r'^\s*min_python\s*=.*$', f'min_python = {python_version}')

]:
    if re.search(pattern, ini_content, re.MULTILINE):
        ini_content = re.sub(pattern, replacement, ini_content, flags=re.MULTILINE)
    else:
        ini_content += f'\n{replacement}'

open(settings_f, 'w').write(ini_content)
