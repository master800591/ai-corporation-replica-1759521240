#!/usr/bin/env python3
"""
AI Platform Utility Functions

Pure utility functions for AI platform operations.
No demos, no interactive code, just tools.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import os


def create_project_structure(project_name: str, project_type: str = "python") -> Dict[str, Any]:
    """Create a basic project structure"""
    project_path = Path(project_name)
    
    try:
        # Create base directories
        project_path.mkdir(exist_ok=True)
        (project_path / "src").mkdir(exist_ok=True)
        (project_path / "tests").mkdir(exist_ok=True)
        (project_path / "docs").mkdir(exist_ok=True)
        
        # Create basic files
        readme_content = f"# {project_name}\n\nA {project_type} project.\n"
        (project_path / "README.md").write_text(readme_content)
        
        gitignore_content = _get_gitignore_template(project_type)
        (project_path / ".gitignore").write_text(gitignore_content)
        
        return {
            'success': True,
            'project_path': str(project_path.absolute()),
            'project_type': project_type,
            'structure': _get_directory_structure(project_path)
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def get_project_info(project_path: str) -> Dict[str, Any]:
    """Get information about a project"""
    path = Path(project_path)
    
    if not path.exists():
        return {'exists': False}
    
    info = {
        'exists': True,
        'name': path.name,
        'path': str(path.absolute()),
        'is_directory': path.is_dir(),
        'files': []
    }
    
    if path.is_dir():
        info['files'] = [f.name for f in path.iterdir()]
        info['structure'] = _get_directory_structure(path)
    
    return info


def list_projects(base_path: str = ".") -> List[Dict[str, Any]]:
    """List all projects in a directory"""
    base = Path(base_path)
    projects = []
    
    try:
        for item in base.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                project_info = get_project_info(str(item))
                projects.append(project_info)
    except Exception:
        pass
    
    return projects


def create_python_module(module_path: str, content: str = None) -> bool:
    """Create a Python module file"""
    try:
        path = Path(module_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if content is None:
            content = f'"""{path.stem} module"""\n\n# Module implementation here\n'
        
        path.write_text(content)
        return True
    except Exception:
        return False


def create_config_file(config_path: str, config_data: Dict[str, Any]) -> bool:
    """Create a configuration file"""
    try:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.endswith('.json'):
            path.write_text(json.dumps(config_data, indent=2))
        else:
            # Simple key=value format
            lines = [f"{k}={v}" for k, v in config_data.items()]
            path.write_text("\n".join(lines))
        
        return True
    except Exception:
        return False


def _get_gitignore_template(project_type: str) -> str:
    """Get gitignore template for project type"""
    common = """
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

.venv/
venv/
ENV/
env/

.idea/
.vscode/
*.swp
*.swo
*~

.DS_Store
Thumbs.db
""".strip()
    
    if project_type == "python":
        return common + "\n\n# Python specific\n*.log\n.pytest_cache/\n.coverage\n"
    elif project_type == "web":
        return common + "\n\n# Web specific\nnode_modules/\n*.log\ndist/\n"
    else:
        return common


def _get_directory_structure(path: Path, max_depth: int = 3, current_depth: int = 0) -> Dict[str, Any]:
    """Get directory structure as a dict"""
    if current_depth >= max_depth:
        return {'type': 'directory', 'children': '...'}
    
    structure = {'type': 'directory', 'children': {}}
    
    try:
        for item in path.iterdir():
            if item.name.startswith('.'):
                continue
                
            if item.is_dir():
                structure['children'][item.name] = _get_directory_structure(
                    item, max_depth, current_depth + 1
                )
            else:
                structure['children'][item.name] = {'type': 'file'}
    except Exception:
        structure['children'] = {'error': 'Permission denied'}
    
    return structure