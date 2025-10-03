"""
Ollama Model Manager

This tool searches https://ollama.com/search for available models and provides
functionality to discover, filter, and pull models automatically.
"""

import requests
import subprocess
import json
import time
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import concurrent.futures
from urllib.parse import urljoin, quote
import sys


def run_ollama_command(cmd: List[str], timeout: int = 30, capture_output: bool = True) -> subprocess.CompletedProcess:
    """
    Helper function to run ollama commands with proper encoding handling.
    
    Args:
        cmd: Command list to execute
        timeout: Timeout in seconds
        capture_output: Whether to capture output
        
    Returns:
        CompletedProcess with proper encoding handling
    """
    try:
        return subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace'
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Command timed out after {timeout} seconds: {' '.join(cmd)}")
    except Exception as e:
        raise RuntimeError(f"Failed to run command {' '.join(cmd)}: {e}")


@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    name: str
    full_name: str
    description: str
    tags: List[str]
    pulls: int
    updated: str
    size: Optional[str] = None
    capabilities: List[str] = None
    url: str = ""
    context_size: Optional[int] = None
    parameter_count: Optional[str] = None
    model_family: Optional[str] = None
    quantization: Optional[str] = None
    license: Optional[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


class OllamaModelManager:
    """
    Manages Ollama models by searching the official website and providing
    tools to discover, filter, and pull models.
    """
    
    def __init__(self):
        self.base_url = "https://ollama.com"
        self.search_url = f"{self.base_url}/search"
        self.api_url = f"{self.base_url}/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self._available_models = []
        self._local_models = []
    
    def search_models(
        self, 
        query: str = "", 
        category: str = "", 
        page_limit: int = 10
    ) -> List[ModelInfo]:
        """
        Search for models on Ollama website.
        
        Args:
            query: Search query term
            category: Model category filter
            page_limit: Maximum pages to search
            
        Returns:
            List of ModelInfo objects
        """
        print(f"Searching Ollama models: query='{query}', category='{category}'")
        
        models = []
        page = 1
        
        while page <= page_limit:
            try:
                # Build search URL
                params = {
                    'q': query,
                    'page': page
                }
                if category:
                    params['category'] = category
                
                print(f"Fetching page {page}...")
                response = self.session.get(self.search_url, params=params, timeout=30)
                response.raise_for_status()
                
                # Parse the HTML response to extract model information
                page_models = self._parse_search_page(response.text)
                
                if not page_models:
                    print(f"No more models found on page {page}")
                    break
                
                models.extend(page_models)
                print(f"Found {len(page_models)} models on page {page}")
                
                page += 1
                time.sleep(1)  # Be respectful to the server
                
            except Exception as e:
                print(f"Error searching page {page}: {e}")
                break
        
        print(f"Total models found: {len(models)}")
        self._available_models = models
        return models
    
    def search_models_with_details(
        self, 
        query: str = "", 
        category: str = "", 
        page_limit: int = 2,
        max_details: int = 5
    ) -> List[ModelInfo]:
        """
        Search for models and fetch detailed information including context size.
        
        Args:
            query: Search query term
            category: Model category filter
            page_limit: Maximum pages to search
            max_details: Maximum models to fetch detailed info for
            
        Returns:
            List of ModelInfo objects with detailed specifications
        """
        print(f"Searching models with detailed info: query='{query}', category='{category}'")
        
        # First get basic models
        models = self.search_models(query, category, page_limit)
        
        if not models:
            return models
        
        # Fetch detailed info for top models
        detailed_models = []
        for i, model in enumerate(models[:max_details]):
            print(f"Fetching detailed info for {model.name} ({i+1}/{min(len(models), max_details)})...")
            
            try:
                details = self.get_model_details(model.name)
                
                # Update model with detailed information
                if details.get('context_size'):
                    model.context_size = details['context_size']
                if details.get('parameter_count'):
                    model.parameter_count = details['parameter_count']
                if details.get('model_family'):
                    model.model_family = details['model_family']
                if details.get('quantization'):
                    model.quantization = details['quantization']
                if details.get('readme'):
                    # Re-extract capabilities with detailed info
                    model.capabilities = self._extract_capabilities(
                        model.tags, 
                        f"{model.description} {details['readme']}"
                    )
                
                detailed_models.append(model)
                time.sleep(0.5)  # Be respectful to the server
                
            except Exception as e:
                print(f"Error fetching details for {model.name}: {e}")
                detailed_models.append(model)  # Add without details
        
        # Add remaining models without detailed info
        detailed_models.extend(models[max_details:])
        
        return detailed_models
    
    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """
        Fetch detailed information for a specific model from its page.
        
        Args:
            model_name: Name of the model to get details for
            
        Returns:
            Dictionary with detailed model information
        """
        model_url = f"{self.base_url}/library/{model_name}"
        
        try:
            response = self.session.get(model_url, timeout=30)
            response.raise_for_status()
            
            html = response.text
            details = {}
            
            # Extract context size from various patterns in the page
            context_patterns = [
                r'context.*?(\d+)k',
                r'(\d+)k\s*context',
                r'context.*?(\d+),?(\d{3})',
                r'(\d+),?(\d{3})\s*context',
                r'window.*?(\d+)k',
                r'(\d+)k\s*window',
                r'max.*?length.*?(\d+)k',
                r'sequence.*?length.*?(\d+)k',
                r'(\d+)k\s*tokens?',
                r'tokens?.*?(\d+)k'
            ]
            
            for pattern in context_patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                if matches:
                    try:
                        if isinstance(matches[0], tuple):
                            if len(matches[0]) == 2 and matches[0][1] in ['000', '00']:
                                details['context_size'] = int(matches[0][0]) * 1000
                            else:
                                details['context_size'] = int(matches[0][0]) * 1000
                        else:
                            details['context_size'] = int(matches[0]) * 1000
                        break
                    except (ValueError, TypeError):
                        continue
            
            # Extract parameter information
            param_patterns = [
                r'(\d+\.?\d*)b\b',
                r'(\d+\.?\d*)\s*billion',
                r'(\d+)m\s*parameters',
                r'(\d+\.?\d*)m\b'
            ]
            
            for pattern in param_patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                if matches:
                    param = matches[0]
                    if 'b' in pattern or 'billion' in pattern:
                        details['parameter_count'] = f"{param}B"
                    else:
                        details['parameter_count'] = f"{param}M"
                    break
            
            # Extract quantization info
            quant_patterns = [
                r'q(\d+)_k_[ms]',
                r'q(\d+)_k',
                r'q(\d+)_0',
                r'q(\d+)_1',
                r'fp16',
                r'fp32',
                r'int8',
                r'int4'
            ]
            
            for pattern in quant_patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                if matches:
                    if pattern.startswith('q'):
                        details['quantization'] = f"Q{matches[0]}"
                    else:
                        details['quantization'] = pattern.upper()
                    break
            
            # Extract model family from title/description
            title_match = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).lower()
                details['title'] = title_match.group(1)
                
                # Extract family from title
                families = ['llama', 'mistral', 'gemma', 'phi', 'qwen', 'falcon', 'vicuna', 'dolphin']
                for family in families:
                    if family in title:
                        details['model_family'] = family.title()
                        break
            
            # Look for README or description content
            readme_match = re.search(r'<div[^>]*class="[^"]*readme[^"]*"[^>]*>(.+?)</div>', html, re.DOTALL | re.IGNORECASE)
            if readme_match:
                readme_content = readme_match.group(1)
                details['readme'] = readme_content
                
                # Try to extract context size from README if not found
                if 'context_size' not in details:
                    for pattern in context_patterns:
                        matches = re.findall(pattern, readme_content, re.IGNORECASE)
                        if matches:
                            try:
                                if isinstance(matches[0], tuple):
                                    details['context_size'] = int(matches[0][0]) * 1000
                                else:
                                    details['context_size'] = int(matches[0]) * 1000
                                break
                            except (ValueError, TypeError):
                                continue
            
            return details
            
        except Exception as e:
            print(f"Error fetching details for {model_name}: {e}")
            return {}
    
    def _parse_search_page(self, html: str) -> List[ModelInfo]:
        """
        Parse HTML search results to extract model information.
        
        Args:
            html: HTML content from search page
            
        Returns:
            List of ModelInfo objects
        """
        models = []
        
        # Try to find JSON data in the page (Ollama often includes model data in JSON)
        json_pattern = r'window\.__INITIAL_STATE__\s*=\s*({.*?});'
        json_match = re.search(json_pattern, html, re.DOTALL)
        
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                models.extend(self._extract_models_from_json(data))
            except json.JSONDecodeError:
                pass
        
        # Fallback: Parse HTML directly
        if not models:
            models.extend(self._parse_html_models(html))
        
        return models
    
    def _extract_models_from_json(self, data: Dict[str, Any]) -> List[ModelInfo]:
        """Extract model information from JSON data."""
        models = []
        
        def find_models_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'models' and isinstance(value, list):
                        for model_data in value:
                            if isinstance(model_data, dict):
                                model = self._create_model_from_data(model_data)
                                if model:
                                    models.append(model)
                    else:
                        find_models_recursive(value, f"{path}.{key}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    find_models_recursive(item, f"{path}[{i}]")
        
        find_models_recursive(data)
        return models
    
    def _create_model_from_data(self, data: Dict[str, Any]) -> Optional[ModelInfo]:
        """Create ModelInfo from data dictionary."""
        try:
            name = data.get('name', '')
            if not name:
                return None
            
            # Extract basic info
            full_name = name
            description = data.get('description', '')
            tags = data.get('tags', [])
            pulls = data.get('pulls', 0)
            updated = data.get('updated_at', '')
            size = data.get('size', '')
            
            # Extract technical specifications
            context_size = self._extract_context_size(data, description)
            parameter_count = self._extract_parameter_count(name, description, data)
            model_family = self._extract_model_family(name, description)
            quantization = self._extract_quantization(name, description)
            license = data.get('license')
            
            # Extract capabilities
            capabilities = self._extract_capabilities(tags, description)
            
            return ModelInfo(
                name=name.split(':')[0] if ':' in name else name,
                full_name=full_name,
                description=description,
                tags=tags if isinstance(tags, list) else [],
                pulls=pulls if isinstance(pulls, (int, str)) and str(pulls).isdigit() else 0,
                updated=updated,
                size=size,
                capabilities=capabilities,
                url=f"{self.base_url}/library/{name.split(':')[0]}",
                context_size=context_size,
                parameter_count=parameter_count,
                model_family=model_family,
                quantization=quantization,
                license=license
            )
        except Exception:
            return None
    
    def _extract_context_size(self, data: Dict[str, Any], description: str) -> Optional[int]:
        """Extract context window size from model data and description."""
        # Check if context size is directly provided in data
        if 'context_length' in data:
            try:
                return int(data['context_length'])
            except (ValueError, TypeError):
                pass
        
        if 'context_size' in data:
            try:
                return int(data['context_size'])
            except (ValueError, TypeError):
                pass
        
        # Extract from description using regex patterns
        text_to_search = f"{description} {data.get('readme', '')} {' '.join(data.get('tags', []))}".lower()
        
        # Common context size patterns
        context_patterns = [
            r'(\d+)k\s*context',
            r'context.*?(\d+)k',
            r'(\d+),?(\d{3})\s*tokens?',
            r'(\d+)k\s*tokens?',
            r'context.*?(\d+),?(\d{3})',
            r'window.*?(\d+)k',
            r'(\d+)k\s*window'
        ]
        
        for pattern in context_patterns:
            matches = re.findall(pattern, text_to_search)
            if matches:
                try:
                    if isinstance(matches[0], tuple):
                        # Handle cases like "128,000" -> ("128", "000")
                        if len(matches[0]) == 2 and matches[0][1] == '000':
                            return int(matches[0][0]) * 1000
                        else:
                            return int(matches[0][0]) * 1000  # Assume k format
                    else:
                        return int(matches[0]) * 1000  # Convert k to actual number
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _extract_parameter_count(self, name: str, description: str, data: Dict[str, Any]) -> Optional[str]:
        """Extract parameter count from model name and description."""
        # Check data first
        if 'parameters' in data:
            return str(data['parameters'])
        
        text_to_search = f"{name} {description}".lower()
        
        # Parameter count patterns
        param_patterns = [
            r'(\d+\.?\d*)b\b',  # 7b, 13b, 3.5b
            r'(\d+)billion',
            r'(\d+\.?\d*)\s*billion',
            r'(\d+)m\b',  # 125m
            r'(\d+)million'
        ]
        
        for pattern in param_patterns:
            matches = re.findall(pattern, text_to_search)
            if matches:
                param = matches[0]
                if 'b' in pattern or 'billion' in pattern:
                    return f"{param}B"
                elif 'm' in pattern or 'million' in pattern:
                    return f"{param}M"
        
        return None
    
    def _extract_model_family(self, name: str, description: str) -> Optional[str]:
        """Extract model family from name and description."""
        text_to_search = f"{name} {description}".lower()
        
        # Known model families
        families = {
            'llama': ['llama', 'alpaca'],
            'mistral': ['mistral', 'mixtral'],
            'gemma': ['gemma'],
            'phi': ['phi'],
            'qwen': ['qwen'],
            'codellama': ['codellama', 'code-llama'],
            'vicuna': ['vicuna'],
            'wizardlm': ['wizardlm', 'wizard'],
            'orca': ['orca'],
            'falcon': ['falcon'],
            'claude': ['claude'],
            'gpt': ['gpt'],
            'llava': ['llava'],
            'dolphin': ['dolphin']
        }
        
        for family, keywords in families.items():
            if any(keyword in text_to_search for keyword in keywords):
                return family.title()
        
        return None
    
    def _extract_quantization(self, name: str, description: str) -> Optional[str]:
        """Extract quantization format from name and description."""
        text_to_search = f"{name} {description}".lower()
        
        # Quantization patterns
        quant_patterns = [
            r'q(\d+)_k_[ms]',  # q4_k_m, q8_k_s
            r'q(\d+)_k',       # q4_k
            r'q(\d+)_0',       # q4_0
            r'q(\d+)_1',       # q4_1
            r'fp16',
            r'fp32',
            r'int8',
            r'int4'
        ]
        
        for pattern in quant_patterns:
            matches = re.findall(pattern, text_to_search)
            if matches:
                if pattern.startswith('q'):
                    return f"Q{matches[0]}"
                else:
                    return pattern.upper()
        
        return None
    
    def _extract_capabilities(self, tags: List[str], description: str) -> List[str]:
        """Extract capabilities from tags and description."""
        capabilities = []
        capability_keywords = {
            'vision': ['vision', 'image', 'visual', 'multimodal', 'llava'],
            'code': ['code', 'coding', 'programming', 'developer', 'codellama'],
            'chat': ['chat', 'conversation', 'assistant'],
            'embedding': ['embedding', 'vector', 'similarity', 'embed'],
            'instruct': ['instruct', 'instruction', 'following'],
            'reasoning': ['reasoning', 'logic', 'problem-solving'],
            'creative': ['creative', 'writing', 'story', 'poem'],
            'multilingual': ['multilingual', 'translation', 'language'],
            'math': ['math', 'mathematics', 'calculation', 'arithmetic'],
            'function-calling': ['function', 'tool', 'api']
        }
        
        text_to_check = f"{' '.join(tags)} {description}".lower()
        for capability, keywords in capability_keywords.items():
            if any(keyword in text_to_check for keyword in keywords):
                capabilities.append(capability)
        
        return capabilities
    
    def _parse_html_models(self, html: str) -> List[ModelInfo]:
        """Parse models from HTML when JSON parsing fails."""
        models = []
        
        # Look for model links and information in HTML
        # This is a simplified parser - you might need to adjust based on actual HTML structure
        model_pattern = r'href="[^"]*\/library\/([^"\/]+)"[^>]*>([^<]+)<'
        matches = re.findall(model_pattern, html, re.IGNORECASE)
        
        for model_name, display_name in matches:
            if model_name and not any(m.name == model_name for m in models):
                models.append(ModelInfo(
                    name=model_name,
                    full_name=model_name,
                    description=display_name,
                    tags=[],
                    pulls=0,
                    updated="",
                    url=f"{self.base_url}/library/{model_name}"
                ))
        
        return models
        """Parse models from HTML when JSON parsing fails."""
        models = []
        
        # Look for model links and information in HTML
        # This is a simplified parser - you might need to adjust based on actual HTML structure
        model_pattern = r'href="[^"]*\/library\/([^"\/]+)"[^>]*>([^<]+)<'
        matches = re.findall(model_pattern, html, re.IGNORECASE)
        
        for model_name, display_name in matches:
            if model_name and not any(m.name == model_name for m in models):
                models.append(ModelInfo(
                    name=model_name,
                    full_name=model_name,
                    description=display_name,
                    tags=[],
                    pulls=0,
                    updated="",
                    url=f"{self.base_url}/library/{model_name}"
                ))
        
        return models
    
    def get_popular_models(self, limit: int = 50) -> List[ModelInfo]:
        """
        Get popular models from Ollama.
        
        Args:
            limit: Maximum number of models to return
            
        Returns:
            List of popular ModelInfo objects
        """
        print("Fetching popular models...")
        
        # Search without query to get popular models
        models = self.search_models(query="", page_limit=5)
        
        # Sort by pulls (popularity)
        models.sort(key=lambda x: x.pulls, reverse=True)
        
        return models[:limit]
    
    def search_by_category(self, category: str) -> List[ModelInfo]:
        """
        Search models by category.
        
        Args:
            category: Category name (e.g., 'embedding', 'code', 'vision')
            
        Returns:
            List of ModelInfo objects in category
        """
        print(f"Searching models in category: {category}")
        return self.search_models(category=category, page_limit=3)
    
    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with detailed model information
        """
        print(f"Getting details for model: {model_name}")
        
        try:
            # Try to get model page
            model_url = f"{self.base_url}/library/{model_name}"
            response = self.session.get(model_url, timeout=30)
            response.raise_for_status()
            
            # Parse model details from page
            details = self._parse_model_details(response.text)
            details['url'] = model_url
            details['name'] = model_name
            
            return details
            
        except Exception as e:
            print(f"Error getting model details: {e}")
            return {'name': model_name, 'error': str(e)}
    
    def _parse_model_details(self, html: str) -> Dict[str, Any]:
        """Parse detailed model information from model page."""
        details = {}
        
        # Extract JSON data if available
        json_pattern = r'window\.__INITIAL_STATE__\s*=\s*({.*?});'
        json_match = re.search(json_pattern, html, re.DOTALL)
        
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                # Navigate through the JSON structure to find model details
                if 'model' in data:
                    model_data = data['model']
                    details.update({
                        'description': model_data.get('description', ''),
                        'tags': model_data.get('tags', []),
                        'pulls': model_data.get('pulls', 0),
                        'size': model_data.get('size', ''),
                        'parameters': model_data.get('parameters', ''),
                        'license': model_data.get('license', ''),
                        'modelfile': model_data.get('modelfile', ''),
                        'template': model_data.get('template', ''),
                        'capabilities': model_data.get('capabilities', [])
                    })
            except json.JSONDecodeError:
                pass
        
        # Fallback: parse HTML for basic info
        if not details:
            # Extract basic information from HTML
            desc_pattern = r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']'
            desc_match = re.search(desc_pattern, html, re.IGNORECASE)
            if desc_match:
                details['description'] = desc_match.group(1)
        
        return details
    
    def get_local_models(self) -> List[str]:
        """
        Get list of locally installed models.
        
        Returns:
            List of local model names
        """
        try:
            result = run_ollama_command(['ollama', 'list'], timeout=30)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                self._local_models = models
                return models
            else:
                print(f"Error listing local models: {result.stderr}")
                return []
        except Exception as e:
            print(f"Error getting local models: {e}")
            return []
    
    def remove_model(self, model_name: str, show_progress: bool = True) -> bool:
        """
        Remove a model from local storage.
        
        Args:
            model_name: Name of the model to remove
            show_progress: Whether to show removal progress
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Removing model: {model_name}")
        
        try:
            result = run_ollama_command(['ollama', 'rm', model_name], timeout=60, capture_output=not show_progress)
            
            success = result.returncode == 0
            
            if success:
                print(f"✓ Successfully removed {model_name}")
                # Update local models list
                self.get_local_models()
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                print(f"✗ Failed to remove {model_name}: {error_msg}")
            
            return success
            
        except Exception as e:
            print(f"Error removing {model_name}: {e}")
            return False
    
    def remove_models_batch(
        self, 
        model_names: List[str], 
        max_workers: int = 3,
        show_progress: bool = True
    ) -> Dict[str, bool]:
        """
        Remove multiple models in parallel.
        
        Args:
            model_names: List of model names to remove
            max_workers: Maximum parallel removals
            show_progress: Whether to show progress
            
        Returns:
            Dictionary mapping model names to success status
        """
        print(f"Removing {len(model_names)} models...")
        
        results = {}
        
        if show_progress:
            # Sequential removal with progress
            for model_name in model_names:
                results[model_name] = self.remove_model(model_name, show_progress=True)
        else:
            # Parallel removal without progress
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_model = {
                    executor.submit(self.remove_model, model, False): model 
                    for model in model_names
                }
                
                for future in concurrent.futures.as_completed(future_to_model):
                    model = future_to_model[future]
                    try:
                        results[model] = future.result()
                    except Exception as e:
                        print(f"Error removing {model}: {e}")
                        results[model] = False
        
        success_count = sum(results.values())
        print(f"Successfully removed {success_count}/{len(model_names)} models")
        
        return results
    
    def remove_all_models(self, confirm: bool = True) -> Dict[str, bool]:
        """
        Remove all locally installed models.
        
        Args:
            confirm: Whether to ask for confirmation
            
        Returns:
            Dictionary mapping model names to removal success status
        """
        local_models = self.get_local_models()
        
        if not local_models:
            print("No models to remove.")
            return {}
        
        print(f"Found {len(local_models)} models to remove:")
        for i, model in enumerate(local_models, 1):
            print(f"  {i:2d}. {model}")
        
        if confirm:
            response = input(f"\nAre you sure you want to remove ALL {len(local_models)} models? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("Operation cancelled.")
                return {}
        
        print(f"\nRemoving all {len(local_models)} models...")
        return self.remove_models_batch(local_models, show_progress=True)
    
    def pull_model(self, model_name: str, show_progress: bool = True) -> bool:
        """
        Pull a model from Ollama.
        
        Args:
            model_name: Name of the model to pull
            show_progress: Whether to show pull progress
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Pulling model: {model_name}")
        
        try:
            if show_progress:
                # Show real-time progress with proper encoding
                process = subprocess.Popen(
                    ['ollama', 'pull', model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    encoding='utf-8',
                    errors='replace'
                )
                
                if process.stdout:
                    for line in process.stdout:
                        print(f"  {line.rstrip()}")
                
                process.wait()
                success = process.returncode == 0
            else:
                # Silent pull using helper function
                result = run_ollama_command(['ollama', 'pull', model_name], timeout=1800)
                success = result.returncode == 0
                if not success and result.stderr:
                    print(f"Error pulling {model_name}: {result.stderr}")
            
            if success:
                print(f"✓ Successfully pulled {model_name}")
                # Update local models list
                self.get_local_models()
            else:
                print(f"✗ Failed to pull {model_name}")
            
            return success
            
        except Exception as e:
            print(f"Error pulling {model_name}: {e}")
            return False
    
    def pull_models_batch(
        self, 
        model_names: List[str], 
        max_workers: int = 3,
        show_progress: bool = True
    ) -> Dict[str, bool]:
        """
        Pull multiple models in parallel.
        
        Args:
            model_names: List of model names to pull
            max_workers: Maximum parallel downloads
            show_progress: Whether to show progress
            
        Returns:
            Dictionary mapping model names to success status
        """
        print(f"Pulling {len(model_names)} models with {max_workers} workers...")
        
        results = {}
        
        # For progress display, we'll do them sequentially if show_progress is True
        if show_progress:
            for model_name in model_names:
                results[model_name] = self.pull_model(model_name, show_progress=True)
        else:
            # Parallel execution without progress
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_model = {
                    executor.submit(self.pull_model, model, False): model 
                    for model in model_names
                }
                
                for future in concurrent.futures.as_completed(future_to_model):
                    model = future_to_model[future]
                    try:
                        results[model] = future.result()
                    except Exception as e:
                        print(f"Error pulling {model}: {e}")
                        results[model] = False
        
        success_count = sum(results.values())
        print(f"Successfully pulled {success_count}/{len(model_names)} models")
        
        return results
    
    def filter_models(
        self, 
        models: List[ModelInfo], 
        min_pulls: int = 0,
        categories: List[str] = None,
        exclude_local: bool = True,
        size_filter: str = None
    ) -> List[ModelInfo]:
        """
        Filter models based on criteria.
        
        Args:
            models: List of models to filter
            min_pulls: Minimum number of pulls
            categories: List of categories to include
            exclude_local: Whether to exclude already installed models
            size_filter: Size filter ('small', 'medium', 'large')
            
        Returns:
            Filtered list of models
        """
        filtered = models.copy()
        
        # Filter by minimum pulls
        if min_pulls > 0:
            filtered = [m for m in filtered if m.pulls >= min_pulls]
        
        # Filter by categories
        if categories:
            filtered = [m for m in filtered 
                       if any(cat.lower() in ' '.join(m.tags).lower() 
                             for cat in categories)]
        
        # Exclude local models
        if exclude_local:
            local_models = self.get_local_models()
            local_names = {m.split(':')[0] for m in local_models}
            filtered = [m for m in filtered if m.name not in local_names]
        
        # Filter by size
        if size_filter:
            size_patterns = {
                'small': r'[0-9.]+[BMG]B.*([0-9.]+B|[0-9.]+MB)',
                'medium': r'[0-9.]+GB',
                'large': r'[0-9]+[0-9]GB|[0-9]+TB'
            }
            if size_filter in size_patterns:
                pattern = size_patterns[size_filter]
                filtered = [m for m in filtered 
                           if m.size and re.search(pattern, m.size, re.IGNORECASE)]
        
        return filtered
    
    def discover_recommended_models(self) -> Dict[str, List[ModelInfo]]:
        """
        Discover recommended models by category.
        
        Returns:
            Dictionary with categories as keys and model lists as values
        """
        print("Discovering recommended models by category...")
        
        categories = {
            'general': ['llama', 'gemma', 'phi', 'qwen'],
            'code': ['codellama', 'codeqwen', 'starcoder'],
            'embedding': ['nomic-embed', 'mxbai-embed', 'all-minilm'],
            'vision': ['llava', 'bakllava', 'moondream'],
            'chat': ['orca', 'vicuna', 'solar'],
            'instruct': ['mixtral', 'wizard', 'dolphin']
        }
        
        recommended = {}
        
        for category, keywords in categories.items():
            print(f"Searching {category} models...")
            category_models = []
            
            for keyword in keywords:
                models = self.search_models(query=keyword, page_limit=2)
                category_models.extend(models)
                time.sleep(0.5)  # Be respectful
            
            # Remove duplicates and sort by popularity
            seen_names = set()
            unique_models = []
            for model in category_models:
                if model.name not in seen_names:
                    unique_models.append(model)
                    seen_names.add(model.name)
            
            unique_models.sort(key=lambda x: x.pulls, reverse=True)
            recommended[category] = unique_models[:10]  # Top 10 per category
        
        return recommended
    
    def save_model_list(self, models: List[ModelInfo], filepath: str):
        """Save model list to JSON file."""
        data = []
        for model in models:
            data.append({
                'name': model.name,
                'full_name': model.full_name,
                'description': model.description,
                'tags': model.tags,
                'pulls': model.pulls,
                'updated': model.updated,
                'size': model.size,
                'capabilities': model.capabilities,
                'url': model.url
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(models)} models to {filepath}")
    
    def load_model_list(self, filepath: str) -> List[ModelInfo]:
        """Load model list from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        models = []
        for item in data:
            models.append(ModelInfo(
                name=item['name'],
                full_name=item['full_name'],
                description=item['description'],
                tags=item['tags'],
                pulls=item['pulls'],
                updated=item['updated'],
                size=item.get('size'),
                capabilities=item.get('capabilities', []),
                url=item.get('url', '')
            ))
        
        print(f"Loaded {len(models)} models from {filepath}")
        return models


def main():
    """Main function demonstrating the model manager."""
    print("OLLAMA MODEL MANAGER")
    print("=" * 50)
    
    manager = OllamaModelManager()
    
    try:
        # Get local models
        print("\n1. Current local models:")
        local_models = manager.get_local_models()
        if local_models:
            for model in local_models:
                print(f"   ✓ {model}")
        else:
            print("   No local models found")
        
        # Search popular models
        print("\n2. Searching popular models...")
        popular_models = manager.get_popular_models(limit=20)
        
        if popular_models:
            print(f"Found {len(popular_models)} popular models:")
            for i, model in enumerate(popular_models[:10], 1):
                print(f"   {i:2d}. {model.name}")
                print(f"       {model.description}")
                print(f"       Pulls: {model.pulls:,}")
                if model.size:
                    print(f"       Size: {model.size}")
                print()
        
        # Filter out already installed models
        new_models = manager.filter_models(
            popular_models, 
            exclude_local=True,
            min_pulls=1000
        )
        
        if new_models:
            print(f"\n3. New models available for installation ({len(new_models)}):")
            for i, model in enumerate(new_models[:5], 1):
                print(f"   {i}. {model.name} - {model.description}")
            
            # Ask user if they want to install any
            choice = input(f"\nWould you like to install models? (y/n): ").lower()
            if choice.startswith('y'):
                # Show installation options
                print("\nInstallation options:")
                print("1. Install top 3 popular models")
                print("2. Install specific models")
                print("3. Install by category")
                
                option = input("Choose option (1-3): ").strip()
                
                if option == "1":
                    # Install top 3
                    models_to_install = [m.name for m in new_models[:3]]
                    print(f"Installing: {', '.join(models_to_install)}")
                    results = manager.pull_models_batch(models_to_install)
                    
                elif option == "2":
                    # Let user choose specific models
                    print("\nAvailable models:")
                    for i, model in enumerate(new_models[:10], 1):
                        print(f"   {i}. {model.name}")
                    
                    choices = input("Enter model numbers (comma-separated): ").strip()
                    try:
                        indices = [int(x.strip()) - 1 for x in choices.split(',')]
                        models_to_install = [new_models[i].name for i in indices 
                                           if 0 <= i < len(new_models)]
                        if models_to_install:
                            results = manager.pull_models_batch(models_to_install)
                    except ValueError:
                        print("Invalid input")
                
                elif option == "3":
                    # Install by category
                    print("\n4. Discovering models by category...")
                    recommended = manager.discover_recommended_models()
                    
                    print("Available categories:")
                    for i, (category, models) in enumerate(recommended.items(), 1):
                        print(f"   {i}. {category.title()} ({len(models)} models)")
                    
                    cat_choice = input("Choose category number: ").strip()
                    try:
                        cat_index = int(cat_choice) - 1
                        categories = list(recommended.keys())
                        if 0 <= cat_index < len(categories):
                            category = categories[cat_index]
                            cat_models = recommended[category]
                            
                            print(f"\nModels in {category}:")
                            for i, model in enumerate(cat_models[:5], 1):
                                print(f"   {i}. {model.name} - {model.description}")
                            
                            # Install top 2 from category
                            models_to_install = [m.name for m in cat_models[:2]]
                            if models_to_install:
                                results = manager.pull_models_batch(models_to_install)
                    except ValueError:
                        print("Invalid input")
        
        # Save discovered models
        print("\n5. Saving model information...")
        all_models = manager._available_models
        if all_models:
            manager.save_model_list(all_models, "discovered_models.json")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()