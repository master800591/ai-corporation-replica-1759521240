"""
Ollama Toolkit - A comprehensive wrapper for ollama-python

This module provides easy-to-use functions and classes that wrap all Ollama features
including chat, generation, embeddings, model management, tools, and more.

Repository: https://github.com/ollama/ollama-python
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union, Sequence, Iterator, AsyncIterator, Callable
from pathlib import Path
import ollama
from ollama import (
    Client, AsyncClient, ChatResponse, GenerateResponse, EmbedResponse, 
    ListResponse, ShowResponse, ProcessResponse, ProgressResponse,
    StatusResponse, WebSearchResponse, WebFetchResponse, Image, Message, Tool, Options
)


class OllamaToolkit:
    """
    A comprehensive toolkit for interacting with Ollama models.
    
    This class provides easy-to-use methods for all Ollama functionality:
    - Chat and generation
    - Embeddings
    - Model management
    - Tool/function calling
    - Web search capabilities
    - Streaming support
    - Async operations
    """
    
    def __init__(self, host: str = 'http://localhost:11434', **kwargs):
        """
        Initialize the Ollama toolkit.
        
        Args:
            host: Ollama server host URL
            **kwargs: Additional arguments passed to Client (headers, timeout, etc.)
        """
        self.sync_client = Client(host=host, **kwargs)
        self.async_client = AsyncClient(host=host, **kwargs)
        self.host = host
    
    # =====================================
    # CHAT OPERATIONS
    # =====================================
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: Optional[List[Union[Dict, Tool, Callable]]] = None,
        think: Optional[Union[bool, str]] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        images: Optional[List[Union[str, bytes, Path]]] = None
    ) -> Union[ChatResponse, Iterator[ChatResponse]]:
        """
        Chat with a model.
        
        Args:
            model: Model name (e.g., 'llama3.2', 'gemma3')
            messages: List of message dictionaries with 'role' and 'content'
            stream: Whether to stream the response
            tools: List of tools/functions the model can call
            think: Enable thinking mode (True/False or 'low'/'medium'/'high')
            format: Response format ('json' or JSON schema)
            options: Model options (temperature, top_p, etc.)
            images: List of image paths or bytes for multimodal models
            
        Returns:
            ChatResponse or Iterator[ChatResponse] if streaming
            
        Example:
            >>> toolkit = OllamaToolkit()
            >>> messages = [{'role': 'user', 'content': 'Hello!'}]
            >>> response = toolkit.chat('llama3.2', messages)
            >>> print(response.message.content)
        """
        return self.sync_client.chat(
            model=model,
            messages=messages,
            stream=stream,
            tools=tools,
            think=think,
            format=format,
            options=options,
            images=images
        )
    
    async def chat_async(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        tools: Optional[List[Union[Dict, Tool, Callable]]] = None,
        think: Optional[Union[bool, str]] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        images: Optional[List[Union[str, bytes, Path]]] = None
    ) -> Union[ChatResponse, AsyncIterator[ChatResponse]]:
        """Async version of chat method."""
        return await self.async_client.chat(
            model=model,
            messages=messages,
            stream=stream,
            tools=tools,
            think=think,
            format=format,
            options=options,
            images=images
        )
    
    def chat_with_tools(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Union[Dict, Tool, Callable]],
        available_functions: Dict[str, Callable],
        max_iterations: int = 5,
        think: Optional[Union[bool, str]] = None
    ) -> ChatResponse:
        """
        Chat with automatic tool execution.
        
        Args:
            model: Model name
            messages: Conversation messages
            tools: Available tools/functions
            available_functions: Dictionary mapping function names to callables
            max_iterations: Maximum tool call iterations
            think: Enable thinking mode
            
        Returns:
            Final ChatResponse after tool execution
            
        Example:
            >>> def add_numbers(a: int, b: int) -> int:
            ...     return a + b
            >>> 
            >>> tools = [add_numbers]
            >>> functions = {'add_numbers': add_numbers}
            >>> messages = [{'role': 'user', 'content': 'What is 5 + 3?'}]
            >>> response = toolkit.chat_with_tools('llama3.2', messages, tools, functions)
        """
        current_messages = messages.copy()
        
        for _ in range(max_iterations):
            response = self.chat(model, current_messages, tools=tools, think=think)
            
            if not response.message.tool_calls:
                return response
            
            # Add the assistant's response
            current_messages.append({
                'role': 'assistant',
                'content': response.message.content or '',
                'tool_calls': [
                    {
                        'id': tool_call.id if hasattr(tool_call, 'id') else '',
                        'type': 'function',
                        'function': {
                            'name': tool_call.function.name,
                            'arguments': tool_call.function.arguments
                        }
                    }
                    for tool_call in response.message.tool_calls
                ]
            })
            
            # Execute tool calls
            for tool_call in response.message.tool_calls:
                function_name = tool_call.function.name
                if function_name in available_functions:
                    try:
                        result = available_functions[function_name](**tool_call.function.arguments)
                        current_messages.append({
                            'role': 'tool',
                            'content': str(result),
                            'tool_call_id': tool_call.id if hasattr(tool_call, 'id') else '',
                            'name': function_name
                        })
                    except Exception as e:
                        current_messages.append({
                            'role': 'tool',
                            'content': f"Error executing {function_name}: {str(e)}",
                            'tool_call_id': tool_call.id if hasattr(tool_call, 'id') else '',
                            'name': function_name
                        })
        
        # Final response without tools
        return self.chat(model, current_messages)
    
    # =====================================
    # GENERATION OPERATIONS
    # =====================================
    
    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        suffix: Optional[str] = None,
        think: Optional[Union[bool, str]] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        images: Optional[List[Union[str, bytes, Path]]] = None
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        """
        Generate text with a model.
        
        Args:
            model: Model name
            prompt: Input prompt
            stream: Whether to stream the response
            system: System prompt
            template: Prompt template
            context: Context from previous generation
            suffix: Text to append after the prompt
            think: Enable thinking mode
            format: Response format
            options: Model options
            images: Images for multimodal models
            
        Returns:
            GenerateResponse or Iterator[GenerateResponse] if streaming
        """
        return self.sync_client.generate(
            model=model,
            prompt=prompt,
            stream=stream,
            system=system,
            template=template,
            context=context,
            suffix=suffix,
            think=think,
            format=format,
            options=options,
            images=images
        )
    
    async def generate_async(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        system: Optional[str] = None,
        template: Optional[str] = None,
        context: Optional[List[int]] = None,
        suffix: Optional[str] = None,
        think: Optional[Union[bool, str]] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        images: Optional[List[Union[str, bytes, Path]]] = None
    ) -> Union[GenerateResponse, AsyncIterator[GenerateResponse]]:
        """Async version of generate method."""
        return await self.async_client.generate(
            model=model,
            prompt=prompt,
            stream=stream,
            system=system,
            template=template,
            context=context,
            suffix=suffix,
            think=think,
            format=format,
            options=options,
            images=images
        )
    
    # =====================================
    # EMBEDDING OPERATIONS
    # =====================================
    
    def embed(
        self,
        model: str,
        input: Union[str, List[str]],
        truncate: Optional[bool] = None,
        dimensions: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> EmbedResponse:
        """
        Generate embeddings for text input.
        
        Args:
            model: Embedding model name
            input: Text or list of texts to embed
            truncate: Whether to truncate input to max token length
            dimensions: Output embedding dimensions
            options: Model options
            
        Returns:
            EmbedResponse with embeddings
            
        Example:
            >>> response = toolkit.embed('nomic-embed-text', 'Hello world!')
            >>> print(response.embeddings[0][:5])  # First 5 dimensions
        """
        return self.sync_client.embed(
            model=model,
            input=input,
            truncate=truncate,
            dimensions=dimensions,
            options=options
        )
    
    async def embed_async(
        self,
        model: str,
        input: Union[str, List[str]],
        truncate: Optional[bool] = None,
        dimensions: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> EmbedResponse:
        """Async version of embed method."""
        return await self.async_client.embed(
            model=model,
            input=input,
            truncate=truncate,
            dimensions=dimensions,
            options=options
        )
    
    def embed_batch(
        self,
        model: str,
        texts: List[str],
        batch_size: int = 10,
        **kwargs
    ) -> List[List[float]]:
        """
        Embed multiple texts in batches.
        
        Args:
            model: Embedding model name
            texts: List of texts to embed
            batch_size: Number of texts per batch
            **kwargs: Additional arguments for embed method
            
        Returns:
            List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.embed(model, batch, **kwargs)
            all_embeddings.extend(response.embeddings)
        
        return all_embeddings
    
    # =====================================
    # MODEL MANAGEMENT
    # =====================================
    
    def list_models(self) -> ListResponse:
        """
        List all available models.
        
        Returns:
            ListResponse with model information
        """
        return self.sync_client.list()
    
    def show_model(self, model: str) -> ShowResponse:
        """
        Show detailed information about a model.
        
        Args:
            model: Model name
            
        Returns:
            ShowResponse with model details
        """
        return self.sync_client.show(model)
    
    def pull_model(
        self,
        model: str,
        stream: bool = False,
        insecure: bool = False
    ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
        """
        Pull a model from the registry.
        
        Args:
            model: Model name to pull
            stream: Whether to stream progress
            insecure: Allow insecure connections
            
        Returns:
            ProgressResponse or Iterator[ProgressResponse] if streaming
        """
        return self.sync_client.pull(model, stream=stream, insecure=insecure)
    
    def push_model(
        self,
        model: str,
        stream: bool = False,
        insecure: bool = False
    ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
        """
        Push a model to the registry.
        
        Args:
            model: Model name to push
            stream: Whether to stream progress
            insecure: Allow insecure connections
            
        Returns:
            ProgressResponse or Iterator[ProgressResponse] if streaming
        """
        return self.sync_client.push(model, stream=stream, insecure=insecure)
    
    def create_model(
        self,
        model: str,
        from_: Optional[str] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        license: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
        """
        Create a custom model.
        
        Args:
            model: Name for the new model
            from_: Base model to create from
            system: System prompt for the model
            template: Custom template
            license: License information
            stream: Whether to stream progress
            **kwargs: Additional creation parameters
            
        Returns:
            ProgressResponse or Iterator[ProgressResponse] if streaming
        """
        return self.sync_client.create(
            model=model,
            from_=from_,
            system=system,
            template=template,
            license=license,
            stream=stream,
            **kwargs
        )
    
    def delete_model(self, model: str) -> StatusResponse:
        """
        Delete a model.
        
        Args:
            model: Model name to delete
            
        Returns:
            StatusResponse indicating success/failure
        """
        return self.sync_client.delete(model)
    
    def copy_model(self, source: str, destination: str) -> StatusResponse:
        """
        Copy a model.
        
        Args:
            source: Source model name
            destination: Destination model name
            
        Returns:
            StatusResponse indicating success/failure
        """
        return self.sync_client.copy(source, destination)
    
    def ps(self) -> ProcessResponse:
        """
        Show running models and their resource usage.
        
        Returns:
            ProcessResponse with running model information
        """
        return self.sync_client.ps()
    
    # =====================================
    # WEB SEARCH CAPABILITIES
    # =====================================
    
    def web_search(self, query: str, max_results: int = 3) -> WebSearchResponse:
        """
        Search the web (requires Ollama API key).
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            WebSearchResponse with search results
        """
        return self.sync_client.web_search(query, max_results)
    
    def web_fetch(self, url: str) -> WebFetchResponse:
        """
        Fetch content from a URL (requires Ollama API key).
        
        Args:
            url: URL to fetch
            
        Returns:
            WebFetchResponse with page content
        """
        return self.sync_client.web_fetch(url)
    
    # =====================================
    # UTILITY METHODS
    # =====================================
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with model details
        """
        try:
            show_response = self.show_model(model)
            return {
                'name': model,
                'modified_at': show_response.modified_at,
                'size': getattr(show_response, 'size', None),
                'digest': getattr(show_response, 'digest', None),
                'details': show_response.details,
                'modelfile': show_response.modelfile,
                'parameters': show_response.parameters,
                'template': show_response.template,
                'capabilities': getattr(show_response, 'capabilities', [])
            }
        except Exception as e:
            return {'error': str(e)}
    
    def is_model_available(self, model: str) -> bool:
        """
        Check if a model is available locally.
        
        Args:
            model: Model name to check
            
        Returns:
            True if model is available, False otherwise
        """
        try:
            models = self.list_models()
            return any(m.model == model for m in models.models)
        except:
            return False
    
    def stream_to_text(self, stream_response: Iterator) -> str:
        """
        Convert a streaming response to complete text.
        
        Args:
            stream_response: Iterator from streaming method
            
        Returns:
            Complete response text
        """
        text_parts = []
        for chunk in stream_response:
            if hasattr(chunk, 'message') and chunk.message.content:
                text_parts.append(chunk.message.content)
            elif hasattr(chunk, 'response'):
                text_parts.append(chunk.response)
        return ''.join(text_parts)


class OllamaConversation:
    """
    A conversation manager for maintaining chat history with Ollama models.
    """
    
    def __init__(self, toolkit: OllamaToolkit, model: str, system_prompt: Optional[str] = None):
        """
        Initialize a conversation.
        
        Args:
            toolkit: OllamaToolkit instance
            model: Model name to use
            system_prompt: Initial system prompt
        """
        self.toolkit = toolkit
        self.model = model
        self.messages = []
        
        if system_prompt:
            self.messages.append({'role': 'system', 'content': system_prompt})
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation."""
        self.messages.append({'role': role, 'content': content})
    
    def chat(self, message: str, **kwargs) -> ChatResponse:
        """
        Send a message and get a response.
        
        Args:
            message: User message
            **kwargs: Additional arguments for chat method
            
        Returns:
            ChatResponse from the model
        """
        self.add_message('user', message)
        response = self.toolkit.chat(self.model, self.messages, **kwargs)
        
        if hasattr(response, 'message'):
            self.add_message('assistant', response.message.content)
        
        return response
    
    def clear_history(self, keep_system: bool = True) -> None:
        """
        Clear conversation history.
        
        Args:
            keep_system: Whether to keep the system prompt
        """
        if keep_system and self.messages and self.messages[0]['role'] == 'system':
            self.messages = [self.messages[0]]
        else:
            self.messages = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.messages.copy()
    
    def save_conversation(self, filepath: Union[str, Path]) -> None:
        """Save conversation to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump({
                'model': self.model,
                'messages': self.messages
            }, f, indent=2, default=str)
    
    def load_conversation(self, filepath: Union[str, Path]) -> None:
        """Load conversation from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.model = data.get('model', self.model)
            self.messages = data.get('messages', [])


# =====================================
# CONVENIENCE FUNCTIONS
# =====================================

def quick_chat(model: str, message: str, **kwargs) -> str:
    """
    Quick chat function for simple interactions.
    
    Args:
        model: Model name
        message: User message
        **kwargs: Additional arguments
        
    Returns:
        Response content as string
        
    Example:
        >>> response = quick_chat('llama3.2', 'What is the capital of France?')
        >>> print(response)
    """
    toolkit = OllamaToolkit()
    messages = [{'role': 'user', 'content': message}]
    response = toolkit.chat(model, messages, **kwargs)
    return response.message.content


def quick_generate(model: str, prompt: str, **kwargs) -> str:
    """
    Quick generation function.
    
    Args:
        model: Model name
        prompt: Generation prompt
        **kwargs: Additional arguments
        
    Returns:
        Generated text
    """
    toolkit = OllamaToolkit()
    response = toolkit.generate(model, prompt, **kwargs)
    return response.response


def quick_embed(model: str, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
    """
    Quick embedding function.
    
    Args:
        model: Embedding model name
        text: Text to embed
        **kwargs: Additional arguments
        
    Returns:
        Embedding vector(s)
    """
    toolkit = OllamaToolkit()
    response = toolkit.embed(model, text, **kwargs)
    if isinstance(text, str):
        return response.embeddings[0]
    return response.embeddings


def list_available_models() -> List[str]:
    """
    Get a list of available model names.
    
    Returns:
        List of model names
    """
    toolkit = OllamaToolkit()
    models = toolkit.list_models()
    return [model.model for model in models.models]


def model_exists(model: str) -> bool:
    """
    Check if a model exists locally.
    
    Args:
        model: Model name
        
    Returns:
        True if model exists, False otherwise
    """
    toolkit = OllamaToolkit()
    return toolkit.is_model_available(model)


# =====================================
# ASYNC CONVENIENCE FUNCTIONS
# =====================================

async def quick_chat_async(model: str, message: str, **kwargs) -> str:
    """Async version of quick_chat."""
    toolkit = OllamaToolkit()
    messages = [{'role': 'user', 'content': message}]
    response = await toolkit.chat_async(model, messages, **kwargs)
    return response.message.content


async def quick_generate_async(model: str, prompt: str, **kwargs) -> str:
    """Async version of quick_generate."""
    toolkit = OllamaToolkit()
    response = await toolkit.generate_async(model, prompt, **kwargs)
    return response.response


async def quick_embed_async(model: str, text: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
    """Async version of quick_embed."""
    toolkit = OllamaToolkit()
    response = await toolkit.embed_async(model, text, **kwargs)
    if isinstance(text, str):
        return response.embeddings[0]
    return response.embeddings


# =====================================
# TOOL CREATION HELPERS
# =====================================

def create_function_tool(func: Callable) -> Dict[str, Any]:
    """
    Create a tool definition from a Python function.
    
    Args:
        func: Python function with proper docstring
        
    Returns:
        Tool definition dictionary
    """
    # This leverages ollama's built-in function conversion
    # The ollama library automatically converts functions to tools
    return func


def create_custom_tool(
    name: str,
    description: str,
    parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a custom tool definition.
    
    Args:
        name: Tool name
        description: Tool description
        parameters: Parameter schema
        
    Returns:
        Tool definition dictionary
    """
    return {
        'type': 'function',
        'function': {
            'name': name,
            'description': description,
            'parameters': parameters
        }
    }


# =====================================
# EXAMPLE USAGE
# =====================================

if __name__ == "__main__":
    # Example usage of the toolkit
    
    # Initialize toolkit
    toolkit = OllamaToolkit()
    
    # Check available models
    print("Available models:")
    for model in list_available_models():
        print(f"  - {model}")
    
    # Quick chat example
    try:
        response = quick_chat('llama3.2', 'What is Python?')
        print(f"\nQuick chat response: {response}")
    except Exception as e:
        print(f"Error in quick chat: {e}")
    
    # Conversation example
    try:
        conversation = OllamaConversation(toolkit, 'llama3.2', 'You are a helpful assistant.')
        response = conversation.chat('Hello!')
        print(f"\nConversation response: {response.message.content}")
    except Exception as e:
        print(f"Error in conversation: {e}")
    
    # Embedding example
    try:
        if model_exists('nomic-embed-text'):
            embeddings = quick_embed('nomic-embed-text', 'Hello world!')
            print(f"\nEmbedding dimensions: {len(embeddings)}")
        else:
            print("\nEmbedding model not available")
    except Exception as e:
        print(f"Error in embedding: {e}")