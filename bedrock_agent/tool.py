import inspect
from typing import Any, Dict, Callable
from pydantic import BaseModel
from functools import wraps

class Tool(BaseModel):
    name: str
    description: str
    properties: Dict[str, Any]
    func: Callable

def parse_docstring(docstring: str) -> Dict[str, str]:
    param_descriptions = {}
    if (docstring):
        lines = docstring.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(':param'):
                parts = line.split(':')
                if len(parts) >= 3:
                    param_name = parts[1].strip().split(' ')[1]
                    param_description = parts[2].strip()
                    param_descriptions[param_name] = param_description
    return param_descriptions

type_mapping = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    Any: "any"
}

def generate_properties(func: callable) -> Dict[str, Any]:
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func)
    param_descriptions = parse_docstring(docstring)
    properties = {}
    for param in signature.parameters.values():
        param_type = type_mapping.get(param.annotation, "unknown")
        properties[param.name] = {
            'type': param_type,
            'description': param_descriptions.get(param.name, '')
        }
    return properties

def tool(func: callable) -> Tool:
    name = func.__name__
    properties = generate_properties(func)
    description = (inspect.getdoc(func) or '').split('\n\n')[0]
    tool = Tool(name=name, properties=properties, description=description, func=func)
    
    def to_json() -> Dict[str, Any]:
        return {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "json": {
                        'type': 'object',
                        'properties': tool.properties,
                        'required': list(tool.properties.keys())
                    }
                }
            }
        }
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    wrapper.tool = tool
    wrapper.to_json = to_json
    return wrapper
