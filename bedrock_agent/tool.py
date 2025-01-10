import inspect
from typing import Any, Dict, Callable
from pydantic import BaseModel
from functools import wraps


class Tool(BaseModel):
    name: str
    description: str
    properties: Dict[str, Any]
    func: Callable

type_mapping = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null"
}

def tool(func: callable) -> Tool:
    name = func.__name__
    signature = inspect.signature(func)
    
    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_mapping.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}
    
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]
    
    description = (inspect.getdoc(func) or '')
    tool = Tool(name=name, properties=parameters, description=description, func=func)
    
    def to_json() -> Dict[str, Any]:
        return {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {
                    "json": {
                        'type': 'object',
                        'properties': parameters,
                        'required': required
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
