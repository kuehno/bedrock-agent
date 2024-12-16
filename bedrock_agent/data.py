from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class ModelConfig(BaseModel):
    model_id: str = "amazon.nova-micro-v1:0"
    temperature: float = 0.9
    top_p: float = 0.8
    max_tokens: int = 1024
    
    @property
    def params(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "topP": self.top_p,
            "maxTokens": self.max_tokens
        }
        
class ToolConfig(BaseModel):
    tools: List[Dict[str, Any]]
    toolChoice: Dict[str, Any] = {"auto": {}}
    
    def to_json(self) -> Dict[str, Any]:
        return {
            "tools": self.tools,
            "toolChoice": self.toolChoice
        }

class ToolUse(BaseModel):
    toolUseId: str
    name: str
    input: Dict[str, Any]
        
    def to_json(self) -> Dict[str, Any]:
        return {
            "toolUseId": self.toolUseId,
            "name": self.name,
            "input": self.input
        }

class Content(BaseModel):
    toolUse: Optional[ToolUse] = None
    text: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        if self.toolUse:
            return {
                "toolUse": self.toolUse.to_json()
            }
        elif self.toolUse and self.text:
            return {
                "toolUse": self.toolUse.to_json(),
                "text": self.text
            }
        else:
            return {
                "text": self.text
            }

class Message(BaseModel):
    role: str
    content: List[Content]
    
    def to_json(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": [content.to_json() for content in self.content]
        }

class Output(BaseModel):
    message: Message

class ResponseMetadata(BaseModel):
    RequestId: str
    HTTPStatusCode: int
    HTTPHeaders: Dict[str, str]
    RetryAttempts: int

class Usage(BaseModel):
    inputTokens: int
    outputTokens: int
    totalTokens: int

class Metrics(BaseModel):
    latencyMs: int

class ConverseResponse(BaseModel):
    ResponseMetadata: ResponseMetadata
    output: Output
    stopReason: str
    usage: Usage
    metrics: Metrics
