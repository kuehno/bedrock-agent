from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json


class PriceType(BaseModel):
        input: float
        output: float

class PriceCategory(BaseModel):
    on_demand: PriceType
    batch: PriceType
    
class PricingInfo(BaseModel):
    model_pricing: Dict[str, PriceCategory] = {}
    
    @classmethod
    def from_json(cls, file_path: str) -> "PricingInfo":
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        pricing_info = cls()
        for model_id, prices in data.items():
            pricing_info.model_pricing[model_id] = PriceCategory(
                on_demand=PriceType(**prices['on_demand']),
                batch=PriceType(**prices['batch'])
            )
        return pricing_info

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
    text: Optional[str] = None
    toolUse: Optional[ToolUse] = None

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
