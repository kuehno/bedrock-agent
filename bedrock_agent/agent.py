from bedrock_agent.data import ConverseResponse, ToolUse, ToolConfig, ModelConfig
from typing import List
import boto3
import json
from loguru import logger


class BedrockAgent():
    def __init__(self, session: boto3.Session, model_config: ModelConfig = None, messages: list = [], tools: list = [], verbose: bool = False):
        self.verbose = verbose
        self.bedrock_runtime = session.client("bedrock-runtime", region_name="us-east-1")
        self.model_config = model_config if model_config else ModelConfig()
        self.tools = tools
        self.messages = messages
        self.tool_map = {}
        if tools:
            for tool in tools:
                self.register_tool(tool)

    @staticmethod
    def get_message(content: str = "Who are you?", role: str = "user"):
        assert role in ["user", "assistant"], "Role must be either 'user' or 'assistant'"
        return {
            "role": role,
            "content": [{"text": content}]
        }
        
    @staticmethod
    def get_tool_result(tool_use: ToolUse, result: str | dict):  
        tool_result = {
            "toolUseId": tool_use.toolUseId,
            "content": [{"json": result}] if isinstance(result, dict) else [{"text": result}]
        }            
        return tool_result
        
    def reset(self):
        self.messages.clear()
        
    def register_tool(self, tool_func):
        self.tool_map[tool_func.__name__] = tool_func
        
    def try_call_tool(self, tool_use: ToolUse):
        if tool_use.name not in self.tool_map:
            raise KeyError(f"Tool '{tool_use.name}' not registered")
        result = self.tool_map[tool_use.name](**tool_use.input)
        assert isinstance(result, str) or isinstance(result, dict), "Tool must return a string or a dictionary"
        return result
        
    def validate_message(self, message: dict):
        if not isinstance(message, dict):
            return False
        if "role" not in message or "content" not in message:
            return False
        if not isinstance(message["content"], list):
            return False
        if not isinstance(message["content"][0], dict) or "text" not in message["content"][0] or "toolResult" not in message["content"][0]:
            return False
        if message["role"] not in ["user", "assistant"]:
            return False
        return True
    
    def chat(self, message: str | dict | List[str], reset: bool = False, *args, **kwargs):
        if reset: self.reset()
        stop_reason = None
        
        if self.tools:
            kwargs = {**kwargs, **{"toolConfig": ToolConfig(tools=[tool.to_json() for tool in self.tools]).to_json()}}
        
        response = self.completion(message, *args, **kwargs)
        
        while stop_reason != "end_turn":           
            stop_reason = response.stopReason
            if stop_reason == "end_turn":
                message = response.output.message.to_json()
                self.messages.append(message)
                logger.info("End of conversation. Returning response.")
                return response
            
            if stop_reason == "tool_use":
                message = response.output.message.to_json()
                self.messages.append(message)
                
                tool_result_message = {"role": "user", "content": []}
                for content in response.output.message.content:
                    if content.toolUse:
                        result = self.try_call_tool(content.toolUse)
                        tool_result = self.get_tool_result(content.toolUse, result)
                        tool_result_message["content"].append({"toolResult": tool_result})
                        if self.verbose:
                            logger.info(json.dumps(message, indent=4))
                self.messages.append(tool_result_message)
                
            response = self.completion(self.messages, *args, **kwargs)
        return response
    
    def completion(self, message: str | dict | list[dict[str, any]], *args, **kwargs):
        if isinstance(message, str):
            message = self.get_message(message, role="user")
            self.messages.append(message)
        elif isinstance(message, dict):
            if self.validate_message(message):
                message = message
                self.messages.append(message)
            else:
                raise ValueError("Invalid message")
        elif isinstance(message, list):
            if all([self.validate_message(m) for m in message]):
                message = message
                self.messages.extend(message)
        else:
            raise ValueError("Invalid message")
                        
        if self.verbose and len(self.messages) == 1:
            logger.info(json.dumps(self.messages[0], indent=4))
        
        response = ConverseResponse(**self.bedrock_runtime.converse(
                modelId=self.model_config.model_id,
                messages=self.messages,
                inferenceConfig=self.model_config.params,
                *args,
                **kwargs
        ))
        if self.verbose:
            logger.info(json.dumps(response.output.message.to_json(), indent=4))
        return response
        
    def draw_trace(self):
        for i, message in enumerate(self.messages, 1):
            print(f"\n{'-'*50}")
            print(f"Message # {i}")
            print(f"{'-'*50}")
            
            role = message["role"].upper()
            print(f"ROLE: {role}")
            
            for content_item in message["content"]:
                if "text" in content_item:
                    print(f"TEXT: {content_item['text']}")
                elif "toolUse" in content_item:
                    tool = content_item["toolUse"]
                    print(f"TOOL USE: {tool['name']}")
                    print(f"INPUT: {tool['input']}")
                elif "toolResult" in content_item:
                    result = content_item["toolResult"]
                    print(f"TOOL RESULT: {result['content'][0]['text']}")
