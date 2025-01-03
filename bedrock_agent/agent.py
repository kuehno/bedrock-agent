from bedrock_agent.types import ConverseResponse, ToolUse, ToolConfig, ModelConfig, PricingInfo
from bedrock_agent.utils import get_boto_session
from typing import List, Optional
from pydantic import BaseModel, ConfigDict
import uuid
from loguru import logger
import os
import boto3
import json


# TODO: Move some functions and vars from BedrockAgent to AgentBase
class AgentBase():    
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

class ToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    value: str | dict = ""
    agent: Optional[AgentBase] = None

class BedrockAgent(AgentBase):
    def __init__(self, name: str = None, session: boto3.Session = None, model_cfg: ModelConfig = None, messages: list = None, tools: list = None, system_message: str = "You are a helpful AI assistant.", verbose: bool = False):
        self.name = name or "BedrockAgent"
        self.verbose = verbose
        self.session = session if session else get_boto_session()
        self.bedrock_runtime = self.session.client("bedrock-runtime")
        self.model_cfg = model_cfg if model_cfg else ModelConfig()
        self.pricing_info = PricingInfo.from_json(os.path.join(os.path.dirname(__file__), "pricing_info.json"))
        self.tokens_used = {"prompt_tokens": 0, "completion_tokens": 0}
        self.tools = tools or []
        self.messages = messages or []
        self.responses = []
        self.tool_map = {}
        self.system_message = system_message
        self.handoff_agents_trace = {}
        if tools:
            for tool in tools:
                if callable(tool):
                    self.register_tool(tool)
    
    @property
    def total_tokens_used(self):
        return self.tokens_used["prompt_tokens"] + self.tokens_used["completion_tokens"]
                
    @property
    def total_costs(self):
        total_costs = 0
        model_pricing = self.pricing_info.model_pricing[self.model_cfg.model_id]
        total_costs += model_pricing.on_demand.input * (self.tokens_used["prompt_tokens"] / 1000)
        total_costs += model_pricing.on_demand.output * (self.tokens_used["completion_tokens"] / 1000)
        return '{:.10f}'.format(total_costs)
    
    def add_handoff_agent_tokens(self, agent: AgentBase):
        self.tokens_used["prompt_tokens"] += agent.tokens_used["prompt_tokens"]
        self.tokens_used["completion_tokens"] += agent.tokens_used["completion_tokens"]
                    
    def reset(self):
        self.messages.clear()
        
    def register_tool(self, tool_func):
        tool_name = tool_func.__name__
        if self.verbose:
            logger.info(f"{self.system_message}, Registering tool: {tool_name}")
        self.tool_map[tool_name] = tool_func
        
    def get_system_message(self):
        return [{"text": self.system_message}]
        
    def try_call_tool(self, tool_use: ToolUse):
        if tool_use.name not in self.tool_map:
            raise KeyError(f"Tool '{tool_use.name}' not registered")
        if self.verbose:
            logger.info(f"Calling tool: {tool_use.name} with input: {tool_use.input}")
        result = self.tool_map[tool_use.name](**tool_use.input)
        assert isinstance(result, str) or isinstance(result, dict) or isinstance(result, list) or isinstance(result, BedrockAgent), "Tool must return a string, dictionary, list or BedrockAgent object"
        return result
    
    def handle_tool_result(self, tool_result: str | dict) -> ToolResult:
        match tool_result:
            case ToolResult() as tool_result:
                return tool_result
            case BedrockAgent() as agent:
                return ToolResult(
                    value=str(tool_result),
                    agent=agent
                    )
            case _:
                try:
                    return ToolResult(value=str(tool_result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {tool_result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    logger.error(error_message)
                    raise TypeError(error_message)
        
    def validate_message(self, message: dict):
        if not isinstance(message, dict):
            if self.verbose:
                logger.error("Invalid message. Must be a dictionary")
            return False
        if "role" not in message or "content" not in message:
            if self.verbose:
                logger.error("Invalid message. Must have 'role' and 'content' keys")
            return False
        if not isinstance(message["content"], list):
            if self.verbose:
                logger.error("Invalid message. 'content' must be a list")
            return False
        if not isinstance(message["content"][0], dict):
            if self.verbose:
                logger.error("Invalid message. 'content' must have 'text' or 'toolResult' keys")
            return False
        if message["role"] not in ["user", "assistant"]:
            if self.verbose:
                logger.error("Invalid message. 'role' must be 'user' or 'assistant'")
            return False
        return True
    
    def format_message(self, message: str | dict | List[str], role: str = "user"):
        if isinstance(message, str):
            message = self.get_message(message, role=role)
            self.messages.append(message)
        elif isinstance(message, dict):
            if self.validate_message(message):
                self.messages.append(message)
            else:
                raise ValueError("Invalid message")
        elif isinstance(message, list):
            if all([self.validate_message(m) for m in message]):
                self.messages.extend(message)
        else:
            raise ValueError("Invalid message")
    
    def chat(self, message: str | dict | List[str], role: str = "user", *args, **kwargs):
        stop_reason = None
        done = False
        handoff_agent = None
        
        if self.tools:
            kwargs = {**kwargs, **{"toolConfig": ToolConfig(tools=[tool.to_json() for tool in self.tools]).to_json()}}
            
        response = self.completion(message, role, *args, **kwargs)
        stop_reason = response.stopReason
        
        # TODO: Refactor the while not done loop to not run indefinitely but rather a fixes number of iterations
        while not done: 
            if stop_reason == "end_turn":
                message = response.output.message.to_json()
                self.messages.append(message)
                done = True
                logger.info(f"Agent: {self.name} | End of conversation. Returning response.")
                return response
            
            if stop_reason == "tool_use":
                self.messages.append(response.output.message.to_json())
                tool_result_message = {"role": "user", "content": []}
                
                # TODO: Use asyncio to call tools asynchronously and fetch the final results for parallel processing
                for content in response.output.message.content:
                    if content.toolUse:
                        raw_tool_result = self.try_call_tool(content.toolUse)
                        tool_result = self.handle_tool_result(raw_tool_result)
                        
                        handoff_agent = tool_result.agent
                        if handoff_agent:
                            # TODO: Modify input from handoff agent such that .get("request") is not needed. Might lead to errors in the current implementation.
                            handoff_response = handoff_agent.chat(content.toolUse.input.get("request"))
                            
                            # add handoff_agents trace to self with unique id -> in case multiple agents are used througoht the whole conversation
                            self.handoff_agents_trace[f"{handoff_agent.name}-{str(uuid.uuid4())[:4]}"] = handoff_agent.messages
                            
                            # add handoff_agents tokens to self
                            self.add_handoff_agent_tokens(handoff_agent)
                            
                            tool_result.value = self.get_tool_result(content.toolUse, handoff_response.output.message.content[0].text)
                        else:
                            tool_result.value = self.get_tool_result(content.toolUse, tool_result.value)
                        tool_result_message["content"].append({"toolResult": tool_result.value})
                        
                        if self.verbose:
                            logger.info(json.dumps(tool_result_message, indent=4))
                response = self.completion(tool_result_message, role, *args, **kwargs)
                stop_reason = response.stopReason
                    
    def completion(self, message: str | dict | list[dict[str, any]], role: str = "user", *args, **kwargs):
        self.format_message(message, role)
                        
        if self.verbose:
            logger.info(f"Calling converse API with: {json.dumps(self.messages, indent=4)}")
        
        response = ConverseResponse(**self.bedrock_runtime.converse(
                modelId=self.model_cfg.model_id,
                messages=self.messages,
                inferenceConfig=self.model_cfg.params,
                system=self.get_system_message(),
                *args,
                **kwargs
        ))
        self.responses.append(response)
        self.tokens_used["prompt_tokens"] += response.usage.inputTokens
        self.tokens_used["completion_tokens"] += response.usage.outputTokens
        if self.verbose:
            logger.info(json.dumps(response.output.message.to_json(), indent=4))
        return response
    
    @staticmethod
    def draw_messages(messages):
        messages = messages.copy()
        for i, message in enumerate(messages, 1):
            print(f"\n{'-'*50}")
            print(f"Message # {i}")
            print(f"{'-'*50}")
            
            role = message["role"].upper()
            print(f"ROLE: {role}")
            
            for content_item in message["content"]:
                if "text" in content_item:
                    print(f"TEXT: {content_item['text']}")
                if "toolUse" in content_item:
                    tool = content_item["toolUse"]
                    print(f"TOOL USE: {tool['name']}")
                    print(f"INPUT: {tool['input']}")
                if "toolResult" in content_item:
                    result = content_item["toolResult"]
                    print(f"TOOL RESULT: {result['content'][0]['text']}")
                    
    def draw_trace(self):
        self.draw_messages(self.messages)
                    
    def draw_handoff_agents_traces(self):
        for agent_name, messages in self.handoff_agents_trace.items():
            print(f"\n{'='*50}")
            print(f"Handoff Agent: {agent_name}")
            print(f"{'='*50}")
            
            self.draw_messages(messages)
