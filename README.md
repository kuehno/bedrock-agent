# Bedrock Agent

A demonstration of AWS Bedrock Agent capabilities with custom tools and chat functionality.

The aim of this repository is to show a minimal implementation of a Bedrock Agent that should be flexible in regards to the tools/functions that it can execute while keeping the dependencies to a minimum.

It does by no means aim to replace existing agent frameworks or tools like autogen, langchain, pydantic.ai etc.

However, since its light-weight by nature, it can be packaged for lambda functions or similar applications that have restrictions in size while still offering somewhat flexibility in tooling and execution.

## Prerequisites

- Python 3.8+ (tested with 3.12.3)
- AWS credentials configured
- AWS Bedrock access

## Installation

```sh
pip install -r requirements.txt
```

or

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file from `.env.template`
2. Configure AWS credentials in `.env`:
    - `AWS_ACCESS_KEY_ID`
    - `AWS_SECRET_ACCESS_KEY`
    - `AWS_REGION`

## Usage

The project includes a `demo.ipynb` notebook that demonstrates:

- Creating custom tools using the `@tool` decorator
- Initializing a BedrockAgent with model configuration
- Running chat interactions with dummy weather and poem generation capabilities

## Creating Custom Tools

To create a custom tool using the `@tool` decorator, you need to:

1. Import the decorator from [bedrock_agent.tool](bedrock_agent/tool.py):
```python
from bedrock_agent.tool import tool
```

2. Create a function with proper type hints and docstring:

The docstring must follow this format:
- First line/paragraph: Brief description of the tool
- Parameters documented with `:param <name>: <description>` format

Example tool implementation:
```python
@tool
def get_weather(city: str, country: str = "US") -> dict:
    """
    Fetches current weather information for a given city.

    :param city: Name of the city
    :param country: Two-letter country code (default: US)
    """
    # Implementation
    return {"temp": 22.5, "humidity": 65}
```

## Project Structure

```
bedrock_agent/        # Main package directory
├── agent.py          # BedrockAgent implementation
├── data.py           # Data models and configurations
├── tool.py           # Tool decorator and utilities
└── utils.py          # Helper functions
demo.ipynb            # Interactive demo notebook
```

## Dependencies

See `requirements.txt` for full list:
- boto3
- pydantic
- python-dotenv
- loguru

## TODOs

- [ ] Add example tool for fetching data from Databricks
- [ ] Implement better error handling
- [ ] Add unit tests
- [ ] Create package
- [ ] Add documentation for tool creation
- [ ] Add example for lambda function (using IaC)
