# Using Tools with Ollama Shinchan Model

## Important Note
Ollama Modelfiles don't support `TOOL` definitions directly. Tools must be passed at runtime via the API.

## How to Use search_query Tool

When calling Ollama via API, you need to pass the tool definition in the request:

### Python Example:

```python
import ollama

# Define the tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_query",
            "description": "Searches for information on the web. Use this when you need current information, facts, or answers to questions that require looking up data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string to look up information"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Call with tools
response = ollama.chat(
    model='shinchan-smollm2',
    messages=[
        {
            'role': 'user',
            'content': 'What is the weather today?'
        }
    ],
    tools=tools
)

# Handle tool calls
if response.get('message', {}).get('tool_calls'):
    for tool_call in response['message']['tool_calls']:
        if tool_call['function']['name'] == 'search_query':
            query = tool_call['function']['arguments']['query']
            # Execute your search function
            results = your_search_function(query)
            # Continue conversation with results
            response = ollama.chat(
                model='shinchan-smollm2',
                messages=[
                    {'role': 'user', 'content': 'What is the weather today?'},
                    {'role': 'assistant', 'content': response['message']['content']},
                    {'role': 'tool', 'name': 'search_query', 'content': str(results)}
                ],
                tools=tools
            )
```

### JavaScript/TypeScript Example:

```typescript
import { Ollama } from 'ollama';

const ollama = new Ollama();

const tools = [
  {
    type: "function",
    function: {
      name: "search_query",
      description: "Searches for information on the web.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The search query string"
          }
        },
        required: ["query"]
      }
    }
  }
];

const response = await ollama.chat({
  model: 'shinchan-smollm2',
  messages: [
    { role: 'user', content: 'What is the weather today?' }
  ],
  tools: tools
});
```

## Tool Definition JSON

```json
{
  "type": "function",
  "function": {
    "name": "search_query",
    "description": "Searches for information on the web. Use this when you need current information, facts, or answers to questions that require looking up data.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query string to look up information"
        }
      },
      "required": ["query"]
    }
  }
}
```

