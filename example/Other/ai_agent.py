import asyncio
import json
import os
from fastmcp import Client
from openai import OpenAI

os.environ['NO_PROXY'] = os.environ.get('NO_PROXY', '') + ',0.0.0.0'

OPENROUTER_API_KEY = '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'
LLM_URL = 'https://openrouter.ai/api/v1'
MCP_URL = 'http://localhost:8787/mcp'

llm_client = OpenAI(
  base_url=LLM_URL,
  api_key=OPENROUTER_API_KEY,
)

mcp_client = Client(MCP_URL)

async def main():
    async with mcp_client:
        mcp_tools = await mcp_client.list_tools()
        tools = [
            {
                'type': 'function',
                'function': {
                    'name': t.name,
                    'description': t.description,
                    'parameters': t.inputSchema
                }
            }
            for t in mcp_tools
        ]

        messages = [
            { 'role': 'user', 'content': '請問15天前是幾月幾號？' }
        ]

        response = llm_client.chat.completions.create(
            model='stepfun/step-3.5-flash:free',
            messages=messages,
            extra_body={
                'plugins': [], 'reasoning': {'enabled': True}, 'tools': tools  #{'id': 'web'} 
            }
        )

        msg = response.choices[0].message

        while msg.tool_calls:
            print('模型選擇調用工具')
            tool_call = msg.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            result = await mcp_client.call_tool(
                tool_call.function.name,
                args
            )
            # 把模型的第一階段的結果丟進 messages 中
            messages.append(msg)

            # 把呼叫 tool 的結果丟進 messages 中
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': result.content[0].text
                }
            )

            # 再次詢問模型
            response = llm_client.chat.completions.create(
                model='stepfun/step-3.5-flash:free',
                messages=messages,
                extra_body={
                    'plugins': [], 'reasoning': {'enabled': True}
                }
            )

            msg = response.choices[0].message


        if msg.tool_calls:
            print('模型選擇調用工具')
            tool_call = msg.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            result = await mcp_client.call_tool(
                tool_call.function.name,
                args
            )
            # 把模型的第一階段的結果丟進 messages 中
            messages.append(msg)

            # 把呼叫 tool 的結果丟進 messages 中
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': result.content[0].text
                }
            )

            # 再次詢問模型
            response = llm_client.chat.completions.create(
                model='stepfun/step-3.5-flash:free',
                messages=messages,
                extra_body={
                    'reasoning': {'enabled': True}, 'tools': tools, # 'plugins': [{'id': 'web'}] 
                }
            )

            msg = response.choices[0].message

        # 直接回應
        print(response.choices[0].message.content)



if __name__ == '__main__':
    asyncio.run(main())