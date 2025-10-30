# app/agent.py

import os
from typing import AsyncGenerator, Literal, List, Optional
import yaml

from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama

from api.tools.contextual_retrieval_tool import ContextualRetrievalTool


class WarAndPeaceAgent:
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3
    ):
        self.llm_model = os.getenv('AGENT_LLM_MODEL', 'qwen3:14b')
        self.ollama_base_url = os.getenv(
            'OLLAMA_BASE_URL', 'http://localhost:11434')
        self.temperature = temperature

        self.system_prompt = system_prompt or WarAndPeaceAgent._get_promt()

        self.tools = [ContextualRetrievalTool()]
        llm = ChatOllama(
            model=self.llm_model,
            base_url=self.ollama_base_url,
            temperature=self.temperature,
        )
        self.llm_with_tools = llm.bind_tools(self.tools)

        self.graph = self._build_graph()

    @staticmethod
    def _get_promt() -> str:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'promt.yaml')
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)['promt']
        return ''

    def _build_graph(self):
        class AgentState(MessagesState):
            pass

        def agent_node(
                state: AgentState, config: RunnableConfig) -> AgentState:
            messages = state['messages']
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages = (
                    [SystemMessage(content=self.system_prompt)] + messages)

            response = self.llm_with_tools.invoke(messages, config)
            return {'messages': messages + [response]}

        def should_continue(state: AgentState) -> Literal['tools', '__end__']:
            last_message = state['messages'][-1]
            if (hasattr(last_message, 'tool_calls') and
                    len(last_message.tool_calls) > 0):
                return 'tools'
            return '__end__'

        workflow = StateGraph(AgentState)
        workflow.add_node('agent', agent_node)
        workflow.add_node('tools', ToolNode(tools=self.tools))

        workflow.set_entry_point('agent')
        workflow.add_conditional_edges(
            'agent',
            should_continue,
            {'tools': 'tools', '__end__': '__end__'}
        )
        workflow.add_edge('tools', 'agent')

        return workflow.compile()

    def invoke(
            self,
            query: str,
            chat_history: List[BaseMessage] | None = None
    ) -> str:
        messages = chat_history or []
        messages.append(HumanMessage(content=query))

        result = self.graph.invoke({'messages': messages})
        final_message = result['messages'][-1]

        if (hasattr(final_message, 'content') and
                isinstance(final_message.content, str)):
            return final_message.content
        return str(final_message)

    async def astream_answer(
        self,
        query: str,
        chat_history: List[BaseMessage] | None = None
    ) -> AsyncGenerator[str, None]:
        messages = chat_history or []
        messages.append(HumanMessage(content=query))

        async for event in self.graph.astream_events({'messages': messages}):
            if (event['event'] == 'on_chat_model_stream' and
                    'agent' in event.get(
                        'metadata', {}).get('langgraph_node', '')):
                chunk = event['data'].get('chunk')
                if (chunk and
                        hasattr(chunk, 'content') and
                        isinstance(chunk.content, str)):
                    yield chunk.content
