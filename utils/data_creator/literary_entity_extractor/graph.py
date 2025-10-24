from langgraph.graph import StateGraph

from .characters_node import CharactersNode
from .locations_node import LocationsNode
from .states import CreatorState
from .summary_node import SummaryNode


class LiteraryEntityExtractor():
    def __init__(self, llm):
        self.workflow = StateGraph(CreatorState)

        self.workflow.add_node('characters_node', CharactersNode(llm=llm).node)
        self.workflow.add_node('locations_node', LocationsNode(llm=llm).node)
        self.workflow.add_node('summary_node', SummaryNode(llm=llm).node)

        self.workflow.set_entry_point('characters_node')
        self.workflow.add_edge('characters_node', 'locations_node')
        self.workflow.add_edge('locations_node', 'summary_node')
        self.workflow.set_finish_point('summary_node')

        self.graph = self.workflow.compile()

    def print_graph(self) -> None:
        self.graph.get_graph().print_ascii()

    def get_graph_ascii(self) -> None:
        self.graph.get_graph().draw_ascii()

    def invoke(self, message: str):
        return self.graph.invoke(CreatorState.create(message))
