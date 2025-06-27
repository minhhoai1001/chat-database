import operator
from typing_extensions import TypedDict
from typing import List, Annotated

class GraphState(TypedDict):
    """
    Graph state is a dictionary that contains information we want to propagate to, and modify in, each graph node.
    """
    question: str  # User question
    sql_query: str  # SQL query
    sql_retriever: str  # Result from SQL query
    answer: str  # Answer to the user question
    max_retries: int  # Max number of retries for answer generation
    loop_step: Annotated[int, operator.add]
    documents: List[str]  # List of retrieved documents
    history: List[str]  # List of messages in the conversation