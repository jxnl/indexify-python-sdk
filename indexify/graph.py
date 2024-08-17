from .extractor_sdk import extractor, Extractor

from typing import Type, Union
from pydantic import BaseModel

from .run_graph import RunGraph
from .local_runner import LocalRunner


def Graph(
    name: str,
    input: Type[BaseModel],
    start_node: Union[extractor, Extractor],
    run_local: bool,
) -> RunGraph:

    if run_local:
        runner = LocalRunner()
    else:
        raise NotImplementedError("Remote runner not supported yet")

    graph = RunGraph(name=name, input=input, start_node=start_node, runner=runner)
    return graph
