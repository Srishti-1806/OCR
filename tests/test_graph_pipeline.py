from app.graph.builder import build_extraction_graph


def test_build_extraction_graph_returns_compiled_graph():
    graph = build_extraction_graph()
    assert graph is not None
    assert hasattr(graph, "invoke")
