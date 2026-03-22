# LangChain Agent Implementation

This directory contains an alternative implementation of the ReviewInsight AI agent using the **LangChain framework**, allowing for direct comparison with the custom ReAct-style agent.

## Purpose

- **Compare Implementations**: Benchmark custom vs. framework-based approaches
- **Learning**: Understand trade-offs between building from scratch vs. using industry tools
- **Portfolio Value**: Demonstrate familiarity with both custom and standard approaches

## Installation

To use the LangChain agent, install additional dependencies:

```bash
pip install langchain langchain-openai
```

## File Structure

```
src/agent/
├── controller.py           # Custom ReAct agent implementation
├── langchain_agent.py      # LangChain-based agent (NEW)
└── tools.py               # Shared tool implementations

compare_agents.py          # Benchmark script (NEW)
```

## Usage

### Basic Usage

```python
from src.agent.langchain_agent import LangChainAgent

# Initialize LangChain agent
agent = LangChainAgent()

# Analyze a review
result = agent.analyze("Employee review text...")
print(result)
```

### Comparison Script

```bash
# Compare both agents on sample reviews
python compare_agents.py --reviews data/labeled/silver_standard.jsonl --samples 10

# Output saved to: data/evaluation/agent_comparison.json
```

## Architecture Comparison

| Aspect | Custom Agent | LangChain Agent |
|---------|--------------|------------------|
| **Framework** | Custom implementation | LangChain |
| **Pattern** | ReAct (Reasoning + Acting) | ReAct |
| **Tool Registry** | `ToolRegistry` class | `Tool` class |
| **Planning** | Custom `plan()` method | Built-in agent planner |
| **Execution** | Custom `execute_plan()` | `AgentExecutor` |
| **Memory** | `AgentMemory` (JSONL) | `ConversationBufferMemory` |
| **Prompt Templates** | `PromptTemplate` dataclass | `PromptTemplate` |

## Key Differences

### Custom Agent Advantages:
- ✅ Full control over implementation
- ✅ Lightweight (no extra dependencies)
- ✅ Tailored to specific use case
- ✅ Easier to debug and modify
- ✅ Faster startup (no framework overhead)

### LangChain Agent Advantages:
- ✅ Industry-standard framework
- ✅ Built-in tool ecosystem
- ✅ Standardized patterns
- ✅ Community support and documentation
- ✅ Easier integration with other LangChain tools
- ✅ Built-in memory types and persistence

## When to Use Each

**Use Custom Agent When:**
- You need maximum control and customization
- Performance is critical
- You want to minimize dependencies
- Your use case is highly specialized

**Use LangChain Agent When:**
- You need to integrate with LangChain ecosystem
- You want standardized, reusable components
- You're building a larger multi-agent system
- You need production-ready patterns and monitoring

## Performance Comparison

Run the comparison script to see actual metrics:

```bash
python compare_agents.py --samples 20
```

Expected comparisons:
- **Latency**: Custom agent typically faster (less overhead)
- **Token Usage**: Similar (same LLM calls)
- **Maintainability**: LangChain better for complex systems
- **Flexibility**: Custom agent better for specialized needs

## Resume Bullet Points

**If you implement both:**
> "Built and compared two agent implementations: custom ReAct-style framework and LangChain-based agent; analyzed trade-offs in performance, maintainability, and flexibility"

**If you only use custom:**
> "Implemented custom ReAct-style agent with tool orchestration, achieving 94.3% F1 score; chose custom implementation over LangChain for reduced latency and dependency overhead"

## Future Enhancements

Possible improvements to the LangChain implementation:

1. **Dynamic Tool Loading**: Add tools based on review content
2. **Multi-Agent Systems**: Create specialized agents for different tasks
3. **Tool Integration**: Connect to external APIs (HR systems, databases)
4. **Streaming**: Enable streaming responses for real-time feedback
5. **Evaluation Framework**: Automated comparison and benchmarking

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [ReAct Pattern](https://python.langchain.com/docs/modules/agents/agent_types/react/)
