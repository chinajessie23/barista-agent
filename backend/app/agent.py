"""
Barista Agent - Refactored for web deployment.

Changes from POC:
- Removed human_node (frontend handles input)
- Removed input()/print() calls
- Added session management via checkpointer
- Agent processes one turn at a time
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env from backend directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


# =============================================================================
# State
# =============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    order: list[str]
    finished: bool


# =============================================================================
# Menu Data
# =============================================================================

MENU = """
DRINKS:
- Espresso: $3.00
- Americano: $3.50
- Latte: $4.50
- Cappuccino: $4.50
- Mocha: $5.00
- Cold Brew: $4.00

FOOD:
- Croissant: $3.50
- Muffin: $3.00
- Bagel: $3.50
- Cookie: $2.50

MODIFIERS (add to any drink):
- Oat milk: +$0.75
- Almond milk: +$0.75
- Extra shot: +$0.50
- Vanilla syrup: +$0.50
"""

PRICES = {
    "espresso": 3.00,
    "americano": 3.50,
    "latte": 4.50,
    "cappuccino": 4.50,
    "mocha": 5.00,
    "cold brew": 4.00,
    "croissant": 3.50,
    "muffin": 3.00,
    "bagel": 3.50,
    "cookie": 2.50,
}

MODIFIER_PRICES = {
    "oat milk": 0.75,
    "almond milk": 0.75,
    "extra shot": 0.50,
    "vanilla syrup": 0.50,
    "vanilla": 0.50,
}


# =============================================================================
# Tools
# =============================================================================

@tool
def get_menu() -> str:
    """Get the coffee shop menu with drinks, food, and modifiers."""
    return MENU


@tool
def add_to_order(item: str) -> str:
    """Add an item to the customer's order.

    Args:
        item: The item to add (e.g., "Latte with oat milk")
    """
    return f"Added {item} to order"


@tool
def get_order() -> str:
    """Get the current order."""
    return "Current order"


@tool
def confirm_order() -> str:
    """Show the order to the customer and ask for confirmation before placing."""
    return "Order confirmation"


@tool
def place_order() -> str:
    """Place the final order after customer confirms."""
    return "Order placed"


@tool
def clear_order() -> str:
    """Clear all items from the current order."""
    return "Order cleared"


@tool
def calculate_total() -> str:
    """Calculate the total price of the current order."""
    return "Total calculated"


STATELESS_TOOLS = [get_menu]
STATEFUL_TOOLS = [add_to_order, get_order, confirm_order, place_order, clear_order, calculate_total]
ALL_TOOLS = STATELESS_TOOLS + STATEFUL_TOOLS


# =============================================================================
# LLM
# =============================================================================

SYSTEM_PROMPT = """You are a friendly barista at a coffee shop.

Your job:
1. Greet customers warmly
2. Help them with their order
3. Use get_menu() when they ask what's available
4. Use add_to_order() for each item they want
5. When they're done ordering, use confirm_order() to show them their order
6. Wait for customer to say "yes" or confirm before using place_order()
7. Use calculate_total() to show price breakdown

Be conversational, helpful, and concise. Don't overwhelm the customer with too much text.
"""


# Singleton LLM instance - created once at module load
_llm_with_tools = None


def get_llm():
    """Get singleton LLM instance with tools bound."""
    global _llm_with_tools
    if _llm_with_tools is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            transport="rest",
            timeout=30,  # 30 second timeout for API calls
        )
        _llm_with_tools = llm.bind_tools(ALL_TOOLS)
    return _llm_with_tools


# =============================================================================
# Nodes
# =============================================================================

def barista_node(state: State) -> State:
    """LLM generates a response, possibly with tool calls."""
    messages = list(state["messages"])
    messages_to_send = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    llm_with_tools = get_llm()
    response = llm_with_tools.invoke(messages_to_send)
    return {"messages": [response]}


def order_node(state: State) -> State:
    """Handle stateful order tools."""
    last_msg = state["messages"][-1]
    order = list(state.get("order", []))
    outbound_msgs = []
    finished = False

    for tool_call in last_msg.tool_calls:
        tool_name = tool_call["name"]

        if tool_name == "add_to_order":
            item = tool_call["args"]["item"]
            order.append(item)
            response = f"Added '{item}' to your order."

        elif tool_name == "get_order":
            if order:
                response = "Current order:\n" + "\n".join(f"  - {item}" for item in order)
            else:
                response = "Your order is empty."

        elif tool_name == "confirm_order":
            if order:
                order_list = "\n".join(f"  - {item}" for item in order)
                # Calculate total for confirmation
                total = calculate_order_total(order)
                response = f"Here's your order:\n{order_list}\n\nTotal: ${total:.2f}\n\nIs this correct?"
            else:
                response = "Order is empty, nothing to confirm."

        elif tool_name == "place_order":
            if order:
                total = calculate_order_total(order)
                response = f"Order placed! Your total is ${total:.2f}. Thank you for your order!"
                finished = True
            else:
                response = "Cannot place empty order."

        elif tool_name == "clear_order":
            order = []
            response = "Order cleared. Starting fresh!"

        elif tool_name == "calculate_total":
            if not order:
                response = "Order is empty. Total: $0.00"
            else:
                total = 0.0
                breakdown = []
                for item in order:
                    item_price = calculate_item_price(item)
                    total += item_price
                    breakdown.append(f"  - {item}: ${item_price:.2f}")
                response = "Order breakdown:\n" + "\n".join(breakdown) + f"\n\nTotal: ${total:.2f}"

        else:
            response = f"Unknown tool: {tool_name}"

        outbound_msgs.append(
            ToolMessage(
                content=response,
                name=tool_name,
                tool_call_id=tool_call["id"],
            )
        )

    return {"messages": outbound_msgs, "order": order, "finished": finished}


def calculate_item_price(item: str) -> float:
    """Calculate price for a single item including modifiers."""
    item_lower = item.lower()
    price = 0.0

    for base_item, base_price in PRICES.items():
        if base_item in item_lower:
            price = base_price
            break

    for modifier, mod_price in MODIFIER_PRICES.items():
        if modifier in item_lower:
            price += mod_price

    return price


def calculate_order_total(order: list[str]) -> float:
    """Calculate total for entire order."""
    return sum(calculate_item_price(item) for item in order)


# =============================================================================
# Routing
# =============================================================================

def route_after_barista(state: State) -> str:
    """Route based on tool calls in the last message."""
    last_msg = state["messages"][-1]

    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return END

    tool_names = [tc["name"] for tc in last_msg.tool_calls]
    stateful_tool_names = {t.name for t in STATEFUL_TOOLS}

    if any(name in stateful_tool_names for name in tool_names):
        return "order_node"

    return "tools"


def route_after_order(state: State) -> str:
    """Continue to barista for response after tool execution."""
    return "barista"


def route_after_tools(state: State) -> str:
    """Continue to barista after stateless tools."""
    return "barista"


# =============================================================================
# Build Graph
# =============================================================================

# Session checkpointer - stores conversation state per session_id.
# Note: MemorySaver stores sessions in-memory and grows unbounded.
# For production with many users, consider Redis or database-backed checkpointer.
memory = MemorySaver()


def build_graph():
    """Construct the barista state graph."""
    graph = StateGraph(State)

    graph.add_node("barista", barista_node)
    graph.add_node("tools", ToolNode(STATELESS_TOOLS))
    graph.add_node("order_node", order_node)

    graph.add_edge(START, "barista")
    graph.add_edge("tools", "barista")
    graph.add_edge("order_node", "barista")

    graph.add_conditional_edges("barista", route_after_barista)

    return graph.compile(checkpointer=memory)


# Singleton graph instance
_graph = None


def get_graph():
    """Get or create the graph instance."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def chat(message: str, session_id: str) -> tuple[str, bool]:
    """
    Process a chat message and return the response.

    Args:
        message: User's message
        session_id: Unique session identifier

    Returns:
        Tuple of (response_text, is_finished)
    """
    graph = get_graph()

    config = {"configurable": {"thread_id": session_id}}

    # Get current state or initialize
    current_state = graph.get_state(config)

    if current_state.values:
        # Existing session - add the new message
        input_state = {"messages": [HumanMessage(content=message)]}
    else:
        # New session - initialize with greeting trigger
        if message.strip():
            input_state = {
                "messages": [HumanMessage(content=message)],
                "order": [],
                "finished": False,
            }
        else:
            # Empty message = initial greeting
            input_state = {
                "messages": [HumanMessage(content="Hello!")],
                "order": [],
                "finished": False,
            }

    # Run the graph
    result = graph.invoke(input_state, config)

    # Extract the last AI message
    messages = result.get("messages", [])
    response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            response = msg.content
            break

    finished = result.get("finished", False)

    return response, finished
