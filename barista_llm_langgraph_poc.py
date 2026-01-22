"""
Barista System - A coffee shop order-taking agent using LangGraph.

Graph structure:
- barista: LLM node that generates responses and may call tools
- tools: ToolNode for stateless tools (get_menu)
- order_node: Custom node for stateful tools (add_to_order, get_order, confirm_order, place_order)
- human: User input node with exit detection
"""

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

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

# Price lookup tables (source of truth)
PRICES = {
    # Drinks
    "espresso": 3.00,
    "americano": 3.50,
    "latte": 4.50,
    "cappuccino": 4.50,
    "mocha": 5.00,
    "cold brew": 4.00,
    # Food
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
    # This is a placeholder - actual logic is in order_node
    return f"Added {item} to order"


@tool
def get_order() -> str:
    """Get the current order."""
    # This is a placeholder - actual logic is in order_node
    return "Current order"


@tool
def confirm_order() -> str:
    """Show the order to the customer and ask for confirmation."""
    # This is a placeholder - actual logic is in order_node
    return "Order confirmed"


@tool
def place_order() -> str:
    """Place the final order after confirmation."""
    # This is a placeholder - actual logic is in order_node
    return "Order placed"


@tool
def clear_order() -> str:
    """Clear all items from the current order."""
    # This is a placeholder - actual logic is in order_node
    return "Order cleared"


@tool
def calculate_total() -> str:
    """Calculate the total price of the current order."""
    # This is a placeholder - actual logic is in order_node
    return "Total calculated"


# Separate tools by type
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
5. When they're done ordering, use confirm_order() to verify
6. After confirmation, use place_order() to finalize

Be conversational, helpful, and concise. Don't overwhelm the customer with too much text.
"""

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm_with_tools = llm.bind_tools(ALL_TOOLS)


# =============================================================================
# Nodes
# =============================================================================

def barista_node(state: State) -> State:
    """LLM generates a response, possibly with tool calls."""
    from langchain_core.messages import SystemMessage, HumanMessage

    messages = list(state["messages"])

    # Always prepend system prompt
    messages_to_send = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    # If this is the very first call (no messages yet), add a starter prompt
    if len(messages) == 0:
        messages_to_send.append(HumanMessage(content="Hello!"))

    response = llm_with_tools.invoke(messages_to_send)
    return {"messages": [response]}


def order_node(state: State) -> State:
    """Handle stateful order tools."""
    last_msg = state["messages"][-1]
    order = state.get("order", [])
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
                print("\n" + "="*40)
                print("YOUR ORDER:")
                for item in order:
                    print(f"  - {item}")
                print("="*40)
                user_confirm = input("Is this correct? (yes/no): ").strip().lower()
                response = f"Customer said: {user_confirm}"
            else:
                response = "Order is empty, nothing to confirm."

        elif tool_name == "place_order":
            if order:
                print("\n" + "="*40)
                print("ORDER PLACED!")
                for item in order:
                    print(f"  - {item}")
                print("="*40 + "\n")
                response = "Order has been placed! Thank you!"
                finished = True
            else:
                response = "Cannot place empty order."

        elif tool_name == "clear_order":
            order.clear()
            response = "Order cleared. Starting fresh!"

        elif tool_name == "calculate_total":
            if not order:
                response = "Order is empty. Total: $0.00"
            else:
                total = 0.0
                breakdown = []

                for item in order:
                    item_lower = item.lower()
                    item_price = 0.0

                    # Find base item price
                    for base_item, price in PRICES.items():
                        if base_item in item_lower:
                            item_price = price
                            break

                    # Add modifier prices
                    for modifier, mod_price in MODIFIER_PRICES.items():
                        if modifier in item_lower:
                            item_price += mod_price

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


def human_node(state: State) -> State:
    """Get user input and check for exit."""
    last_msg = state["messages"][-1]

    # Print the barista's response
    print(f"\nBarista: {last_msg.content}\n")

    # Get user input
    user_input = input("You: ").strip()

    # Check for exit keywords
    exit_keywords = {"bye", "goodbye", "quit", "exit", "done", "no thanks"}
    finished = user_input.lower() in exit_keywords

    return {"messages": [("user", user_input)], "finished": finished}


# =============================================================================
# Routing
# =============================================================================

def route_after_barista(state: State) -> str:
    """Route based on tool calls in the last message."""
    last_msg = state["messages"][-1]

    # No tool calls -> go to human
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return "human"

    # Check what tools are being called
    tool_names = [tc["name"] for tc in last_msg.tool_calls]
    stateful_tool_names = {t.name for t in STATEFUL_TOOLS}

    # If any stateful tool -> order_node
    if any(name in stateful_tool_names for name in tool_names):
        return "order_node"

    # Otherwise -> tools (ToolNode)
    return "tools"


def route_after_order(state: State) -> str:
    """Check if order was placed (finished)."""
    if state.get("finished", False):
        return END
    return "barista"


def route_after_human(state: State) -> str:
    """Check if user wants to exit."""
    if state.get("finished", False):
        return END
    return "barista"


# =============================================================================
# Build Graph
# =============================================================================

def build_graph():
    """Construct the barista state graph."""
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("barista", barista_node)
    graph.add_node("tools", ToolNode(STATELESS_TOOLS))
    graph.add_node("order_node", order_node)
    graph.add_node("human", human_node)

    # Add edges
    graph.add_edge(START, "barista")
    graph.add_edge("tools", "barista")

    # Conditional edges
    graph.add_conditional_edges("barista", route_after_barista)
    graph.add_conditional_edges("order_node", route_after_order)
    graph.add_conditional_edges("human", route_after_human)

    return graph.compile()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the barista chatbot."""
    print("\n" + "="*50)
    print("   Welcome to the Coffee Shop!")
    print("   (Type 'bye' to exit)")
    print("="*50 + "\n")

    app = build_graph()

    # Initial state
    initial_state = {
        "messages": [],
        "order": [],
        "finished": False,
    }

    # Run the graph
    app.invoke(initial_state)

    print("\nThanks for visiting! Have a great day!\n")


if __name__ == "__main__":
    main()
