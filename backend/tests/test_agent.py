"""
Tests for the barista agent.

These tests cover the core business logic without requiring API keys.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent import (
    calculate_item_price,
    calculate_order_total,
    get_menu,
    order_node,
    barista_node,
    route_after_barista,
    MENU,
    PRICES,
    MODIFIER_PRICES,
)


class TestPriceCalculation:
    """Test price calculation logic."""

    def test_base_item_price(self):
        """Base items should return their menu price."""
        assert calculate_item_price("Latte") == 4.50
        assert calculate_item_price("espresso") == 3.00
        assert calculate_item_price("Croissant") == 3.50

    def test_item_with_modifier(self):
        """Items with modifiers should include modifier price."""
        # Latte ($4.50) + oat milk ($0.75) = $5.25
        assert calculate_item_price("Latte with oat milk") == 5.25

        # Cappuccino ($4.50) + extra shot ($0.50) = $5.00
        assert calculate_item_price("Cappuccino with extra shot") == 5.00

    def test_item_with_multiple_modifiers(self):
        """Items can have multiple modifiers."""
        # Latte ($4.50) + oat milk ($0.75) + extra shot ($0.50) = $5.75
        assert calculate_item_price("Latte with oat milk and extra shot") == 5.75

    def test_unknown_item_returns_zero(self):
        """Unknown items should return 0 (graceful handling)."""
        assert calculate_item_price("Unknown Item XYZ") == 0.0

    def test_order_total(self):
        """Order total should sum all items."""
        order = ["Latte", "Croissant", "Espresso"]
        # $4.50 + $3.50 + $3.00 = $11.00
        assert calculate_order_total(order) == 11.00

    def test_order_total_with_modifiers(self):
        """Order total should include modifier prices."""
        order = ["Latte with oat milk", "Muffin"]
        # $5.25 + $3.00 = $8.25
        assert calculate_order_total(order) == 8.25

    def test_empty_order_total(self):
        """Empty order should return 0."""
        assert calculate_order_total([]) == 0.0


class TestMenu:
    """Test menu data."""

    def test_menu_contains_drinks(self):
        """Menu should contain drink items."""
        assert "Latte" in MENU
        assert "Espresso" in MENU
        assert "Cold Brew" in MENU

    def test_menu_contains_food(self):
        """Menu should contain food items."""
        assert "Croissant" in MENU
        assert "Muffin" in MENU

    def test_menu_contains_modifiers(self):
        """Menu should list available modifiers."""
        assert "Oat milk" in MENU
        assert "Extra shot" in MENU

    def test_get_menu_tool_returns_menu(self):
        """get_menu tool should return the menu string."""
        result = get_menu.invoke({})
        assert "DRINKS" in result
        assert "FOOD" in result
        assert "MODIFIERS" in result

    def test_prices_match_menu(self):
        """All price dict items should be mentioned in menu."""
        for item in PRICES.keys():
            assert item.lower() in MENU.lower() or item.capitalize() in MENU


class TestOrderNode:
    """Test the order_node logic (stateful tool handling)."""

    def _make_tool_call_message(self, tool_name: str, args: dict, tool_id: str = "call_123"):
        """Helper to create an AIMessage with tool calls."""
        msg = AIMessage(content="")
        msg.tool_calls = [{"name": tool_name, "args": args, "id": tool_id}]
        return msg

    def test_add_to_order(self):
        """add_to_order should append item to order list."""
        msg = self._make_tool_call_message("add_to_order", {"item": "Latte"})
        state = {"messages": [msg], "order": [], "finished": False}

        result = order_node(state)

        assert result["order"] == ["Latte"]
        assert "Added" in result["messages"][0].content
        assert not result["finished"]

    def test_add_multiple_items(self):
        """Multiple add_to_order calls should accumulate."""
        msg = self._make_tool_call_message("add_to_order", {"item": "Croissant"})
        state = {"messages": [msg], "order": ["Latte"], "finished": False}

        result = order_node(state)

        assert result["order"] == ["Latte", "Croissant"]

    def test_get_order_empty(self):
        """get_order on empty order should indicate empty."""
        msg = self._make_tool_call_message("get_order", {})
        state = {"messages": [msg], "order": [], "finished": False}

        result = order_node(state)

        assert "empty" in result["messages"][0].content.lower()

    def test_get_order_with_items(self):
        """get_order should list current items."""
        msg = self._make_tool_call_message("get_order", {})
        state = {"messages": [msg], "order": ["Latte", "Muffin"], "finished": False}

        result = order_node(state)

        assert "Latte" in result["messages"][0].content
        assert "Muffin" in result["messages"][0].content

    def test_clear_order(self):
        """clear_order should empty the order list."""
        msg = self._make_tool_call_message("clear_order", {})
        state = {"messages": [msg], "order": ["Latte", "Muffin"], "finished": False}

        result = order_node(state)

        assert result["order"] == []
        assert "cleared" in result["messages"][0].content.lower()

    def test_place_order_sets_finished(self):
        """place_order should set finished=True."""
        msg = self._make_tool_call_message("place_order", {})
        state = {"messages": [msg], "order": ["Latte"], "finished": False}

        result = order_node(state)

        assert result["finished"] is True
        assert "placed" in result["messages"][0].content.lower()

    def test_place_empty_order_rejected(self):
        """Cannot place an empty order."""
        msg = self._make_tool_call_message("place_order", {})
        state = {"messages": [msg], "order": [], "finished": False}

        result = order_node(state)

        assert result["finished"] is False
        assert "empty" in result["messages"][0].content.lower()

    def test_confirm_order_shows_total(self):
        """confirm_order should show items and total."""
        msg = self._make_tool_call_message("confirm_order", {})
        state = {"messages": [msg], "order": ["Latte", "Croissant"], "finished": False}

        result = order_node(state)

        content = result["messages"][0].content
        assert "Latte" in content
        assert "Croissant" in content
        assert "$" in content  # Should show price


class TestBaristaNodeMocked:
    """Test barista_node with mocked LLM."""

    def test_barista_node_returns_llm_response(self):
        """barista_node should return the LLM's response."""
        mock_response = AIMessage(content="Welcome! What can I get you today?")

        with patch("app.agent.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            state = {
                "messages": [HumanMessage(content="Hello")],
                "order": [],
                "finished": False,
            }
            result = barista_node(state)

            assert len(result["messages"]) == 1
            assert result["messages"][0].content == "Welcome! What can I get you today?"

    def test_barista_node_passes_system_prompt(self):
        """barista_node should include system prompt in messages."""
        mock_response = AIMessage(content="Hi there!")

        with patch("app.agent.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_get_llm.return_value = mock_llm

            state = {
                "messages": [HumanMessage(content="Hello")],
                "order": [],
                "finished": False,
            }
            barista_node(state)

            # Check what was passed to invoke
            call_args = mock_llm.invoke.call_args[0][0]
            assert call_args[0].content  # First message should be system prompt
            assert "barista" in call_args[0].content.lower()


class TestRouting:
    """Test graph routing logic."""

    def test_route_to_end_when_no_tool_calls(self):
        """Should route to END when LLM doesn't call tools."""
        msg = AIMessage(content="Here's your order!")
        state = {"messages": [msg], "order": [], "finished": False}

        result = route_after_barista(state)

        assert result == "__end__"  # LangGraph's END constant

    def test_route_to_order_node_for_stateful_tools(self):
        """Should route to order_node for stateful tools."""
        msg = AIMessage(content="")
        msg.tool_calls = [{"name": "add_to_order", "args": {"item": "Latte"}, "id": "1"}]
        state = {"messages": [msg], "order": [], "finished": False}

        result = route_after_barista(state)

        assert result == "order_node"

    def test_route_to_tools_for_stateless_tools(self):
        """Should route to tools node for stateless tools like get_menu."""
        msg = AIMessage(content="")
        msg.tool_calls = [{"name": "get_menu", "args": {}, "id": "1"}]
        state = {"messages": [msg], "order": [], "finished": False}

        result = route_after_barista(state)

        assert result == "tools"
