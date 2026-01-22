"use client";

import { useState, useRef, useEffect, useCallback } from "react";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
}

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isFinished, setIsFinished] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const startConversation = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/start`, {
        method: "POST",
      });
      const data = await response.json();
      setSessionId(data.session_id);
      setMessages([{ id: generateId(), role: "assistant", content: data.response }]);
      setIsFinished(data.finished);
    } catch (error) {
      console.error("Failed to start conversation:", error);
      setMessages([
        {
          id: generateId(),
          role: "assistant",
          content: "Sorry, I couldn't connect to the server. Please try again later.",
        },
      ]);
    }
    setIsLoading(false);
  }, [API_URL]);

  // Start conversation on mount
  useEffect(() => {
    startConversation();
  }, [startConversation]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || isFinished) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { id: generateId(), role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId,
        }),
      });
      const data = await response.json();
      setMessages((prev) => [...prev, { id: generateId(), role: "assistant", content: data.response }]);
      setIsFinished(data.finished);
    } catch (error) {
      console.error("Failed to send message:", error);
      setMessages((prev) => [
        ...prev,
        { id: generateId(), role: "assistant", content: "Sorry, something went wrong. Please try again." },
      ]);
    }
    setIsLoading(false);
  };

  const resetConversation = () => {
    setMessages([]);
    setSessionId(null);
    setIsFinished(false);
    startConversation();
  };

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8">
      <div className="w-full max-w-2xl flex flex-col h-[90vh]">
        {/* Header */}
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold text-amber-900 mb-2">
            ☕ Barista Agent
          </h1>
          <p className="text-amber-700 text-sm">
            Your AI-powered coffee shop assistant
          </p>
        </div>

        {/* Chat Container */}
        <div className="flex-1 overflow-y-auto bg-white rounded-xl shadow-lg p-4 mb-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`mb-4 ${
                message.role === "user" ? "text-right" : "text-left"
              }`}
            >
              <div
                className={`inline-block max-w-[80%] p-3 rounded-2xl ${
                  message.role === "user"
                    ? "bg-amber-600 text-white rounded-br-md"
                    : "bg-amber-100 text-amber-900 rounded-bl-md"
                }`}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="text-left mb-4">
              <div className="inline-block bg-amber-100 text-amber-900 p-3 rounded-2xl rounded-bl-md">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-amber-600 rounded-full animate-bounce" />
                  <div className="w-2 h-2 bg-amber-600 rounded-full animate-bounce delay-100" />
                  <div className="w-2 h-2 bg-amber-600 rounded-full animate-bounce delay-200" />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Form */}
        {isFinished ? (
          <div className="text-center">
            <p className="text-amber-700 mb-4">Order complete! Thank you for visiting.</p>
            <button
              onClick={resetConversation}
              className="bg-amber-600 text-white px-6 py-2 rounded-full hover:bg-amber-700 transition"
            >
              Start New Order
            </button>
          </div>
        ) : (
          <form onSubmit={sendMessage} className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 p-3 border border-amber-300 rounded-full focus:outline-none focus:ring-2 focus:ring-amber-500"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="bg-amber-600 text-white px-6 py-3 rounded-full hover:bg-amber-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </form>
        )}

        {/* Footer */}
        <p className="text-center text-xs text-amber-600 mt-4">
          Powered by LangGraph + Gemini
        </p>
      </div>
    </main>
  );
}
