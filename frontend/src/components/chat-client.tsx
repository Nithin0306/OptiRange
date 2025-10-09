"use client";

import type React from "react";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MessageBubble } from "./message-bubble";
import { cn } from "@/lib/utils";
import { Loader2, Send } from "lucide-react";

type Role = "user" | "assistant";
type Message = { id: string; role: Role; content: string; createdAt: number };

const DEFAULT_SESSION_KEY = "stock-analyzer-chat:session-id";
const HISTORY_PREFIX = "stock-analyzer-chat:history:";

const BACKEND_URL =
  (typeof process !== "undefined" &&
    (process as any).env?.VITE_PUBLIC_BACKEND_URL) ||
  "http://localhost:5000";

function uid() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto)
    return crypto.randomUUID();
  return Math.random().toString(36).slice(2);
}

export function ChatClient() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages.length]);

  const disabled = sending || input.trim().length === 0;

  const sendMessage = async () => {
    const text = input.trim();
    if (!text) return;

    const userMsg: Message = {
      id: uid(),
      role: "user",
      content: text,
      createdAt: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setSending(true);

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });

      if (!res.ok) throw new Error(`Backend error: ${res.status}`);

      const data = (await res.json()) as { answer?: string; error?: string };
      const answerText = data?.answer ?? "I could not find a suitable answer.";

      const botMsg: Message = {
        id: uid(),
        role: "assistant",
        content: answerText,
        createdAt: Date.now(),
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: uid(),
          role: "assistant",
          content:
            "I’m having trouble reaching the server. Please ensure the backend is running and CORS is enabled.",
          createdAt: Date.now(),
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !disabled) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="mx-auto max-w-3xl p-4 md:p-6">
      <Card className="border-muted">
        <CardHeader className="pb-2">
          <CardTitle className="text-pretty text-xl md:text-2xl">
            Stock Analyzer Chatbot
          </CardTitle>
        </CardHeader>

        <CardContent className="pt-0">
          <div
            ref={scrollRef}
            className={cn(
              "h-[65dvh] w-full overflow-y-auto rounded-md border bg-muted/20 p-3 md:p-4",
              "scrollbar-thin scrollbar-thumb-muted-foreground/20 scrollbar-track-transparent"
            )}
            aria-live="polite"
            aria-busy={sending}
            role="log"
          >
            {messages.length === 0 ? (
              <div className="flex h-full items-center justify-center text-center text-sm text-muted-foreground">
                Ask anything about AAPL, MSFT, or TSLA closing prices.
              </div>
            ) : (
              <div className="flex flex-col gap-3 md:gap-4">
                {messages.map((m) => (
                  <MessageBubble
                    key={m.id}
                    role={m.role}
                    content={m.content}
                    timestamp={m.createdAt}
                  />
                ))}
                {sending && (
                  <MessageBubble
                    role="assistant"
                    content="Thinking…"
                    timestamp={Date.now()}
                    pending
                  />
                )}
              </div>
            )}
          </div>

          <form
            className="mt-3 flex items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (!disabled) sendMessage();
            }}
          >
            <label htmlFor="chat-input" className="sr-only">
              Type your message
            </label>
            <Input
              id="chat-input"
              placeholder="e.g., Average AAPL close in Jan 2025?"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={sending}
              className="font-sans"
              aria-label="Message"
            />
            <Button
              type="submit"
              disabled={disabled}
              className="min-w-24 gap-2"
            >
              {sending ? (
                <>
                  <Loader2 className="size-4 animate-spin" />
                  Sending
                </>
              ) : (
                <>
                  <Send className="size-4" />
                  Send
                </>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
