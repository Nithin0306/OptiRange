"use client";
import { cn } from "@/lib/utils";

export function MessageBubble({
  role,
  content,
  timestamp: _timestamp,
  pending = false,
}: {
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  pending?: boolean;
}) {
  const isUser = role === "user";

  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] rounded-lg px-3 py-2 md:px-4 md:py-3 text-sm leading-relaxed",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-foreground border"
        )}
        aria-busy={pending}
      >
        <div className="whitespace-pre-wrap">{content}</div>
       
      </div>
    </div>
  );
}
