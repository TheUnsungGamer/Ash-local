import type { ChatMessage } from "../types/chat";

export const starterMessages: ChatMessage[] = [
  {
    id: crypto.randomUUID(),
    role: "assistant",
    content: "SYSTEM ONLINE // LOCAL MODEL ACTIVE",
    createdAt: Date.now(),
  },
  {
    id: crypto.randomUUID(),
    role: "assistant",
    content: "LOG STREAM READY // MAP MODULE STANDBY",
    createdAt: Date.now(),
  },
];