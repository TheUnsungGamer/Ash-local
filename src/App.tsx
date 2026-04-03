import { useState } from "react";
import "./styles/global.css";
import { initialLogs } from "./constants/initialLogs";

import { useAudioController } from "./hooks/useAudioController";
import { useChat } from "./hooks/useChat";
import { starterMessages } from "./constants/starterMessages";
import { AppShell } from "./components/layout/AppShell";
import { MainPanel } from "./components/layout/MainPanel";
import { AppHeader } from "./components/layout/AppHeader";
import { ChatSection } from "./components/chat/ChatSection";
import { StatusStrip } from "./components/layout/StatusStrip";
import { Sidebar } from "./components/controls/Sidebar";

export default function App() {
  const [draftMessage, setDraftMessage] = useState("");
  const [logs, setLogs] = useState<string[]>(initialLogs);

  const audioController = useAudioController();

  function pushLog(logLine: string) {
    setLogs((prev) => [...prev.slice(-19), logLine]);
  }

  const {
    messages,
    availableModels,
    selectedModel,
    isStreaming,
    errorMessage,
    setSelectedModel,
    sendMessage,
  } = useChat({
    initialMessages: starterMessages,
    onAssistantMessageComplete: async (message) => {
      pushLog("[LLM] Response complete");
      await audioController.speak(message.content, message.id);
    },
  });

  async function handleCopyMessage(messageContent: string) {
    try {
      await navigator.clipboard.writeText(messageContent);
      pushLog("[UI ] Message copied to clipboard");
    } catch {
      pushLog("[ERR] Clipboard write failed");
    }
  }

  async function handleSpeakMessage(messageContent: string, messageId: string) {
    await audioController.speak(messageContent, messageId);
  }

  async function handleSubmit() {
    const trimmedMessage = draftMessage.trim();

    if (!trimmedMessage) {
      return;
    }

    pushLog(`[USR] ${trimmedMessage.slice(0, 48)}`);
    await sendMessage(trimmedMessage);
    setDraftMessage("");
  }

  return (
    <AppShell>
      <AppHeader
        title="Ash UI Rebuild"
        subtitle="Modular chat test shell"
      />

      <StatusStrip
        selectedModel={selectedModel}
        audioStatus={audioController.audioState.status}
        voiceMode={audioController.audioState.voiceMode}
      />

      <MainPanel>
        <Sidebar
          availableModels={availableModels}
          selectedModel={selectedModel}
          isStreaming={isStreaming}
          errorMessage={errorMessage ?? undefined}
          logs={logs}
          audioState={audioController.audioState}
          onSelectModel={setSelectedModel}
          onToggleVoice={audioController.toggleVoice}
          onToggleMode={audioController.toggleMode}
          onPauseAudio={audioController.pause}
          onResumeAudio={audioController.resume}
          onStopAudio={audioController.stop}
          messageCount={messages.length}
        />

        <ChatSection
  messages={messages}
  isStreaming={isStreaming}
  draftMessage={draftMessage}
  onDraftChange={setDraftMessage}
  onSubmit={handleSubmit}
  onCopyMessage={(message) => void handleCopyMessage(message.content)}
  onSpeakMessage={(message) =>
    void handleSpeakMessage(message.content, message.id)
  }
/>
      </MainPanel>
    </AppShell>
  );
}