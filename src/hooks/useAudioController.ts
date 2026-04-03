import { useCallback, useEffect, useRef, useState } from "react";
import type { AudioController, AudioState, VoiceMode } from "../types/audio";

const DEFAULT_AUDIO_STATE: AudioState = {
  status: "idle",
  activeText: null,
  activeMessageId: null,
  isVoiceEnabled: true,
  voiceMode: "tech_priest",
  errorMessage: null,
};

function getPreferredBrowserVoice(): SpeechSynthesisVoice | null {
  if (!("speechSynthesis" in window)) {
    return null;
  }

  const availableVoices = window.speechSynthesis.getVoices();

  if (availableVoices.length === 0) {
    return null;
  }

  return (
    availableVoices.find((voice) => voice.lang?.toLowerCase().includes("en-us")) ??
    availableVoices.find((voice) => voice.lang?.toLowerCase().startsWith("en")) ??
    availableVoices[0] ??
    null
  );
}

export function useAudioController(): AudioController {
  const [audioState, setAudioState] = useState<AudioState>(DEFAULT_AUDIO_STATE);

  const speechUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const htmlAudioRef = useRef<HTMLAudioElement | null>(null);
  const activeObjectUrlRef = useRef<string | null>(null);
  const loadedVoiceRef = useRef<boolean>(false);

  const clearActiveObjectUrl = useCallback(() => {
    if (activeObjectUrlRef.current) {
      URL.revokeObjectURL(activeObjectUrlRef.current);
      activeObjectUrlRef.current = null;
    }
  }, []);

  const clearHtmlAudio = useCallback(() => {
    if (htmlAudioRef.current) {
      htmlAudioRef.current.pause();
      htmlAudioRef.current.src = "";
      htmlAudioRef.current = null;
    }

    clearActiveObjectUrl();
  }, [clearActiveObjectUrl]);

  const clearSpeechUtterance = useCallback(() => {
    speechUtteranceRef.current = null;
  }, []);

  const stopBrowserSpeech = useCallback(() => {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }

    clearSpeechUtterance();
  }, [clearSpeechUtterance]);

  const interrupt = useCallback(() => {
    stopBrowserSpeech();
    clearHtmlAudio();

    setAudioState((previousState) => ({
      ...previousState,
      status: "interrupted",
      activeText: null,
      activeMessageId: null,
      errorMessage: null,
    }));
  }, [clearHtmlAudio, stopBrowserSpeech]);

  const stop = useCallback(() => {
    stopBrowserSpeech();
    clearHtmlAudio();

    setAudioState((previousState) => ({
      ...previousState,
      status: "stopped",
      activeText: null,
      activeMessageId: null,
      errorMessage: null,
    }));
  }, [clearHtmlAudio, stopBrowserSpeech]);

  const pause = useCallback(() => {
    const currentMode = audioState.voiceMode;

    try {
      if (currentMode === "normal") {
        if ("speechSynthesis" in window && window.speechSynthesis.speaking) {
          window.speechSynthesis.pause();
        }
      } else {
        htmlAudioRef.current?.pause();
      }

      setAudioState((previousState) => ({
        ...previousState,
        status: "paused",
      }));
    } catch (error) {
      setAudioState((previousState) => ({
        ...previousState,
        status: "error",
        errorMessage: error instanceof Error ? error.message : "Failed to pause audio playback.",
      }));
    }
  }, [audioState.voiceMode]);

  const resume = useCallback(() => {
    const currentMode = audioState.voiceMode;

    try {
      if (currentMode === "normal") {
        if ("speechSynthesis" in window) {
          window.speechSynthesis.resume();
        }
      } else {
        void htmlAudioRef.current?.play();
      }

      setAudioState((previousState) => ({
        ...previousState,
        status: "playing",
        errorMessage: null,
      }));
    } catch (error) {
      setAudioState((previousState) => ({
        ...previousState,
        status: "error",
        errorMessage: error instanceof Error ? error.message : "Failed to resume audio playback.",
      }));
    }
  }, [audioState.voiceMode]);

  const speakWithBrowserSpeech = useCallback(
    async (text: string, messageId?: string) => {
      if (!("speechSynthesis" in window)) {
        throw new Error("Browser speech synthesis is not supported in this environment.");
      }

      stopBrowserSpeech();
      clearHtmlAudio();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1;
      utterance.lang = "en-US";

      const preferredVoice = getPreferredBrowserVoice();
      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }

      speechUtteranceRef.current = utterance;

      await new Promise<void>((resolve, reject) => {
        utterance.onstart = () => {
          setAudioState((previousState) => ({
            ...previousState,
            status: "playing",
            activeText: text,
            activeMessageId: messageId ?? null,
            errorMessage: null,
          }));
        };

        utterance.onend = () => {
          clearSpeechUtterance();

          setAudioState((previousState) => ({
            ...previousState,
            status: "idle",
            activeText: null,
            activeMessageId: null,
            errorMessage: null,
          }));

          resolve();
        };

        utterance.onerror = (event) => {
          clearSpeechUtterance();

          setAudioState((previousState) => ({
            ...previousState,
            status: "error",
            activeText: null,
            activeMessageId: null,
            errorMessage: event.error || "Browser speech playback failed.",
          }));

          reject(new Error(event.error || "Browser speech playback failed."));
        };

        window.speechSynthesis.speak(utterance);
      });
    },
    [clearHtmlAudio, clearSpeechUtterance, stopBrowserSpeech]
  );

  const speakWithDynamicTts = useCallback(
  async (text: string, messageId?: string) => {
    try {
      stopBrowserSpeech();
      clearHtmlAudio();

      setAudioState((previousState) => ({
        ...previousState,
        status: "playing",
        activeText: text,
        activeMessageId: messageId ?? null,
        errorMessage: null,
      }));

      const response = await fetch("http://127.0.0.1:8000/tts", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      // 🔇 Silent 500
      if (!response.ok) {
        return;
      }

      const audioBlob = await response.blob();
      const objectUrl = URL.createObjectURL(audioBlob);
      activeObjectUrlRef.current = objectUrl;

      const audioElement = new Audio(objectUrl);

      // 🔫 Kill switch
      if (htmlAudioRef.current) {
        htmlAudioRef.current.pause();
      }

      htmlAudioRef.current = audioElement;

      await new Promise<void>((resolve, reject) => {
        audioElement.onended = () => {
          clearHtmlAudio();

          setAudioState((previousState) => ({
            ...previousState,
            status: "idle",
            activeText: null,
            activeMessageId: null,
            errorMessage: null,
          }));

          resolve();
        };

        audioElement.onerror = () => {
          clearHtmlAudio();
          reject(new Error("Dynamic TTS audio playback failed."));
        };

        void audioElement.play().catch(reject);
      });
    } catch {
      return;
    }
  },
  [clearHtmlAudio, stopBrowserSpeech]
);

  const speak = useCallback(
    async (text: string, messageId?: string) => {
      const trimmedText = text.trim();

      if (!audioState.isVoiceEnabled || trimmedText.length === 0) {
        return;
      }

      interrupt();

      setAudioState((previousState) => ({
        ...previousState,
        status: "playing",
        activeText: trimmedText,
        activeMessageId: messageId ?? null,
        errorMessage: null,
      }));

      try {
        if (audioState.voiceMode === "normal") {
          await speakWithBrowserSpeech(trimmedText, messageId);
          return;
        }

        await speakWithDynamicTts(trimmedText, messageId);
      } catch (error) {
        setAudioState((previousState) => ({
          ...previousState,
          status: "error",
          activeText: null,
          activeMessageId: null,
          errorMessage: error instanceof Error ? error.message : "Audio playback failed.",
        }));
      }
    },
    [
      audioState.isVoiceEnabled,
      audioState.voiceMode,
      interrupt,
      speakWithBrowserSpeech,
      speakWithDynamicTts,
    ]
  );

  const toggleVoice = useCallback(() => {
    setAudioState((previousState) => {
      const nextVoiceEnabledState = !previousState.isVoiceEnabled;

      if (!nextVoiceEnabledState) {
        stopBrowserSpeech();
        clearHtmlAudio();
      }

      return {
        ...previousState,
        isVoiceEnabled: nextVoiceEnabledState,
        status: nextVoiceEnabledState ? previousState.status : "stopped",
        activeText: nextVoiceEnabledState ? previousState.activeText : null,
        activeMessageId: nextVoiceEnabledState ? previousState.activeMessageId : null,
        errorMessage: null,
      };
    });
  }, [clearHtmlAudio, stopBrowserSpeech]);

  const toggleMode = useCallback(() => {
    setAudioState((previousState) => {
      const nextVoiceMode: VoiceMode =
        previousState.voiceMode === "normal" ? "tech_priest" : "normal";

      stopBrowserSpeech();
      clearHtmlAudio();

      return {
        ...previousState,
        voiceMode: nextVoiceMode,
        status: "stopped",
        activeText: null,
        activeMessageId: null,
        errorMessage: null,
      };
    });
  }, [clearHtmlAudio, stopBrowserSpeech]);

  useEffect(() => {
    if (!("speechSynthesis" in window) || loadedVoiceRef.current) {
      return;
    }

    const handleVoicesChanged = () => {
      loadedVoiceRef.current = true;
      window.speechSynthesis.getVoices();
    };

    window.speechSynthesis.getVoices();
    window.speechSynthesis.addEventListener("voiceschanged", handleVoicesChanged);

    return () => {
      window.speechSynthesis.removeEventListener("voiceschanged", handleVoicesChanged);
    };
  }, []);

  useEffect(() => {
    return () => {
      stopBrowserSpeech();
      clearHtmlAudio();
    };
  }, [clearHtmlAudio, stopBrowserSpeech]);

  return {
    audioState,
    speak,
    pause,
    resume,
    stop,
    interrupt,
    toggleVoice,
    toggleMode,
  };
}