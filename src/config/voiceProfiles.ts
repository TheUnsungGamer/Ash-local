export const VOICE_PROFILES = {
  VERITY: {
    systemPrompt:
      "Act as a spacecraft AI (Verity). British RP, concise, technical, no filler. Confirmations are final.",
    piperModel: "en_GB-alba-medium",
    audioSettings: {
      rate: 0.95,
      pitch: 1.1,
      highPassHz: 200,
    },
  },
} as const;