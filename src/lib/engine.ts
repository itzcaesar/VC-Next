import { invoke } from "@tauri-apps/api/core";

export type ConversionMode = "quality" | "balanced" | "latency";

export interface SystemProfile {
  os: string;
  gpu: string;
  vramMb: number;
  driverVersion: string;
  source: "prototype-baseline" | "native-probe";
}

export interface ModelPreset {
  id: string;
  name: string;
  initials: string;
  format: "RVC v2" | "RVC ONNX";
  sampleRate: number;
}

export const FALLBACK_PROFILE: SystemProfile = {
  os: "Windows 11",
  gpu: "NVIDIA GeForce RTX 4050 Laptop GPU",
  vramMb: 6141,
  driverVersion: "610.62",
  source: "prototype-baseline",
};

export const MODEL_PRESETS: ModelPreset[] = [
  { id: "reference-rvc", name: "Reference RVC voice", initials: "RV", format: "RVC v2", sampleRate: 40000 },
  { id: "reference-onnx", name: "Reference ONNX voice", initials: "OX", format: "RVC ONNX", sampleRate: 48000 },
];

export async function getSystemProfile(): Promise<SystemProfile> {
  try {
    return await invoke<SystemProfile>("get_system_profile");
  } catch {
    return FALLBACK_PROFILE;
  }
}
