import type { JsonRecord } from "./api";

export type ViewName = "trainer" | "auto" | "firmware" | "captured" | "samples" | "data";
export type SampleBucket = "personal" | "negative";

export interface LanguageOption extends JsonRecord {
  code: string;
  label: string;
  engines?: string[];
  quality?: string;
}

export interface TrainingState extends JsonRecord {
  running: boolean;
  exit_code: number | null;
  log_lines: string[];
}

export interface SessionPayload extends JsonRecord {
  safe_word?: string;
  raw_phrase?: string;
  language?: string;
  tts_mode?: string;
  takes_received?: number;
  available_languages?: LanguageOption[];
  training?: TrainingState;
}

export interface AudioItem extends JsonRecord {
  saved_as: string;
  original_name?: string;
  audio_url?: string;
  final_format?: JsonRecord;
}

export interface SamplesPayload extends JsonRecord {
  personal: AudioItem[];
  negative: AudioItem[];
  personal_count: number;
  negative_count: number;
}

export interface CapturedPayload extends JsonRecord {
  items: AudioItem[];
  captured_count: number;
  personal_count: number;
  negative_count: number;
}

export interface AutoTrainForm extends JsonRecord {
  enabled: boolean;
  wake_phrase: string;
  language: string;
  stt_engine: string;
  minimum_transcript_chars: number;
  delete_confirmed_wakes: boolean;
  promote_close_misses: boolean;
  schedule_hours: number;
  minimum_new_negatives: number;
  advertised_base_url: string;
  tater_url: string;
  notify_satellites: boolean;
}

export interface AutoTrainPayload extends JsonRecord {
  config?: Partial<AutoTrainForm>;
  state?: JsonRecord;
  runtime?: JsonRecord;
  trainer_link?: JsonRecord;
  advertised_base_url?: string;
}

export interface WakeWordItem extends JsonRecord {
  key?: string;
  label?: string;
  url?: string;
  json_url?: string;
  jsonUrl?: string;
  esphome_json_url?: string;
  esphomeJsonUrl?: string;
  model_url?: string;
  modelUrl?: string;
}

export interface ManagedDataItem extends JsonRecord {
  id: string;
  label: string;
  category: string;
  description: string;
  location: string;
  size_bytes: number;
  file_count: number;
  exists: boolean;
  rebuild_note?: string;
}

export interface ManagedDataPayload extends JsonRecord {
  items: ManagedDataItem[];
  total_size_bytes: number;
  total_file_count: number;
}

export interface ToastState {
  message: string;
  tone: "success" | "warning" | "error";
  serial: number;
}
