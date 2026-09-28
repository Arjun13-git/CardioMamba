// API contract types (System-Design/08-api-contracts/01-prediction-api.md).

export type ClassName = "NORM" | "MI" | "STTC" | "CD" | "HYP";

export interface LabelPrediction {
  label: ClassName;
  probability: number;
  threshold: number;
  positive: boolean;
}

export interface PredictionResponse {
  request_id: string;
  model: {
    name: string;
    version: string;
    task: string;
    preprocessing_version: string;
    checkpoint_epoch: number;
  };
  input: {
    format: string;
    leads: number;
    samples: number;
    sampling_rate_hz: number;
    model_sampling_rate_hz: number;
  };
  predictions: LabelPrediction[];
  waveform: {
    sampling_rate_hz: number;
    units: string;
    leads: string[];
    data: number[][]; // [12][1000]
  };
  inference_ms: number;
  device: string;
  disclaimer: string;
}

export interface ModelInfo {
  name: string;
  parameters: number;
  checkpoint_epoch: number;
  best_val_macro_auroc: number;
  preprocessing_version: string;
  thresholds: Record<ClassName, number>;
  device: string;
  precision: string;
}

export interface DemoSample {
  id: string;
  file: string;
  ecg_id: number;
  fold: number;
  reference_labels: ClassName[];
  sampling_rate_hz: number;
  samples: number;
}

export interface DemoManifest {
  provenance: string;
  selection_rule: string;
  samples: DemoSample[];
}
