/**
 * The environment fields every record starts with (after its `format` and `page`): when, which
 * browser, whether the page was cross-origin isolated, and the adapter and device facts.
 */

import type { GpuContext } from "@karume/runtime";

import {
  type AdapterSummary,
  type DeviceFacts,
  summarizeAdapter,
  summarizeDevice,
} from "@/lib/record";

export type RecordEnvironment = {
  readonly recordedAt: string;
  readonly userAgent: string;
  readonly crossOriginIsolated: boolean;
  readonly adapter: AdapterSummary;
  readonly device: DeviceFacts;
};

export const recordEnvironment = (gpu: GpuContext): RecordEnvironment => ({
  recordedAt: new Date().toISOString(),
  userAgent: navigator.userAgent,
  crossOriginIsolated: typeof crossOriginIsolated === "boolean" && crossOriginIsolated,
  adapter: summarizeAdapter(gpu.adapterInfo),
  device: summarizeDevice(gpu),
});
