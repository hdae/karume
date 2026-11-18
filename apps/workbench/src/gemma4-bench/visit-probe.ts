/**
 * The bench loop's probe when every visit has a device of its own (a bench with a profile switch —
 * `benchHasProfileSwitch` of `./plan.ts`): the loop takes one probe for the whole run
 * (`BenchDependencies.probe`), and this one stands for the probe of the device of the visit under
 * way.
 *
 * Between two visits there is no device. The loop turns the probe off before it builds a pipeline
 * and after a chat, which then has nothing to turn off; reading the records or resetting them
 * without a device is a bug of the caller and throws.
 *
 * Pure: `vp test` covers it with a fake probe.
 */

import type { BenchProbe } from "./run.ts";

/** What stands behind the visit's probe: the tap's probe of one device (`DeviceProbe`). */
export type VisitDeviceProbe = Required<BenchProbe>;

export type VisitProbe = {
  /** The probe to give the bench loop. */
  readonly probe: BenchProbe;
  /** A visit's device was acquired: its probe is the one in use, turned off until a chat turns it on. */
  readonly attach: (device: VisitDeviceProbe) => void;
  /** The visit's device is released. */
  readonly detach: () => void;
};

export const createVisitProbe = (): VisitProbe => {
  let current: VisitDeviceProbe | undefined;
  const live = (): VisitDeviceProbe => {
    if (current === undefined) throw new Error("The bench read its probe with no visit's device");
    return current;
  };
  return {
    probe: {
      get enabled() {
        return current?.enabled ?? false;
      },
      set enabled(value: boolean) {
        if (current !== undefined) current.enabled = value;
        else if (value) live();
      },
      get mode() {
        return current?.mode;
      },
      set mode(value) {
        if (value !== undefined) live().mode = value;
      },
      passes: () => live().passes(),
      submits: () => live().submits(),
      marks: () => live().marks(),
      readTimes: () => live().readTimes(),
      reset: () => live().reset(),
    },
    attach: (device) => {
      device.enabled = false;
      current = device;
    },
    detach: () => {
      current = undefined;
    },
  };
};
