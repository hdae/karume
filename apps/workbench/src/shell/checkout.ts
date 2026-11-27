/**
 * The checkout a browser run records: asked of the dev server when the run starts (so an edit
 * between two runs shows), or baked into a production build when its config loaded.
 */

import type { Checkout } from "@karume/sweep-app-support/record";

/** Defined by the results plugin in a production build only. */
declare const __WORKBENCH_CHECKOUT__: Checkout;

const isCheckout = (value: unknown): value is Checkout =>
  typeof value === "object" &&
  value !== null &&
  "commit" in value &&
  typeof value.commit === "string" &&
  "dirty" in value &&
  typeof value.dirty === "boolean" &&
  "runtimeVersion" in value &&
  typeof value.runtimeVersion === "string";

export const currentCheckout = async (): Promise<Checkout> => {
  if (!import.meta.env.DEV) return __WORKBENCH_CHECKOUT__;
  const response = await fetch("/workbench/checkout");
  if (!response.ok) throw new Error(`GET /workbench/checkout answered ${response.status}`);
  const body: unknown = await response.json();
  if (!isCheckout(body)) throw new Error("GET /workbench/checkout answered no checkout");
  return { commit: body.commit, dirty: body.dirty, runtimeVersion: body.runtimeVersion };
};
