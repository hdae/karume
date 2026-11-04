// The minimal asset fixture the `assetComponentOpener` checks use (shared by
// asset_container_components_test.ts and asset_container_components_test.ts).
//
// The container is written through `container-fixture.ts` (the format tooling — see the NOTE
// there).

import { assetComponentOpener, type ComponentOpener } from "../../src/hub/components.ts";
import { linearComponent, type TestContainer, writeContainer } from "./container-fixture.ts";

/** A container of one runnable component (a single `linear`). Its weights are only in part 2. */
const ditContainer: TestContainer = await writeContainer(linearComponent("dit"));

/** The single form (the whole container in one piece). */
export const single = (): Uint8Array<ArrayBuffer> => ditContainer.single;

/** One part of the split form (`index` is the part number — 0 is the descriptor). */
export const part = (index: number): Uint8Array<ArrayBuffer> => ditContainer.parts[index];

/** The split form's parts under `<role>[i]` keys (zero-length parts are listed too). */
export const parts = (key: string): Record<string, Uint8Array<ArrayBuffer>> =>
  Object.fromEntries(ditContainer.parts.map((bytes, index) => [`${key}[${index}]`, bytes]));

/** A minimal implementation shaped like the families' asset accessor (`assetBuffer`). */
const bufferOf =
  (assets: Readonly<Record<string, Uint8Array<ArrayBuffer>>>) =>
  (key: string): ArrayBuffer => {
    if (!Object.hasOwn(assets, key)) {
      throw new Error(
        `test: asset '${key}' is missing (keys present: ${Object.keys(assets).join(" / ")})`,
      );
    }
    return assets[key].buffer;
  };

export const openerOf = (
  assets: Record<string, Uint8Array<ArrayBuffer>>,
  componentKeys: readonly string[] = ["dit"],
): Promise<ComponentOpener> =>
  assetComponentOpener("test", assets, bufferOf(assets), componentKeys);
