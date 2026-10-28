import { Field } from "@/components/field";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { SourceKind } from "@/lib/distributions";

/**
 * The Source field: **Cached** (the hub's cache over `/hf/`) or **Direct** (plain fetch from
 * `/models/`). `value` is the effective kind (`source.kind` of {@link usePipelineSource}); Cached is
 * disabled while the mount has no HF-compatible route.
 */
export const SourcePicker = ({
  id,
  value,
  cachedAvailable,
  onChange,
  disabled,
}: {
  id: string;
  value: SourceKind;
  cachedAvailable: boolean;
  onChange: (kind: SourceKind) => void;
  disabled?: boolean;
}) => (
  <Field
    id={id}
    label="Source"
    hint={
      !cachedAvailable
        ? "Direct only: the dev server serves no HF-compatible route for this distribution."
        : value === "cached"
          ? "Through the hub's cache (/hf/): the first run stores the weights in this browser, later runs read them from its disk."
          : "Plain fetch from /models/ on every run; nothing is cached."
    }
  >
    <ToggleGroup
      id={id}
      variant="outline"
      spacing={0}
      value={[value]}
      onValueChange={(next) => {
        const [kind] = next;
        if (kind === "cached" || kind === "direct") onChange(kind);
      }}
      disabled={disabled ?? false}
    >
      <ToggleGroupItem value="cached" disabled={!cachedAvailable}>
        Cached
      </ToggleGroupItem>
      <ToggleGroupItem value="direct">Direct</ToggleGroupItem>
    </ToggleGroup>
  </Field>
);
