import { ChoiceSelect } from "@/components/choice-select";
import { Field } from "@/components/field";
import { type Preset, presetChoices } from "@/lib/presets";

/**
 * The Preset field: the manifest's quants (`manifestPresets`), the default marked "(default)", the
 * chosen preset's description as the hint.
 */
export const PresetSelect = ({
  id,
  value,
  presets,
  defaultQuant,
  onChange,
  disabled,
}: {
  id: string;
  value: string;
  presets: readonly Preset[];
  defaultQuant: string;
  onChange: (quant: string) => void;
  disabled?: boolean;
}) => {
  const chosen = presets.find(({ name }) => name === value);
  return (
    <Field
      id={id}
      label="Preset"
      hint={chosen?.description === "" ? undefined : chosen?.description}
    >
      <ChoiceSelect
        id={id}
        value={value}
        choices={presetChoices(presets, defaultQuant)}
        onChange={onChange}
        disabled={disabled ?? false}
      />
    </Field>
  );
};
