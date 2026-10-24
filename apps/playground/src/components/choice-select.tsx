import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export type Choice<Value extends string> = {
  readonly value: Value;
  readonly label: string;
  readonly disabled?: boolean;
};

/** The Shadcn Select for a flat list of string choices (the trigger shows the chosen label). */
export const ChoiceSelect = <Value extends string>({
  id,
  value,
  choices,
  onChange,
  disabled,
}: {
  id: string;
  value: Value;
  choices: readonly Choice<Value>[];
  onChange: (value: Value) => void;
  disabled?: boolean;
}) => (
  <Select
    value={value}
    items={choices.map(({ value: item, label }) => ({ value: item, label }))}
    onValueChange={(next) => {
      const found = choices.find((choice) => choice.value === next);
      if (found !== undefined) onChange(found.value);
    }}
    disabled={disabled ?? false}
  >
    <SelectTrigger id={id} className="w-full">
      <SelectValue />
    </SelectTrigger>
    <SelectContent>
      {choices.map((choice) => (
        <SelectItem key={choice.value} value={choice.value} disabled={choice.disabled ?? false}>
          {choice.label}
        </SelectItem>
      ))}
    </SelectContent>
  </Select>
);
