import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { type ExperimentalRule, experimentalRuleRefusal } from "@/lib/experimental-rule";

/**
 * The "Experimental fusion rule" control (`@/lib/experimental-rule`): off by default; disabled with
 * its one-line reason where the rule must not run (Metal for the VAE rule).
 */
export const ExperimentalRuleSwitch = ({
  id,
  rule,
  checked,
  metal,
  onCheckedChange,
  disabled,
}: {
  id: string;
  rule: ExperimentalRule;
  checked: boolean;
  /** Whether this page's WebGPU runs on Metal (`runsOnMetal`). */
  metal: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled: boolean;
}) => {
  const refusal = experimentalRuleRefusal(rule, metal);
  return (
    <div className="flex flex-col gap-1.5">
      <Label className="flex items-center gap-2">
        <Switch
          id={id}
          checked={checked && refusal === undefined}
          onCheckedChange={onCheckedChange}
          disabled={disabled || refusal !== undefined}
        />
        Experimental fusion rule: <code>{rule.name}</code>
      </Label>
      <p id={`${id}-note`} className="text-xs text-muted-foreground">
        {refusal ??
          `Off by default. On: the pipeline opens with this candidate rule (${rule.folds}), passed through its experimental fusionRules; the output is meant to stay bit-identical, only the time moves. The result and the record name the rule that ran.`}
      </p>
    </div>
  );
};
