import type { ReactNode } from "react";

import { Label } from "@/components/ui/label";

/** A labelled form control with an optional hint underneath. */
export const Field = ({
  id,
  label,
  hint,
  children,
}: {
  id: string;
  label: string;
  hint?: ReactNode;
  children: ReactNode;
}) => (
  <div className="flex flex-col gap-1.5">
    <Label htmlFor={id}>{label}</Label>
    {children}
    {hint === undefined ? null : <p className="text-xs text-muted-foreground">{hint}</p>}
  </div>
);
