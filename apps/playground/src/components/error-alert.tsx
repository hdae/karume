import { CircleAlertIcon } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

/** A thrown value that is not an `Error` (anything can be thrown), as text. */
export const thrownText = (value: unknown): string => {
  if (value instanceof Error) return value.message;
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value) ?? typeof value;
  } catch {
    return typeof value;
  }
};

/** Message, stack and the `cause` chain of an error — in full (the point of the page is diagnosis). */
const describe = (error: unknown): string => {
  const lines: string[] = [];
  let current: unknown = error;
  for (let depth = 0; current !== undefined && depth < 8; depth += 1) {
    if (depth > 0) lines.push("", "Caused by:");
    if (current instanceof Error) {
      lines.push(current.stack ?? `${current.name}: ${current.message}`);
      current = current.cause;
    } else {
      lines.push(thrownText(current));
      current = undefined;
    }
  }
  return lines.join("\n");
};

export const ErrorAlert = ({ title, error }: { title: string; error: unknown }) => (
  <Alert variant="destructive">
    <CircleAlertIcon />
    <AlertTitle>{title}</AlertTitle>
    <AlertDescription>
      <pre className="max-h-80 overflow-auto text-xs whitespace-pre-wrap">{describe(error)}</pre>
    </AlertDescription>
  </Alert>
);
