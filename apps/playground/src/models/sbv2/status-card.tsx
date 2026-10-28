import { ModelInputError } from "@karume/models/sbv2";
import type { ReactNode } from "react";

import { ErrorAlert } from "@/components/error-alert";
import { KeyValueTable } from "@/components/key-value-table";
import { GpuAlerts, RunStatus } from "@/components/run-status-card";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { GpuState } from "@/hooks/use-gpu";
import type { RunTaskView } from "@/hooks/use-run-task";
import { adapterText } from "@/lib/record";

/**
 * A failed run: an input the pipeline refused (`InputError`, a `ModelInputError`) under its own
 * title with `inputHint`, anything else as a failure — both in full (message, stack, cause chain).
 */
const RunFailure = ({ task, inputHint }: { task: RunTaskView; inputHint: ReactNode }) => {
  const { state } = task;
  if (state.kind !== "error") return null;
  if (!(state.error instanceof ModelInputError)) {
    return <ErrorAlert title="Run failed" error={state.error} />;
  }
  return (
    <div id="sbv2-input-error" className="flex flex-col gap-2">
      <ErrorAlert title={`Input rejected — ${state.error.name}`} error={state.error} />
      <p className="text-xs text-muted-foreground">{inputHint}</p>
    </div>
  );
};

/**
 * The Status card of the page: the GPU and cross-origin isolation, the page's `rows` and controls
 * (`children`), the GPU's failure states and the synthesis: {@link RunStatus} while it runs (the
 * page reports every stage as not cancellable), the failure with the input hint afterwards.
 */
export const Sbv2StatusCard = ({
  gpu,
  onReacquire,
  task,
  rows,
  inputHint,
  children,
}: {
  gpu: GpuState;
  onReacquire: () => void;
  task: RunTaskView;
  rows: readonly (readonly [label: string, value: ReactNode])[];
  inputHint: ReactNode;
  children?: ReactNode;
}) => (
  <Card>
    <CardHeader>
      <CardTitle>Status</CardTitle>
    </CardHeader>
    <CardContent className="flex flex-col gap-4">
      <KeyValueTable
        rows={[
          ["GPU", gpu.kind === "ready" ? adapterText(gpu.gpu.adapterInfo) : gpu.kind],
          [
            "Cross-origin isolated",
            String(typeof crossOriginIsolated === "boolean" && crossOriginIsolated),
          ],
          ...rows,
        ]}
      />
      {children}
      <GpuAlerts gpu={gpu} onReacquire={onReacquire} />
      {task.state.kind === "running" ? <RunStatus task={task} /> : null}
      <RunFailure task={task} inputHint={inputHint} />
    </CardContent>
  </Card>
);
