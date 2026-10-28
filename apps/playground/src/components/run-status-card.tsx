import type { ReactNode } from "react";

import { ErrorAlert } from "@/components/error-alert";
import { KeyValueTable } from "@/components/key-value-table";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import type { GpuState } from "@/hooks/use-gpu";
import type { RunTaskView } from "@/hooks/use-run-task";
import { adapterText } from "@/lib/record";

/**
 * A run's state: while running the elapsed seconds, the stage, the counter (with a bar) and Cancel
 * (disabled, with the stage's `cancelNote`, while the stage is not cancellable); afterwards
 * "Cancelled" or the error in full. Nothing while idle or done (the page shows the result).
 */
export const RunStatus = ({ task }: { task: RunTaskView }) => {
  const { state, now } = task;
  if (state.kind === "cancelled") {
    return (
      <Alert>
        <AlertTitle>Cancelled</AlertTitle>
        <AlertDescription>
          The run stopped at the next point it checks for a cancel.
        </AlertDescription>
      </Alert>
    );
  }
  if (state.kind === "error") return <ErrorAlert title="Run failed" error={state.error} />;
  if (state.kind !== "running") return null;
  const { stage, count, note, cancellable, cancelNote } = state.progress;
  return (
    <div className="flex flex-col gap-2 text-sm">
      <div className="flex items-center gap-2 font-medium">
        <Spinner />
        <span>
          Running — {((now - state.startedAt) / 1000).toFixed(1)} s
          {stage === undefined ? "" : ` · ${stage}`}
        </span>
      </div>
      {count === undefined ? null : (
        <div className="flex flex-col gap-1">
          <Progress value={count.total === 0 ? 0 : (count.done / count.total) * 100} />
          <span className="font-mono text-xs text-muted-foreground tabular-nums">
            {count.done} / {count.total} {count.unit}
          </span>
        </div>
      )}
      {note === undefined ? null : (
        <span className="font-mono text-xs text-muted-foreground">{note}</span>
      )}
      <Button
        id="run-cancel"
        variant="outline"
        size="sm"
        className="self-start"
        onClick={task.cancel}
        disabled={cancellable === false}
      >
        Cancel
      </Button>
      {cancellable === false ? (
        <span id="run-cancel-note" className="text-xs text-muted-foreground">
          {cancelNote}
        </span>
      ) : null}
    </div>
  );
};

/** The GPU's failure states: the acquisition error, or a lost device with "Acquire the GPU again". */
export const GpuAlerts = ({ gpu, onReacquire }: { gpu: GpuState; onReacquire: () => void }) => {
  if (gpu.kind === "error") return <ErrorAlert title="GPU" error={gpu.error} />;
  if (gpu.kind !== "lost") return null;
  return (
    <Alert variant="destructive">
      <AlertTitle>GPU device lost ({gpu.reason})</AlertTitle>
      <AlertDescription className="flex flex-col gap-2">
        <span>{gpu.message}</span>
        <Button id="gpu-reacquire" variant="outline" size="sm" onClick={onReacquire}>
          Acquire the GPU again
        </Button>
      </AlertDescription>
    </Alert>
  );
};

/**
 * The Status card of a model page: the GPU and cross-origin isolation, the page's own `rows`, the
 * page's own controls (`children`), the GPU's failure states and the run ({@link RunStatus}).
 */
export const RunStatusCard = ({
  gpu,
  onReacquire,
  task,
  rows = [],
  children,
}: {
  gpu: GpuState;
  onReacquire: () => void;
  task: RunTaskView;
  rows?: readonly (readonly [label: string, value: ReactNode])[];
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
      <RunStatus task={task} />
    </CardContent>
  </Card>
);
