import { useEffect, useState } from "react";

import { ErrorAlert } from "@/components/error-alert";
import { Button } from "@/components/ui/button";
import { downloadBlob } from "@/lib/download";

type ClipState =
  | { readonly kind: "encoding" }
  | { readonly kind: "ready"; readonly blob: Blob; readonly url: string }
  | { readonly kind: "error"; readonly error: unknown };

/**
 * Plays a mono f32 waveform and offers it as a WAV download. The WAV comes from `encodeWav` of the
 * models barrel (loaded here, not at page load), and the player plays that same file. Key it by the
 * clip, or pass a new `samples` array: either re-encodes.
 */
export const AudioClip = ({
  samples,
  sampleRate,
  filename,
}: {
  samples: Float32Array<ArrayBuffer>;
  sampleRate: number;
  filename: string;
}) => {
  const [state, setState] = useState<ClipState>({ kind: "encoding" });
  useEffect(() => {
    let disposed = false;
    let url: string | undefined;
    setState({ kind: "encoding" });
    import("@karume/models")
      .then(({ encodeWav }) => {
        const blob = new Blob([encodeWav(samples, sampleRate)], { type: "audio/wav" });
        if (disposed) return;
        url = URL.createObjectURL(blob);
        setState({ kind: "ready", blob, url });
      })
      .catch((error: unknown) => {
        if (!disposed) setState({ kind: "error", error });
      });
    return () => {
      disposed = true;
      if (url !== undefined) URL.revokeObjectURL(url);
    };
  }, [samples, sampleRate]);

  if (state.kind === "error") return <ErrorAlert title="WAV" error={state.error} />;
  return (
    <div className="flex flex-col gap-2">
      <audio controls src={state.kind === "ready" ? state.url : undefined} className="w-full" />
      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="outline"
          size="sm"
          disabled={state.kind !== "ready"}
          onClick={() => {
            if (state.kind === "ready") downloadBlob(state.blob, filename);
          }}
        >
          Download WAV
        </Button>
        <span className="font-mono text-xs text-muted-foreground tabular-nums">
          {(samples.length / sampleRate).toFixed(2)} s · {sampleRate} Hz
        </span>
      </div>
    </div>
  );
};
