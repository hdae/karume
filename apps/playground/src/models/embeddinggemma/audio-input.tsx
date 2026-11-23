import { useState } from "react";

import { ErrorAlert } from "@/components/error-alert";
import { Field } from "@/components/field";
import { Input } from "@/components/ui/input";

import { audioSummary, decodeAudioFile, type LoadedAudio } from "./audio";

/**
 * An audio file field of a row: the browser decodes the file, resamples it to `sampleRate` and mixes
 * it to mono (`./audio.ts`); `onAudio` gets the result, which the page keeps (`audio`) so the summary
 * survives a re-render.
 */
export const AudioInput = ({
  id,
  label = "Audio",
  sampleRate,
  audio,
  onAudio,
  disabled,
}: {
  id: string;
  label?: string;
  sampleRate: number;
  audio: LoadedAudio | undefined;
  onAudio: (audio: LoadedAudio) => void;
  disabled?: boolean;
}) => {
  const [error, setError] = useState<unknown>(undefined);
  const [decoding, setDecoding] = useState(false);
  return (
    <div className="flex flex-col gap-3">
      <Field
        id={id}
        label={label}
        hint={
          decoding
            ? "Decoding…"
            : audio === undefined
              ? `Any format the browser decodes; resampled to ${sampleRate} Hz and mixed to mono by the browser.`
              : audioSummary(audio)
        }
      >
        <Input
          id={id}
          type="file"
          accept="audio/*,.wav"
          disabled={(disabled ?? false) || decoding}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file === undefined) return;
            setError(undefined);
            setDecoding(true);
            decodeAudioFile(file, sampleRate).then(
              (decoded) => {
                setDecoding(false);
                onAudio(decoded);
              },
              (failure: unknown) => {
                setDecoding(false);
                setError(failure);
              },
            );
          }}
        />
      </Field>
      {error === undefined ? null : <ErrorAlert title="Audio" error={error} />}
    </div>
  );
};
