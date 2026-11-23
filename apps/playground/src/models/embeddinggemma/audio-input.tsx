import { useEffect, useRef, useState } from "react";

import { ErrorAlert } from "@/components/error-alert";
import { Field } from "@/components/field";
import { Input } from "@/components/ui/input";

import { audioSummary, decodeAudioFile, type LoadedAudio } from "./audio";

/**
 * An audio file field of a row: the browser decodes the file, resamples it to `sampleRate` and mixes
 * it to mono (`./audio.ts`); `onAudio` gets the result, which the page keeps (`audio`) so the summary
 * survives a re-render, or `undefined` when the file does not decode (the row then holds no audio
 * rather than the one before, which the error would otherwise sit beside). The file input is emptied
 * whenever the row comes to hold no audio and after a failed decode, so the same file can be chosen
 * again after "Remove audio" or a failure (choosing the file an input already holds fires no
 * change).
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
  onAudio: (audio: LoadedAudio | undefined) => void;
  disabled?: boolean;
}) => {
  const [error, setError] = useState<unknown>(undefined);
  const [decoding, setDecoding] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const emptyInput = (): void => {
    if (input.current !== null) input.current.value = "";
  };
  useEffect(() => {
    if (audio === undefined && input.current !== null) input.current.value = "";
  }, [audio]);
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
          ref={input}
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
                // The row may hold no audio already, so the effect above would not run again.
                emptyInput();
                onAudio(undefined);
              },
            );
          }}
        />
      </Field>
      {error === undefined ? null : <ErrorAlert title="Audio" error={error} />}
    </div>
  );
};
