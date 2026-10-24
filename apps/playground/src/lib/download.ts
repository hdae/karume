/** Browser-side saving: hand a Blob to the user as a download. */

export const downloadBlob = (blob: Blob, filename: string): void => {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  // The click starts the download synchronously; revoke on the next task.
  setTimeout(() => URL.revokeObjectURL(url), 0);
};

export const downloadJson = (value: unknown, filename: string): void => {
  downloadBlob(
    new Blob([`${JSON.stringify(value, null, 2)}\n`], { type: "application/json" }),
    filename,
  );
};

/** `2026-10-24T06-30-12` — a timestamp that is safe in file names on every OS. */
export const fileStamp = (at: Date): string => at.toISOString().slice(0, 19).replaceAll(":", "-");
