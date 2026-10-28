import { useEffect, useRef, useState } from "react";

import { ErrorAlert } from "@/components/error-alert";
import { Button } from "@/components/ui/button";
import { downloadBlob } from "@/lib/download";
import type { RgbaPixels } from "@/lib/image";

/**
 * Draws RGBA pixels in a canvas (scaled down to fit, never up), with **Download PNG** when `filename`
 * is given (lossless, from the same bytes; the PNG encoder of the models barrel loads on the click).
 *
 * MUST: pass `filename` only for an opaque image. `encodePng` writes RGB and refuses any pixel whose
 * alpha is not 255 (ADR 0107), so a translucent image (a cut-out over its matte) cannot be saved this
 * way: such a page shows it without `filename` and offers the matte, or an opaque composite, for
 * download instead. A refused encode is shown under the button, never swallowed.
 */
export const RgbaImage = ({
  image,
  filename,
  alt,
}: {
  image: RgbaPixels;
  filename?: string;
  alt: string;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [saving, setSaving] = useState(false);
  // The failure is kept with the image it belongs to, so a new image does not show a stale error.
  const [saveFailure, setSaveFailure] = useState<{ image: RgbaPixels; error: unknown }>();
  useEffect(() => {
    const context = canvasRef.current?.getContext("2d");
    if (context === null || context === undefined) return undefined;
    const pixels = new ImageData(image.width, image.height);
    pixels.data.set(image.data);
    context.putImageData(pixels, 0, 0);
    // Clear on the way out, so the old pixels do not stay up while the next image draws.
    return () => context.clearRect(0, 0, image.width, image.height);
  }, [image]);

  const save = async (name: string): Promise<void> => {
    setSaving(true);
    setSaveFailure(undefined);
    try {
      const { encodePng } = await import("@karume/models");
      const png = await encodePng(image.data, image.width, image.height);
      downloadBlob(new Blob([png], { type: "image/png" }), name);
    } catch (error: unknown) {
      setSaveFailure({ image, error });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <canvas
        ref={canvasRef}
        width={image.width}
        height={image.height}
        role="img"
        aria-label={alt}
        className="h-auto max-h-[70vh] w-auto max-w-full self-center rounded-md border bg-[repeating-conic-gradient(var(--muted)_0_25%,transparent_0_50%)] bg-size-[16px_16px]"
      />
      {filename === undefined ? null : (
        <Button
          variant="outline"
          size="sm"
          className="self-start"
          disabled={saving}
          onClick={() => void save(filename)}
        >
          {saving ? "Encoding PNG…" : "Download PNG"}
        </Button>
      )}
      {saveFailure === undefined || saveFailure.image !== image ? null : (
        <ErrorAlert title="Could not save the PNG" error={saveFailure.error} />
      )}
    </div>
  );
};
