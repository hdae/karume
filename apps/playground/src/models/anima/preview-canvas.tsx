import { useEffect, useRef } from "react";

import type { RgbaPixels } from "@/lib/image";

/**
 * The live preview of a denoise step. `approximatePreview` returns RGBA at latent resolution (1/8 of
 * the image), so the canvas holds those pixels and CSS scales it up without smoothing — the blocky
 * look keeps the approximation from being mistaken for the decoded image. (`RgbaImage` never scales
 * up, so it would show the preview at 1/8 size.)
 */
export const PreviewCanvas = ({ image, alt }: { image: RgbaPixels; alt: string }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = canvasRef.current?.getContext("2d");
    if (context === null || context === undefined) return undefined;
    const pixels = new ImageData(image.width, image.height);
    pixels.data.set(image.data);
    context.putImageData(pixels, 0, 0);
    return () => context.clearRect(0, 0, image.width, image.height);
  }, [image]);
  return (
    <canvas
      ref={canvasRef}
      width={image.width}
      height={image.height}
      role="img"
      aria-label={alt}
      className="h-auto max-h-[70vh] w-full max-w-xl self-center rounded-md border [image-rendering:pixelated]"
      style={{ aspectRatio: `${image.width} / ${image.height}` }}
    />
  );
};
