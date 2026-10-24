import { PauseIcon, PlayIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import type { RgbaStack } from "@/lib/record";

/**
 * Plays RGBA frames in a canvas at `fps`, with a scrubber and play / pause. Key it by the video
 * (a new video starts at frame 0).
 */
export const FramePlayer = ({ stack, fps }: { stack: RgbaStack; fps: number }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frame, setFrame] = useState(0);
  // The frame playback resumes from (read when playing starts, not a dependency of the loop).
  const frameRef = useRef(0);
  useEffect(() => {
    frameRef.current = frame;
  }, [frame]);
  const [playing, setPlaying] = useState(true);

  // Draw the current frame (one reusable ImageData; copying a frame is a memcpy).
  const imageRef = useRef<ImageData | undefined>(undefined);
  useEffect(() => {
    const context = canvasRef.current?.getContext("2d");
    if (context === null || context === undefined) return;
    let image = imageRef.current;
    if (image === undefined || image.width !== stack.width || image.height !== stack.height) {
      image = new ImageData(stack.width, stack.height);
      imageRef.current = image;
    }
    image.data.set(stack.frame(frame));
    context.putImageData(image, 0, 0);
  }, [stack, frame]);

  useEffect(() => {
    if (!playing) return undefined;
    let handle = 0;
    const started = performance.now();
    const first = frameRef.current;
    const tick = (now: number): void => {
      setFrame((first + Math.floor(((now - started) * fps) / 1000)) % stack.frames);
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [playing, stack, fps]);

  return (
    <div className="flex flex-col gap-3">
      <canvas
        ref={canvasRef}
        width={stack.width}
        height={stack.height}
        className="h-auto max-h-[70vh] w-auto max-w-full self-center rounded-md border bg-black"
      />
      <div className="flex items-center gap-3">
        <Button
          id="player-toggle"
          variant="outline"
          size="icon"
          aria-label={playing ? "Pause" : "Play"}
          onClick={() => setPlaying((value) => !value)}
        >
          {playing ? <PauseIcon /> : <PlayIcon />}
        </Button>
        <Slider
          id="player-scrubber"
          aria-label="Frame"
          min={0}
          max={stack.frames - 1}
          step={1}
          value={[frame]}
          onValueChange={(value) => {
            setPlaying(false);
            const next = Array.isArray(value) ? value[0] : value;
            if (typeof next === "number") setFrame(next);
          }}
        />
        <span className="w-24 text-right font-mono text-xs tabular-nums">
          {frame + 1} / {stack.frames}
        </span>
      </div>
    </div>
  );
};
