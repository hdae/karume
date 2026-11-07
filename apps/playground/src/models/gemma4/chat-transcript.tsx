import type { Rgb8Image } from "@karume/models/gemma";
import { memo, useEffect, useRef, useSyncExternalStore } from "react";

import { Bubble, BubbleContent } from "@/components/ui/bubble";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Message, MessageContent, MessageFooter, MessageHeader } from "@/components/ui/message";
import {
  MessageScroller,
  MessageScrollerButton,
  MessageScrollerContent,
  MessageScrollerItem,
  MessageScrollerProvider,
  MessageScrollerViewport,
} from "@/components/ui/message-scroller";
import { Spinner } from "@/components/ui/spinner";

import type { TurnRecord } from "./chat-record";
import { thumbnailOf } from "./image-input";
import type { LiveReplyStore } from "./live-reply";
import { type TranscriptEntry, turnSummary } from "./turns";

/** The longest side of a thumbnail on the canvas (CSS pixels stay smaller, for a sharp draw). */
const THUMBNAIL_SIDE = 256;

/** One image as a small canvas: the composer's attachments and a user turn's images. */
export const ImageThumbnail = ({ image }: { image: Rgb8Image }) => {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const target = canvas.current;
    const context = target?.getContext("2d");
    if (target === null || context === null || context === undefined) return undefined;
    const { data, width, height } = thumbnailOf(image, THUMBNAIL_SIDE);
    target.width = width;
    target.height = height;
    context.putImageData(new ImageData(data, width, height), 0, 0);
    return undefined;
  }, [image]);
  return (
    <canvas
      ref={canvas}
      role="img"
      aria-label={`Image, ${image.width}×${image.height}`}
      className="max-h-32 max-w-48 rounded-md border"
    />
  );
};

/**
 * The entries that carry their turn's footer: the last assistant entry of each turn (a turn that
 * wrote text before its calls, or continued after tool results, has several).
 */
const footerEntries = (entries: readonly TranscriptEntry[]): ReadonlySet<number> => {
  const last = new Map<number, number>();
  for (const entry of entries) {
    if (entry.role === "assistant" && entry.turn !== undefined) last.set(entry.turn, entry.id);
  }
  return new Set(last.values());
};

/**
 * An entry's bubbles: a tool result (labelled with its tool), or the text with its thought
 * (collapsed) above it and its tool calls (name and JSON arguments) below it.
 */
const EntryBubbles = ({ entry }: { entry: TranscriptEntry }) => {
  if (entry.role === "tool") {
    return (
      <>
        <MessageHeader>Result of {entry.toolName ?? "a tool call"}</MessageHeader>
        <Bubble variant="outline">
          <BubbleContent className="font-mono text-xs whitespace-pre-wrap">
            {entry.text}
          </BubbleContent>
        </Bubble>
      </>
    );
  }
  const calls = entry.toolCalls ?? [];
  const images = entry.images ?? [];
  return (
    <>
      {images.length === 0 ? null : (
        <div className="flex flex-wrap justify-end gap-2">
          {images.map((image, index) => (
            <ImageThumbnail key={index} image={image} />
          ))}
        </div>
      )}
      {entry.thinking === undefined ? null : (
        <details className="px-3 text-xs text-muted-foreground">
          <summary className="cursor-pointer select-none">Thinking</summary>
          <p className="mt-1 whitespace-pre-wrap">{entry.thinking}</p>
        </details>
      )}
      {entry.text === "" && (calls.length > 0 || images.length > 0) ? null : (
        <Bubble variant={entry.role === "user" ? "default" : "muted"}>
          <BubbleContent className="whitespace-pre-wrap">
            {entry.text === "" ? (
              <span className="text-muted-foreground italic">(empty reply)</span>
            ) : (
              entry.text
            )}
          </BubbleContent>
        </Bubble>
      )}
      {calls.length === 0 ? null : (
        <MessageHeader>
          {calls.length === 1 ? "Tool call" : `${calls.length} tool calls`}
        </MessageHeader>
      )}
      {calls.map((call, index) => (
        <Bubble key={call.id ?? index} variant="outline">
          <BubbleContent className="font-mono text-xs whitespace-pre-wrap">
            <span className="font-semibold">{call.name}</span> {call.arguments}
          </BubbleContent>
        </Bubble>
      ))}
    </>
  );
};

/** The footer of an assistant entry: how the turn stopped and its figures. */
const footerOf = (record: TurnRecord | undefined): string | undefined => {
  if (record === undefined) return undefined;
  const stop = record.outcome.kind === "stopped" ? record.outcome.stop : undefined;
  return turnSummary(stop, record.prefill.promptTokens, record.timing);
};

/**
 * The reply of the running turn. The only part that re-renders while tokens arrive: it subscribes to
 * the store, which flushes at most once per animation frame.
 */
const LiveReplyBubble = ({ live }: { live: LiveReplyStore }) => {
  const { text: streaming, stage } = useSyncExternalStore(live.subscribe, live.snapshot);
  return (
    <MessageScrollerItem messageId="streaming">
      <Message align="start">
        <MessageContent>
          <Bubble variant="muted">
            <BubbleContent className="whitespace-pre-wrap">
              {streaming === "" ? (
                <span className="flex items-center gap-2 text-muted-foreground">
                  <Spinner />
                  {stage ?? "waiting for the first token…"}
                </span>
              ) : (
                streaming
              )}
            </BubbleContent>
          </Bubble>
          {streaming === "" || stage === undefined ? null : (
            <MessageFooter className="font-mono font-normal tabular-nums">{stage}</MessageFooter>
          )}
        </MessageContent>
      </Message>
    </MessageScrollerItem>
  );
};

/**
 * The conversation: one bubble per history entry (entries the overflow policy dropped are dimmed and
 * say so; tool calls and results are labelled, a thought is collapsed above its answer), the
 * streaming reply while a turn runs, and a scroll-to-end button. The scroller follows
 * new text while the view is at the end. Memoized: its props change only when a turn starts or
 * settles, so the page's other re-renders (the status ticker) skip it.
 */
const ChatTranscriptView = ({
  entries,
  turns,
  running,
  pending,
  pendingImages,
  live,
}: {
  entries: readonly TranscriptEntry[];
  turns: readonly TurnRecord[];
  running: boolean;
  /** The message of the running turn (not in the history until the turn settles). */
  pending: string | undefined;
  /** The images of that message. */
  pendingImages: readonly Rgb8Image[];
  /** The reply streamed so far in the running turn. */
  live: LiveReplyStore;
}) => {
  const withFooter = footerEntries(entries);
  return (
    <MessageScrollerProvider autoScroll>
      <MessageScroller id="gemma4-transcript" className="h-[28rem] rounded-lg border">
        <MessageScrollerViewport>
          <MessageScrollerContent className="p-4">
            {entries.length === 0 && !running ? (
              <Empty>
                <EmptyHeader>
                  <EmptyTitle>No messages yet</EmptyTitle>
                  <EmptyDescription>
                    Load a model, then send a message. Everything runs on this browser's GPU.
                  </EmptyDescription>
                </EmptyHeader>
              </Empty>
            ) : null}
            {entries.map((entry) => {
              const footer =
                withFooter.has(entry.id) && entry.turn !== undefined
                  ? footerOf(turns[entry.turn])
                  : undefined;
              return (
                <MessageScrollerItem
                  key={entry.id}
                  messageId={String(entry.id)}
                  scrollAnchor={entry.role === "user"}
                >
                  <Message
                    align={entry.role === "user" ? "end" : "start"}
                    className={entry.inContext ? undefined : "opacity-50"}
                  >
                    <MessageContent>
                      <EntryBubbles entry={entry} />
                      {footer === undefined && entry.inContext ? null : (
                        <MessageFooter className="flex-wrap gap-x-2 font-mono font-normal tabular-nums">
                          {entry.inContext ? null : <span>dropped from the context ·</span>}
                          {footer === undefined ? null : <span>{footer}</span>}
                        </MessageFooter>
                      )}
                    </MessageContent>
                  </Message>
                </MessageScrollerItem>
              );
            })}
            {running && pending !== undefined ? (
              <MessageScrollerItem messageId="pending" scrollAnchor>
                <Message align="end">
                  <MessageContent>
                    {pendingImages.length === 0 ? null : (
                      <div className="flex flex-wrap justify-end gap-2">
                        {pendingImages.map((image, index) => (
                          <ImageThumbnail key={index} image={image} />
                        ))}
                      </div>
                    )}
                    {pending === "" ? null : (
                      <Bubble variant="default">
                        <BubbleContent className="whitespace-pre-wrap">{pending}</BubbleContent>
                      </Bubble>
                    )}
                  </MessageContent>
                </Message>
              </MessageScrollerItem>
            ) : null}
            {running ? <LiveReplyBubble live={live} /> : null}
          </MessageScrollerContent>
        </MessageScrollerViewport>
        <MessageScrollerButton />
      </MessageScroller>
    </MessageScrollerProvider>
  );
};

export const ChatTranscript = memo(ChatTranscriptView);
