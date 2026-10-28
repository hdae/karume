import { memo, useSyncExternalStore } from "react";

import { Bubble, BubbleContent } from "@/components/ui/bubble";
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from "@/components/ui/empty";
import { Message, MessageContent, MessageFooter } from "@/components/ui/message";
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
import type { LiveReplyStore } from "./live-reply";
import { type TranscriptEntry, turnSummary } from "./turns";

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
 * say so), the streaming reply while a turn runs, and a scroll-to-end button. The scroller follows
 * new text while the view is at the end. Memoized: its props change only when a turn starts or
 * settles, so the page's other re-renders (the status ticker) skip it.
 */
const ChatTranscriptView = ({
  entries,
  turns,
  running,
  pending,
  live,
}: {
  entries: readonly TranscriptEntry[];
  turns: readonly TurnRecord[];
  running: boolean;
  /** The message of the running turn (not in the history until the turn settles). */
  pending: string | undefined;
  /** The reply streamed so far in the running turn. */
  live: LiveReplyStore;
}) => {
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
                entry.role === "assistant" && entry.turn !== undefined
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
                      <Bubble variant={entry.role === "user" ? "default" : "muted"}>
                        <BubbleContent className="whitespace-pre-wrap">
                          {entry.text === "" ? (
                            <span className="text-muted-foreground italic">(empty reply)</span>
                          ) : (
                            entry.text
                          )}
                        </BubbleContent>
                      </Bubble>
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
                    <Bubble variant="default">
                      <BubbleContent className="whitespace-pre-wrap">{pending}</BubbleContent>
                    </Bubble>
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
