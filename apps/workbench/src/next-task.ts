/**
 * The turn of the event loop an entry hands back between measured spans, so a browser paints a
 * status line outside them.
 */

/**
 * One turn of the event loop (a macrotask): a browser may render before it ends.
 *
 * The task is a message, not a timer: a browser throttles the timers of a page that is not shown
 * (a display that went to sleep counts) to one a second or one a minute, and a sweep that hands a
 * turn back between its kernels then spends hours waiting. A cancel arrives as a click here, never
 * as a timer, so nothing depends on the order of this task and a timer's.
 */
export const nextTask = (): Promise<void> =>
  new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => {
      // A port left open keeps a Node process alive
      channel.port1.close();
      resolve();
    };
    channel.port2.postMessage(undefined);
  });
