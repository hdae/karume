/**
 * The turn of the event loop an entry hands back between measured spans, so a browser paints a
 * status line outside them.
 */

/** One turn of the event loop (a macrotask): a browser may render before it ends. */
export const nextTask = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
