/**
 * One run at a time: while a run holds the lock, its panel's Run and the entry menu are off and its
 * Cancel is on. The menu stays off for the whole run, so the panel that took the lock is the one on
 * the page when it is released.
 */

export type RunButtons = {
  readonly run: HTMLButtonElement;
  readonly cancel: HTMLButtonElement;
};

export type RunLock = {
  readonly running: () => boolean;
  /** Takes the lock for a run started from `buttons`. */
  readonly take: (buttons: RunButtons) => void;
  /** Gives it back: Run and the menu on, Cancel off. */
  readonly release: () => void;
};

export const createRunLock = (menu: readonly HTMLButtonElement[]): RunLock => {
  let held: RunButtons | undefined;
  const show = ({ run, cancel }: RunButtons, running: boolean): void => {
    run.disabled = running;
    cancel.disabled = !running;
    for (const button of menu) button.disabled = running;
  };
  return {
    running: () => held !== undefined,
    take: (buttons) => {
      held = buttons;
      show(buttons, true);
    },
    release: () => {
      if (held === undefined) return;
      show(held, false);
      held = undefined;
    },
  };
};
