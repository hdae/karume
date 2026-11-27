/**
 * The page's DOM helpers: an element with properties and children, a text that is set only when it
 * changes (the page paints on every frame while a run goes), and the download of a text as a file.
 */

export const element = <Tag extends keyof HTMLElementTagNameMap>(
  tag: Tag,
  properties: Partial<HTMLElementTagNameMap[Tag]> = {},
  children: readonly (Node | string)[] = [],
): HTMLElementTagNameMap[Tag] => {
  const node = document.createElement(tag);
  Object.assign(node, properties);
  node.append(...children);
  return node;
};

/** Sets `text` as the node's text unless it already is. */
export const setText = (node: HTMLElement, text: string): void => {
  if (node.textContent !== text) node.textContent = text;
};

/** Hands `text` to the browser as a download named `fileName` (Blob + `a[download]`). */
export const downloadText = (fileName: string, text: string): void => {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const anchor = element("a", { href: url, download: fileName });
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  // The click starts the download synchronously; revoke on the next task.
  setTimeout(() => URL.revokeObjectURL(url), 0);
};
