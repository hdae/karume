/** The one DOM helper of the shell: an element with properties and children. */

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
