/**
 * The name map: data that says which tensor of a weights file is which initializer of which
 * component, and its interpreter.
 *
 * A map lists the layouts a model's files exist in upstream. A layout is recognised from the
 * file's keys (a prefix every key carries, and marker keys); its components each take the keys
 * that start with their route prefix and turn each one into an initializer name with an ordered
 * rule list. The map is plain JSON (no function): the recipe that exports a model writes it beside
 * the recipe, where both names are known.
 *
 * SPEC: docs/spec/convert.md#the-name-map
 */

import { ConvertError } from "./errors.ts";

/** The format tag of a name map. */
export const NAME_MAP_FORMAT = "karume-name-map/1";

/** One rule of a component's ordered list, applied to a key as the rules before it left it. */
export type NameMapRule =
  /** Removes this prefix when the key starts with it. */
  | { readonly stripPrefix: string }
  /** Replaces every occurrence of `replace`, left to right (Python's `str.replace`). */
  | { readonly replace: string; readonly with: string }
  /** Takes the key out when it contains this text: the graph does not use that tensor. */
  | { readonly exclude: string };

/** The keys of one component and how each becomes an initializer name. */
export type NameMapComponent = {
  /** The component's role in the distribution (`transformer`, `text_conditioner`). */
  readonly role: string;
  /** Where this rule list comes from (the recipe's lines, the upstream function). */
  readonly source: string;
  /**
   * The keys of this component: those that start with this prefix (`""` takes every key). A key
   * goes to the component whose route is the longest prefix it starts with.
   */
  readonly route: string;
  /** Applied in order to each key routed here; the order is part of the data. */
  readonly rules: readonly NameMapRule[];
  /** Put before what the rules leave: the initializer name. */
  readonly initializerPrefix: string;
};

/** One layout a model's file exists in upstream. */
export type NameMapLayout = {
  readonly name: string;
  /** Where the prefixes and the markers come from. */
  readonly source: string;
  /**
   * Prefixes one of which every key of the file starts with — the same one for every key — and
   * which is removed from every key first. Empty: the keys are taken as they are.
   */
  readonly stripPrefix: readonly string[];
  /** Keys, after that prefix is removed, a file of this layout holds (all of them). */
  readonly markers: readonly string[];
  readonly components: readonly NameMapComponent[];
};

/** A name map: every layout a model's files exist in. */
export type NameMap = {
  /** {@link NAME_MAP_FORMAT}. */
  readonly format: string;
  readonly layouts: readonly NameMapLayout[];
};

/** Where every key of a file goes. */
export type TensorPlacement = {
  /** The name of the layout that recognised the file. */
  readonly layout: string;
  /**
   * Per component role of the layout (every role, possibly empty): initializer name → file key, in
   * file order — the `tensorName` mapping of `openConvertingSource`. The component's tensor list is
   * the file's tensors under these initializer names (then no `tensorName` is needed, and an
   * initializer no key is placed at is reported as missing, never read from a key of its name).
   */
  readonly components: ReadonlyMap<string, ReadonlyMap<string, string>>;
  /** The keys an exclusion rule took out, in file order. */
  readonly excluded: readonly string[];
  /** The keys no component's route takes, in file order. */
  readonly unplaced: readonly string[];
};

const quoted = (values: readonly string[]): string =>
  values.map((value) => `'${value}'`).join(", ");

/** The file's keys with the layout's prefix removed, or why the layout does not recognise them. */
const recognise = (
  layout: NameMapLayout,
  names: readonly string[],
): { readonly keys: readonly string[] } | { readonly reason: string } => {
  let keys = names;
  if (layout.stripPrefix.length > 0) {
    const prefix = layout.stripPrefix.find((candidate) =>
      names.every((name) => name.startsWith(candidate)),
    );
    if (prefix === undefined) {
      return { reason: `not every key starts with the same one of ${quoted(layout.stripPrefix)}` };
    }
    keys = names.map((name) => name.slice(prefix.length));
  }
  const present = new Set(keys);
  const missing = layout.markers.filter((marker) => !present.has(marker));
  return missing.length > 0 ? { reason: `the file has no key ${quoted(missing)}` } : { keys };
};

/**
 * The indexes of the components whose route is the longest prefix of `key` (more than one:
 * ambiguous).
 */
const routeOf = (components: readonly NameMapComponent[], key: string): readonly number[] => {
  let best: number[] = [];
  let length = -1;
  for (const [index, { route }] of components.entries()) {
    if (!key.startsWith(route)) continue;
    if (route.length > length) {
      best = [index];
      length = route.length;
    } else if (route.length === length) {
      best.push(index);
    }
  }
  return best;
};

/** The initializer name of `key`, or `undefined` when a rule excludes it. */
const initializerOf = (component: NameMapComponent, key: string): string | undefined => {
  let name = key;
  for (const rule of component.rules) {
    if ("stripPrefix" in rule) {
      if (name.startsWith(rule.stripPrefix)) name = name.slice(rule.stripPrefix.length);
    } else if ("replace" in rule) {
      // `split` / `join` and not `replaceAll`: the replacement is taken literally (no `$` patterns).
      name = name.split(rule.replace).join(rule.with);
    } else if (name.includes(rule.exclude)) {
      return undefined;
    }
  }
  return component.initializerPrefix + name;
};

/**
 * Places a file's tensors: the layout that recognises the file's keys, and per component the
 * initializer name each key becomes. Refused, with every problem listed together: a file no layout
 * (or more than one) recognises, two keys that map to one initializer of a component, and a key
 * the routes leave between two components.
 *
 * SPEC: docs/spec/convert.md#placing-a-files-tensors
 */
export const placeTensors = (map: NameMap, keys: Iterable<string>): TensorPlacement => {
  if (map.format !== NAME_MAP_FORMAT) {
    throw new ConvertError(
      `the name map's format is ${JSON.stringify(map.format)}, not ${JSON.stringify(NAME_MAP_FORMAT)}`,
    );
  }
  const names = [...keys];
  const recognised: { readonly layout: NameMapLayout; readonly keys: readonly string[] }[] = [];
  const reasons: string[] = [];
  for (const layout of map.layouts) {
    const result = recognise(layout, names);
    if ("keys" in result) recognised.push({ layout, keys: result.keys });
    else reasons.push(`layout '${layout.name}': ${result.reason}`);
  }
  const [found] = recognised;
  if (found === undefined) {
    throw new ConvertError(
      `no layout of the name map recognises the file (${names.length} keys):\n- ${reasons.join("\n- ")}`,
    );
  }
  if (recognised.length > 1) {
    throw new ConvertError(
      `more than one layout of the name map recognises the file: ${quoted(
        recognised.map(({ layout }) => layout.name),
      )}`,
    );
  }

  const { layout } = found;
  const components = new Map<string, Map<string, string>>();
  // Per component of the layout, the map of its role (two components of one role share it).
  const placedOf = layout.components.map(({ role }) => {
    const placed = components.get(role) ?? new Map<string, string>();
    components.set(role, placed);
    return placed;
  });
  const excluded: string[] = [];
  const unplaced: string[] = [];
  const problems: string[] = [];
  for (const [index, name] of names.entries()) {
    const routed = routeOf(layout.components, found.keys[index]);
    if (routed.length === 0) {
      unplaced.push(name);
      continue;
    }
    if (routed.length > 1) {
      problems.push(
        `key '${name}' is routed to more than one component (${quoted(
          routed.map((at) => layout.components[at].role),
        )})`,
      );
      continue;
    }
    const [at] = routed;
    const component = layout.components[at];
    const initializer = initializerOf(component, found.keys[index]);
    if (initializer === undefined) {
      excluded.push(name);
      continue;
    }
    const placed = placedOf[at];
    const other = placed.get(initializer);
    if (other !== undefined) {
      problems.push(
        `keys '${other}' and '${name}' both map to initializer '${initializer}' of component '${component.role}'`,
      );
      continue;
    }
    placed.set(initializer, name);
  }
  if (problems.length > 0) {
    throw new ConvertError(
      `the name map cannot place the file's tensors (layout '${layout.name}'):\n- ${problems.join("\n- ")}`,
    );
  }
  return { layout: layout.name, components, excluded, unplaced };
};
