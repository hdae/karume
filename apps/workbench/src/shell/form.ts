/**
 * The parameter form of an entry, generated from its declaration. It reads back as the same string
 * map a URL query or the command line gives, so `parseParameters` is the only parser.
 */

import {
  type Parameter,
  parameterText,
  type ParameterValue,
  type ParameterValues,
} from "../entry.ts";
import { element } from "./dom.ts";

export type ParameterForm = {
  readonly element: HTMLElement;
  readonly read: () => Record<string, string>;
};

type Field = { readonly element: HTMLElement; readonly read: () => string };

const field = (parameter: Parameter, value: ParameterValue): Field => {
  switch (parameter.kind) {
    case "boolean": {
      const input = element("input", { type: "checkbox", checked: value === true });
      return {
        element: element("label", {}, [input, ` ${parameter.label}`]),
        read: () => String(input.checked),
      };
    }
    case "integer": {
      const input = element("input", {
        type: "number",
        step: "1",
        min: String(parameter.min),
        max: String(parameter.max),
        value: parameterText(value),
      });
      return {
        element: element("label", {}, [`${parameter.label} `, input]),
        read: () => input.value,
      };
    }
    case "choice": {
      const select = element(
        "select",
        {},
        parameter.options.map((option) => element("option", { value: option }, [option])),
      );
      select.value = parameterText(value);
      return {
        element: element("label", {}, [`${parameter.label} `, select]),
        read: () => select.value,
      };
    }
    case "choices": {
      const boxes = parameter.options.map((option) => ({
        option,
        input: element("input", {
          type: "checkbox",
          checked: typeof value === "object" && value.includes(option),
        }),
      }));
      return {
        element: element("fieldset", {}, [
          element("legend", {}, [parameter.label]),
          ...boxes.map(({ option, input }) => element("label", {}, [input, ` ${option}`])),
        ]),
        read: () =>
          boxes
            .filter(({ input }) => input.checked)
            .map(({ option }) => option)
            .join(","),
      };
    }
  }
};

export const parameterForm = (
  parameters: readonly Parameter[],
  values: ParameterValues,
): ParameterForm => {
  const fields = parameters.map((parameter) => ({
    name: parameter.name,
    ...field(parameter, values[parameter.name]),
  }));
  return {
    element: element(
      "div",
      { className: "parameters" },
      fields.map(({ element: node }) => node),
    ),
    read: () => Object.fromEntries(fields.map(({ name, read }) => [name, read()])),
  };
};
