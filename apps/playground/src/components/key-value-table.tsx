import type { ReactNode } from "react";

/** A two-column table of record rows. */
export const KeyValueTable = ({
  rows,
}: {
  rows: readonly (readonly [label: string, value: ReactNode])[];
}) => (
  <table className="w-full text-sm">
    <tbody>
      {rows.map(([label, value]) => (
        <tr key={label} className="border-b last:border-0">
          <th className="py-1 pr-4 text-left align-top font-medium whitespace-nowrap text-muted-foreground">
            {label}
          </th>
          <td className="py-1 font-mono text-xs break-all">{value}</td>
        </tr>
      ))}
    </tbody>
  </table>
);
