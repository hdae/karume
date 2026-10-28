import type { ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { downloadJson } from "@/lib/download";

/**
 * **Download record (JSON)** (saves `filename` in the browser's download folder) and **Copy record**
 * (the same JSON to the clipboard). `children` (more buttons of the page, such as a media download)
 * go between the two.
 */
export const RecordActions = ({
  record,
  filename,
  children,
}: {
  record: unknown;
  filename: string;
  children?: ReactNode;
}) => (
  <div className="flex flex-wrap gap-2">
    <Button
      id="record-json"
      variant="outline"
      size="sm"
      onClick={() => downloadJson(record, filename)}
    >
      Download record (JSON)
    </Button>
    {children}
    <Button
      id="record-copy"
      variant="outline"
      size="sm"
      onClick={() => void navigator.clipboard.writeText(JSON.stringify(record, null, 2))}
    >
      Copy record
    </Button>
  </div>
);
