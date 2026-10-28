import { useState } from "react";

import { ErrorAlert } from "@/components/error-alert";
import { Field } from "@/components/field";
import { RgbaImage } from "@/components/rgba-image";
import { Input } from "@/components/ui/input";
import { decodeImageFile, type LoadedImage } from "@/lib/image";

/**
 * An image file field with a preview of the decoded pixels (what the pipeline will see). `onImage`
 * gets the decoded image; the page keeps it (`image`) so the preview survives a re-render.
 */
export const ImageInput = ({
  id,
  label = "Image",
  image,
  onImage,
  disabled,
}: {
  id: string;
  label?: string;
  image: LoadedImage | undefined;
  onImage: (image: LoadedImage) => void;
  disabled?: boolean;
}) => {
  const [error, setError] = useState<unknown>(undefined);
  const [decoding, setDecoding] = useState(false);
  return (
    <div className="flex flex-col gap-3">
      <Field
        id={id}
        label={label}
        hint={
          decoding
            ? "Decoding…"
            : image === undefined
              ? "Any format the browser decodes; alpha is dropped."
              : `${image.name} · ${image.rgba.width}×${image.rgba.height}`
        }
      >
        <Input
          id={id}
          type="file"
          accept="image/*"
          disabled={(disabled ?? false) || decoding}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file === undefined) return;
            setError(undefined);
            setDecoding(true);
            decodeImageFile(file).then(
              (decoded) => {
                setDecoding(false);
                onImage(decoded);
              },
              (failure: unknown) => {
                setDecoding(false);
                setError(failure);
              },
            );
          }}
        />
      </Field>
      {error === undefined ? null : <ErrorAlert title="Image" error={error} />}
      {image === undefined ? null : <RgbaImage image={image.rgba} alt={image.name} />}
    </div>
  );
};
