/**
 * The one error of the package: the inputs cannot give the container the plan asks for (a tensor
 * missing or of another shape, a codec this package does not encode, a value the target codec
 * cannot store as given). The message names the initializer.
 */
export class ConvertError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConvertError";
  }
}
