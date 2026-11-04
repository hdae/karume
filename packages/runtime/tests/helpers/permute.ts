/**
 * Independent oracle mapping output coordinates back to input coordinates. It does not use an
 * element-order identity check.
 */
export const permutedSourceIndices = (
  shape: readonly number[],
  axes: readonly number[],
): number[] => {
  const outputShape = axes.map((axis) => shape[axis]);
  const count = shape.reduce((a, b) => a * b, 1);
  return Array.from({ length: count }, (_, index) => {
    const coordinates = new Array<number>(shape.length);
    let rest = index;
    for (let axis = shape.length - 1; axis >= 0; axis--) {
      coordinates[axes[axis]] = rest % outputShape[axis];
      rest = Math.floor(rest / outputShape[axis]);
    }
    return coordinates.reduce((offset, coordinate, axis) => offset * shape[axis] + coordinate, 0);
  });
};
