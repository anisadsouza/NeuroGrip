import { COMMIT_COST, GESTURE_LABEL, UNKNOWN_COMMIT_COST } from '../decode/commitCost.js';

export const CARTOGRAPHY_FUNCTIONS = Object.keys(GESTURE_LABEL).filter(
  (gesture) => gesture !== 'rest',
);

export interface PrototypeFunctionAssignment {
  prototype: string;
  function: string;
  misclassificationRisk: number;
  functionCriticality: number;
  expectedCommitCost: number;
}

/**
 * Assign prototypes to distinct prosthetic functions, minimizing the sum of
 * misclassification probability times the function's commit cost.
 */
export function assignFunctionsByRisk(
  prototypes: readonly string[],
  misclassificationRisks: readonly number[],
  functions: readonly string[] = CARTOGRAPHY_FUNCTIONS,
): PrototypeFunctionAssignment[] {
  if (prototypes.length !== misclassificationRisks.length) {
    throw new RangeError('each prototype must have one misclassification risk');
  }
  if (prototypes.length === 0 || functions.length < prototypes.length) {
    throw new RangeError('assignment needs prototypes and at least as many functions');
  }
  if (new Set(prototypes).size !== prototypes.length) {
    throw new RangeError('prototype names must be unique');
  }
  if (new Set(functions).size !== functions.length) {
    throw new RangeError('function names must be unique');
  }
  if (
    misclassificationRisks.some(
      (risk) => !Number.isFinite(risk) || risk < 0 || risk > 1,
    )
  ) {
    throw new RangeError('misclassification risks must be finite values from 0 to 1');
  }

  const costs = misclassificationRisks.map((risk) =>
    functions.map((name) => risk * (COMMIT_COST[name] ?? UNKNOWN_COMMIT_COST)),
  );
  const rowCount = costs.length;
  const columnCount = costs[0]!.length;
  const rowPotential = new Array<number>(rowCount + 1).fill(0);
  const columnPotential = new Array<number>(columnCount + 1).fill(0);
  const matchedRow = new Array<number>(columnCount + 1).fill(0);
  const previousColumn = new Array<number>(columnCount + 1).fill(0);

  for (let row = 1; row <= rowCount; row++) {
    matchedRow[0] = row;
    let currentColumn = 0;
    const minCost = new Array<number>(columnCount + 1).fill(Number.POSITIVE_INFINITY);
    const visited = new Array<boolean>(columnCount + 1).fill(false);

    do {
      visited[currentColumn] = true;
      const currentRow = matchedRow[currentColumn]!;
      let delta = Number.POSITIVE_INFINITY;
      let nextColumn = 0;

      for (let column = 1; column <= columnCount; column++) {
        if (visited[column]) continue;
        const reducedCost =
          costs[currentRow - 1]![column - 1]! -
          rowPotential[currentRow]! -
          columnPotential[column]!;
        if (reducedCost < minCost[column]!) {
          minCost[column] = reducedCost;
          previousColumn[column] = currentColumn;
        }
        if (minCost[column]! < delta) {
          delta = minCost[column]!;
          nextColumn = column;
        }
      }

      for (let column = 0; column <= columnCount; column++) {
        if (visited[column]) {
          const assignedRow = matchedRow[column]!;
          rowPotential[assignedRow] = rowPotential[assignedRow]! + delta;
          columnPotential[column] = columnPotential[column]! - delta;
        } else {
          minCost[column] = minCost[column]! - delta;
        }
      }
      currentColumn = nextColumn;
    } while (matchedRow[currentColumn] !== 0);

    do {
      const priorColumn = previousColumn[currentColumn]!;
      matchedRow[currentColumn] = matchedRow[priorColumn]!;
      currentColumn = priorColumn;
    } while (currentColumn !== 0);
  }

  const assignedColumn = new Array<number>(rowCount).fill(-1);
  for (let column = 1; column <= columnCount; column++) {
    if (matchedRow[column] !== 0) {
      assignedColumn[matchedRow[column]! - 1] = column - 1;
    }
  }

  return prototypes.map((prototype, index) => {
    const functionIndex = assignedColumn[index]!;
    if (functionIndex < 0) {
      throw new Error(`no function was assigned to prototype ${prototype}`);
    }
    const name = functions[functionIndex]!;
    const risk = misclassificationRisks[index]!;
    const criticality = COMMIT_COST[name] ?? UNKNOWN_COMMIT_COST;
    return {
      prototype,
      function: name,
      misclassificationRisk: risk,
      functionCriticality: criticality,
      expectedCommitCost: risk * criticality,
    };
  });
}
