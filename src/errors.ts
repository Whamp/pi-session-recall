/** Reports that recall search could not enter the current bounded write window wait. */
export class RecallSearchBusyError extends Error {
  constructor() {
    super('Recall search is busy with a current write window; retry the search shortly');
    this.name = 'RecallSearchBusyError';
  }
}

/** Reports that an interrupted writer requires the external recovery worker. */
export class RecallRecoveryRequiredError extends Error {
  constructor() {
    super('Recall write recovery required; run pi-session-recall recover before searching again');
    this.name = 'RecallRecoveryRequiredError';
  }
}

/** Reports that the checksummed active-generation pointer cannot safely select a search directory. */
export class RecallGenerationPointerError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(
      `Recall active generation pointer unavailable: ${message}; repair with pi-session-recall recover or create a fresh generation with pi-session-recall rebuild`,
      options,
    );
    this.name = 'RecallGenerationPointerError';
  }
}
