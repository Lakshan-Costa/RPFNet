/**
 * Accessibility (a11y) utilities
 * Helpers for implementing WCAG 2.1 AA standards
 */

/**
 * Announce a message to screen readers
 * @param message The message to announce
 * @param priority The priority level (polite or assertive)
 */
export function announceToScreenReader(message: string, priority: "polite" | "assertive" = "polite") {
  const ariaLiveRegion = document.getElementById("aria-live-region");
  if (ariaLiveRegion) {
    ariaLiveRegion.setAttribute("aria-live", priority);
    ariaLiveRegion.textContent = message;
  }
}

/**
 * Generate a unique ID for ARIA labels
 */
export function generateUniqueId(prefix: string = "id"): string {
  return `${prefix}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Focus management - focus on element with optional announcement
 */
export function focusElement(element: HTMLElement | null, announce?: string) {
  if (element) {
    element.focus();
    if (announce) {
      announceToScreenReader(announce);
    }
  }
}

/**
 * Format number for accessibility
 * Use spelled-out format for better screen reader experience
 */
export function formatNumberForA11y(num: number): string {
  return Number.isFinite(num) ? num.toLocaleString() : "-";
}

/**
 * Format percentage for accessibility
 */
export function formatPercentageForA11y(percentage: number): string {
  return `${(percentage * 100).toFixed(2)} percent`;
}

/**
 * Create a label context ID pair
 */
export function createLabelPair(baseId: string) {
  return {
    labelId: `${baseId}-label`,
    inputId: baseId,
    descId: `${baseId}-desc`,
  };
}

/**
 * Keyboard event handler - check if Enter or Space was pressed
 */
export function isActivationKey(event: React.KeyboardEvent): boolean {
  return event.key === "Enter" || event.key === " ";
}

/**
 * Keyboard event handler - check if Escape was pressed
 */
export function isEscapeKey(event: React.KeyboardEvent): boolean {
  return event.key === "Escape";
}

/**
 * Ensure minimum color contrast between two colors
 * Returns WCAG level (AAA, AA, or fail)
 */
export function getContrastLevel(foreground: string, background: string): "AAA" | "AA" | "fail" {
  // Simplified implementation - in production, use a library like polished
  // This is a placeholder that should be enhanced
  return "AA";
}

/**
 * Convert status to human-readable aria-label
 */
export function getStatusAriaLabel(status: string, context?: string): string {
  const contextStr = context ? ` - ${context}` : "";
  if (status.includes("Loading") || status.includes("Analyzing")) {
    return `Loading${contextStr}`;
  } else if (status.includes("Failed") || status.includes("Error")) {
    return `Error: ${status}${contextStr}`;
  } else if (status.includes("Success")) {
    return `Success: ${status}${contextStr}`;
  }
  return `Status: ${status}${contextStr}`;
}

export default {
  announceToScreenReader,
  generateUniqueId,
  focusElement,
  formatNumberForA11y,
  formatPercentageForA11y,
  createLabelPair,
  isActivationKey,
  isEscapeKey,
  getContrastLevel,
  getStatusAriaLabel,
};
