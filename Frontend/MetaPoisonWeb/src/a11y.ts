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

export function generateUniqueId(prefix: string = "id"): string {
  return `${prefix}-${Math.random().toString(36).substr(2, 9)}`;
}

export function focusElement(element: HTMLElement | null, announce?: string) {
  if (element) {
    element.focus();
    if (announce) {
      announceToScreenReader(announce);
    }
  }
}

export function formatNumberForA11y(num: number): string {
  return Number.isFinite(num) ? num.toLocaleString() : "-";
}

export function formatPercentageForA11y(percentage: number): string {
  return `${(percentage * 100).toFixed(2)} percent`;
}

export function createLabelPair(baseId: string) {
  return {
    labelId: `${baseId}-label`,
    inputId: baseId,
    descId: `${baseId}-desc`,
  };
}

export function isActivationKey(event: React.KeyboardEvent): boolean {
  return event.key === "Enter" || event.key === " ";
}

export function isEscapeKey(event: React.KeyboardEvent): boolean {
  return event.key === "Escape";
}


export function getContrastLevel(foreground: string, background: string): "AAA" | "AA" | "fail" {
  // basic WCAG contrast ratio calculation for hex colors
  function luminance(hex: string) {
    let c = hex.replace("#", "");
    if (c.length === 3) c = c.split("").map((x) => x + x).join("");
    const rgb = [0, 1, 2].map((i) => parseInt(c.substr(i * 2, 2), 16) / 255);
    const a = rgb.map((v) => {
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2];
  }

  try {
    const L1 = luminance(foreground);
    const L2 = luminance(background);
    const ratio = (Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05);
    if (ratio >= 7) return "AAA";
    if (ratio >= 4.5) return "AA";
    return "fail";
  } catch {
    return "AA"; // fallback
  }
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
