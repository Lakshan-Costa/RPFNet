import { render, screen } from '@testing-library/react';
import Dashboard, { computeMetrics, quantile } from '../Dashboard';

describe('Dashboard component', () => {
  it('renders header and upload instructions', () => {
    render(<Dashboard />);
    expect(screen.getByRole('heading', { name: /Poison Detection Dashboard/i })).toBeInTheDocument();
    expect(screen.getByText(/Upload a dataset or provide a UCI dataset ID/i)).toBeInTheDocument();
  });
});

describe('utility functions', () => {
  it('computes metrics correctly for a small labeled dataset', () => {
    const rows = [
      { score: 0.1, y_true: 0 },
      { score: 0.8, y_true: 1 },
      { score: 0.6, y_true: 1 },
      { score: 0.4, y_true: 0 },
    ];
    const thr = 0.5;
    const metrics = computeMetrics(rows, thr);

    // Manual calculation: two positives predicted (0.8,0.6), both true positives; two negatives predicted (0.1,0.4), both true negatives
    expect(metrics).not.toBeNull();
    expect(metrics?.TP).toBe(2);
    expect(metrics?.TN).toBe(2);
    expect(metrics?.FP).toBe(0);
    expect(metrics?.FN).toBe(0);
    expect(metrics?.precision).toBeCloseTo(1);
    expect(metrics?.recall).toBeCloseTo(1);
    expect(metrics?.f1).toBeCloseTo(1);
    expect(metrics?.acc).toBeCloseTo(1);
  });

  it('returns null when no labels present', () => {
    const rows = [{ score: 0.1 }, { score: 0.9 }];
    // @ts-ignore
    const metrics = computeMetrics(rows, 0.5);
    expect(metrics).toBeNull();
  });

  it('calculates quantiles correctly', () => {
    const arr = [0, 1, 2, 3, 4];
    expect(quantile(arr, 0)).toBe(0);
    expect(quantile(arr, 0.5)).toBe(2);
    expect(quantile(arr, 1)).toBe(4);
    expect(quantile(arr, 0.25)).toBe(1);
  });
});
