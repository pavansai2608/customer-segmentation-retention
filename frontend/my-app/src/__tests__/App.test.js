import '@testing-library/jest-dom';
import { fireEvent, render, screen } from '@testing-library/react';
import axios from 'axios';
import App from '../App';

beforeAll(() => {
  window.matchMedia = jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  }));
  window.localStorage.clear();
});

jest.mock('axios', () => ({
  get: jest.fn(() => Promise.resolve({ data: [] })),
  post: jest.fn(() => Promise.resolve({ data: { action: 'retain', action_code: 'retain', action_label: '🔴 Retain Immediately' } })),
}));

describe('App', () => {
  it('renders the dashboard title', async () => {
    render(<App />);
    expect(await screen.findByText(/Segmentation & Retention/i)).toBeInTheDocument();
  });

  it('looks up a customer when the search button is clicked', async () => {
    axios.get.mockImplementation((url) => {
      if (url.includes('/segments')) return Promise.resolve({ data: [{ segment: 'At Risk', count: 1 }] });
      if (url.includes('/actions')) return Promise.resolve({ data: [{ action: 'retain', count: 1 }] });
      if (url.includes('/retain')) return Promise.resolve({ data: [] });
      if (url.includes('/customer/42')) return Promise.resolve({ data: { 'Customer ID': 42, Segment: 'At Risk', predicted_ltv: 100, action: 'retain', action_code: 'retain', churn_probability: 0.8 } });
      return Promise.resolve({ data: [] });
    });

    render(<App />);
    const input = await screen.findByPlaceholderText(/Customer ID/i);
    fireEvent.change(input, { target: { value: '42' } });
    fireEvent.click(screen.getByRole('button', { name: /search/i }));

    expect(await screen.findByText(/At Risk/i)).toBeInTheDocument();
  });
});
