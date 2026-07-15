import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
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
});
