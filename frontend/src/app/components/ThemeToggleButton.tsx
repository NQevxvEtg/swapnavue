'use client';

import { useTheme } from '../context/ThemeContext';
import React from 'react';

export const ThemeToggleButton = () => {
  const { theme, setTheme } = useTheme();

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };

  const buttonStyle: React.CSSProperties = {
    padding: '10px 20px',
    cursor: 'pointer',
    backgroundColor: 'transparent',
    border: 'none',
    color: 'var(--foreground-subtle)',
    fontSize: '1.5rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
  };

  return (
    <button onClick={toggleTheme} style={buttonStyle} aria-label="Toggle theme">
      {theme === 'light' ? '☀️' : '🌙'}
    </button>
  );
};