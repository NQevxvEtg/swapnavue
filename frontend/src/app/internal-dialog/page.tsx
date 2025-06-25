'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { useTraining } from '../context/TrainingContext';

// Define the shape of an internal thought message
interface InternalThought {
  thought: string;
  timestamp: string;
  confidence: number;
  meta_error: number;
  focus: number;
  curiosity: number;
  prompt_text: string;
}

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL;
const POLLING_INTERVAL = 3000;

export default function InternalDialogPage() {
  const [thoughts, setThoughts] = useState<InternalThought[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const { trainingStatus } = useTraining();
  const thoughtsEndRef = useRef<HTMLDivElement>(null);

  const fetchInternalThoughts = useCallback(async () => {
    if (trainingStatus?.is_training_active) {
      console.log("Internal dialog paused: Training is active.");
      setIsLoading(false);
      return;
    }

    try {
      if (!BACKEND_API_URL) {
        throw new Error("BACKEND_API_URL is not defined. Check .env.local file.");
      }
      const res = await fetch(`${BACKEND_API_URL}/internal_thought`);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data: InternalThought[] = await res.json();
      if (data.length !== thoughts.length) {
        setThoughts(data);
      }
    } catch (error) {
      console.error('Error fetching internal thoughts:', error);
    } finally {
      setIsLoading(false);
    }
  }, [trainingStatus?.is_training_active, thoughts]);

  useEffect(() => {
    fetchInternalThoughts();
    const interval = setInterval(fetchInternalThoughts, POLLING_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchInternalThoughts]);

  useEffect(() => {
    thoughtsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [thoughts]);

  return (
    <main style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'flex-start',
      minHeight: '100%',
      padding: '2rem',
      fontFamily: 'Arial, sans-serif',
    }}>
      <h1 style={{ marginBottom: '1.5rem' }}>swapnavue&apos;s Internal Dialog</h1>
      <p style={{ marginBottom: '2rem', color: 'var(--foreground-subtle)', textAlign: 'center' }}>
        Observe swapnavue&apos;s continuous internal reflections and mental state changes.
      </p>

      {trainingStatus?.is_training_active && (
        <div style={{
          backgroundColor: 'var(--card-background)',
          color: 'var(--foreground)',
          border: '1px solid var(--card-border)',
          borderRadius: '8px',
          padding: '1rem',
          marginBottom: '1rem',
          width: '100%',
          maxWidth: '900px',
          textAlign: 'center'
        }}>
          Internal dialog paused: Training is currently active.
        </div>
      )}

      <div style={{
        flexGrow: 1,
        width: '100%',
        maxWidth: '900px',
        backgroundColor: 'var(--card-background)',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05)',
        padding: '1.5rem',
        marginBottom: '1.5rem',
        display: 'flex',
        flexDirection: 'column',
        border: '1px solid var(--card-border)'
      }}>
        <div style={{ overflowY: 'auto', maxHeight: '70vh', paddingRight: '1rem' }}>
          {isLoading && thoughts.length === 0 ? (
            <p style={{ textAlign: 'center', color: 'var(--foreground-subtle)' }}>Waking swapnavue&apos;s mind... Please wait.</p>
          ) : thoughts.length === 0 && !trainingStatus?.is_training_active ? (
            <p style={{ textAlign: 'center', color: 'var(--foreground-subtle)' }}>No internal thoughts yet. Ensure backend is running and generating thoughts.</p>
          ) : (
            thoughts.map((thought, index) => (
              <div key={index} style={{
                backgroundColor: 'var(--chat-bubble-swapnavue-bg)', // Using a consistent bubble color
                borderRadius: '10px',
                padding: '1rem',
                marginBottom: '1rem',
                boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
                wordWrap: 'break-word',
                borderLeft: '4px solid var(--button-primary-bg)',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.5rem'
              }}>
                <div style={{ fontSize: '0.9rem', color: 'var(--foreground-subtle)', display: 'flex', justifyContent: 'space-between' }}>
                  <span>{new Date(thought.timestamp).toLocaleString()}</span>
                  <span style={{ fontWeight: 'bold', color: 'var(--nav-link-color)' }}>Internal Reflection</span>
                </div>
                <div style={{ fontSize: '1rem', lineHeight: '1.4' }}>
                  {thought.thought}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--foreground-subtle)', marginTop: '0.5rem' }}>
                  Prompt: &quot;{thought.prompt_text}&quot;<br/>
                  Confidence: {thought.confidence.toFixed(4)}, Meta-Error: {thought.meta_error.toFixed(4)}, Focus: {thought.focus}, Curiosity: {thought.curiosity.toFixed(6)}
                </div>
              </div>
            ))
          )}
          <div ref={thoughtsEndRef} />
        </div>
      </div>
    </main>
  );
}