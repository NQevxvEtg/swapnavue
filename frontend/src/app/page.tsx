'use client';

import { useState, FormEvent, useEffect, useRef, CSSProperties } from 'react';
import { useTraining } from './context/TrainingContext';

// Define the shape of a chat message
interface ChatMessage {
  sender: 'user' | 'swapnavue';
  text: string;
}

// Define the shape of the API response from the backend
interface swapnavueResponse {
  response: string;
  confidence: number;
  meta_error: number;
  focus: number;
  curiosity: number;
  continuous_learning_loss?: number;
}

// The unused 'SimpleTrainingStatus' interface has been removed.

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL;

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const { trainingStatus } = useTraining();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!inputPrompt.trim() || isLoading || (trainingStatus?.is_training_active ?? false)) return;

    const userMessage: ChatMessage = { sender: 'user', text: inputPrompt };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInputPrompt('');
    setIsLoading(true);

    try {
      if (!BACKEND_API_URL) {
        throw new Error("BACKEND_API_URL is not defined. Check .env.local file.");
      }

      const res = await fetch(`${BACKEND_API_URL}/generate_response`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: inputPrompt, max_length: 256 }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(`HTTP error! status: ${res.status}, detail: ${errorData.detail || res.statusText}`);
      }

      const data: swapnavueResponse = await res.json();
      const swapnavueResponse: ChatMessage = { sender: 'swapnavue', text: data.response || "swapnavue couldn't generate a response." };
      const debugInfoText = `(Confidence: ${data.confidence.toFixed(4)}, Meta-Error: ${data.meta_error.toFixed(4)}, Focus: ${data.focus}, Curiosity: ${data.curiosity.toFixed(6)}${
        data.continuous_learning_loss !== undefined && data.continuous_learning_loss !== null
          ? `, CL Loss: ${data.continuous_learning_loss.toFixed(4)}`
          : ''
      })`;
      const swapnavueDebugInfo: ChatMessage = { sender: 'swapnavue', text: debugInfoText };
      setMessages((prevMessages) => [...prevMessages, swapnavueResponse, swapnavueDebugInfo]);

    } catch (error) {
      console.error('Error generating response:', error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: 'swapnavue', text: `Error: Failed to connect to swapnavue's core or functionality paused. (${error instanceof Error ? error.message : String(error)})` },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = async () => {
    if (!confirm('Are you sure you want to clear the entire chat history? This cannot be undone.')) return;
    setIsLoading(true);
    try {
      if (!BACKEND_API_URL) throw new Error("BACKEND_API_URL is not defined.");
      const res = await fetch(`${BACKEND_API_URL}/clear_chat_history`, { method: 'DELETE' });
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      setMessages([]);
      alert(data.message);
    } catch (error) {
      alert(`Failed to clear chat history: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleExportChat = async () => {
    try {
      if (!BACKEND_API_URL) throw new Error("Backend URL not configured.");
      window.open(`${BACKEND_API_URL}/export_chat`, '_blank');
    } catch (error) {
      alert(`Failed to export history: ${error instanceof Error ? error.message : String(error)}`);
    }
  };
  
  const handleStartTraining = async () => {
    setIsLoading(true);
    try {
        if (!BACKEND_API_URL) throw new Error("Backend URL not configured.");
        const res = await fetch(`${BACKEND_API_URL}/start_training`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.message || 'Failed to start training');
        alert(data.message);
    } catch (error) {
        alert(`Error starting training: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
        setIsLoading(false);
    }
  };

  const handleStopTraining = async () => {
    setIsLoading(true);
    try {
        if (!BACKEND_API_URL) throw new Error("Backend URL not configured.");
        const res = await fetch(`${BACKEND_API_URL}/stop_training`, { method: 'POST' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.message || 'Failed to stop training');
        alert(data.message);
    } catch (error) {
        alert(`Error stopping training: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
        setIsLoading(false);
    }
  };

  const isChatDisabled = isLoading || (trainingStatus?.is_training_active ?? false);

  const buttonStyle: CSSProperties = {
    padding: '0.6rem 1rem',
    color: 'var(--button-primary-text)',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
  };

  const disabledButtonStyle: CSSProperties = {
    ...buttonStyle,
    backgroundColor: 'var(--card-border)',
    cursor: 'not-allowed',
  };

  return (
    <div style={{ padding: '1rem', width: '100%' }}>
      <h1 style={{ marginBottom: '1.5rem', textAlign: 'center' }}>swapnavue</h1>
      
      <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        <button
          onClick={handleStartTraining}
          disabled={isChatDisabled}
          style={isChatDisabled ? disabledButtonStyle : { ...buttonStyle, backgroundColor: 'var(--button-primary-bg)'}}
        >
          Start Training
        </button>
        <button
          onClick={handleStopTraining}
          disabled={!trainingStatus?.is_training_active}
          style={!trainingStatus?.is_training_active ? disabledButtonStyle : { ...buttonStyle, backgroundColor: 'var(--button-danger-bg)'}}
        >
          Stop Training
        </button>
      </div>

      <div style={{
        maxWidth: '800px', margin: '0 auto', backgroundColor: 'var(--card-background)',
        borderRadius: '8px', boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        marginBottom: '1.5rem', display: 'flex', flexDirection: 'column',
        border: '1px solid var(--card-border)',
      }}>
        <div style={{ flexGrow: 1, overflowY: 'auto', maxHeight: '60vh', padding: '1.5rem' }}>
          {messages.length === 0 ? (
            <p style={{ textAlign: 'center', color: 'var(--foreground-subtle)' }}>
              {isChatDisabled ? "Chat is paused during training." : "Start a conversation with swapnavue..."}
            </p>
          ) : (
            messages.map((msg, index) => (
              <div key={index} style={{ display: 'flex', justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start', marginBottom: '0.8rem' }}>
                <div style={{
                  backgroundColor: msg.sender === 'user' ? 'var(--chat-bubble-user-bg)' : 'var(--chat-bubble-swapnavue-bg)',
                  border: msg.sender === 'swapnavue' ? `1px solid var(--card-border)` : 'none',
                  borderRadius: '12px', padding: '0.7rem 1rem', maxWidth: '80%', wordWrap: 'break-word',
                  fontSize: msg.sender === 'swapnavue' && msg.text.startsWith('(') ? '0.75rem' : '1rem',
                  color: msg.sender === 'swapnavue' && msg.text.startsWith('(') ? 'var(--foreground-subtle)' : 'var(--foreground)'
                }}>
                  <strong>{msg.sender === 'user' ? 'You' : 'swapnavue'}:</strong> {msg.text}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem', padding: '1rem', borderTop: `1px solid var(--card-border)` }}>
            <button onClick={handleExportChat} disabled={isChatDisabled} style={isChatDisabled ? {display:'none'} : {}}>Export Chat</button>
            <button onClick={handleClearChat} disabled={isChatDisabled} style={isChatDisabled ? {display:'none'} : {}}>Clear Chat</button>
        </div>
      </div>

      <form onSubmit={handleSubmit} style={{ maxWidth: '800px', margin: '0 auto', display: 'flex', gap: '1rem' }}>
        <input
          type="text"
          value={inputPrompt}
          onChange={(e) => setInputPrompt(e.target.value)}
          placeholder={isChatDisabled ? "Training in progress..." : "Type your message..."}
          disabled={isChatDisabled}
          style={{
            flexGrow: 1, padding: '0.8rem 1rem', border: `1px solid var(--card-border)`,
            borderRadius: '8px', fontSize: '1rem', outline: 'none',
            backgroundColor: isChatDisabled ? 'var(--card-border)' : 'var(--card-background)',
            color: 'var(--foreground)'
          }}
        />
        <button
          type="submit"
          disabled={isChatDisabled}
          style={isChatDisabled ? disabledButtonStyle : { ...buttonStyle, backgroundColor: 'var(--button-primary-bg)'}}
        >
          Send
        </button>
      </form>
    </div>
  );
}