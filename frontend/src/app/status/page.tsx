'use client';

import { useState, useEffect, useMemo } from 'react';
import { useTraining, TrainingStatusResponse } from '../context/TrainingContext';
import { CognitiveStateChart } from '../components/CognitiveStateChart';

const COLORS = {
  focus: '#8884d8',
  confidence: '#82ca9d',
  metaError: '#ca8282',
  amplitude: '#ff8042',
  drift: '#00C49F',
  growth: '#0088FE',
  trainLoss: '#3498db',
  valLoss: '#9b59b6',
  clLoss: '#e74c3c',
  currentAmplitude: '#8a2be2',
};

const chartKeys = [
  'focus', 'confidence', 'meta_error', 'state_drift', 'base_focus',
  'current_amplitude', 'target_amplitude', 'train_loss', 'val_loss', 'continuous_learning_loss'
] as const;

type ChartKey = typeof chartKeys[number];

const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL;

const calculateAdaptiveDomain = (data: (TrainingStatusResponse & { time: string })[], dataKey: keyof TrainingStatusResponse): [number, number] => {
    if (!data || data.length === 0) return [0, 1];
    const values = data.map(d => d[dataKey] as number).filter(v => v !== null && isFinite(v));
    if (values.length === 0) return [0, 1];

    const dataMin = Math.min(...values);
    const dataMax = Math.max(...values);
    const range = dataMax - dataMin;

    if (range === 0) {
        // If all values are the same, provide a very small arbitrary range around that value
        return [dataMin - 0.005, dataMax + 0.005]; 
    } else {
        // Use dataMin and dataMax directly without any padding
        return [dataMin, dataMax];
    }
};

export default function StatusPage() {
  const { cognitiveStateHistory, trainingStatus } = useTraining();
  const [displayHistory, setDisplayHistory] = useState<(TrainingStatusResponse & { time: string })[]>([]);
  const [windowSize, setWindowSize] = useState(150);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    setDisplayHistory(cognitiveStateHistory.slice(-windowSize));
  }, [cognitiveStateHistory, windowSize]);

  const yDomains = useMemo(() => {
    const newDomains: { [key in ChartKey]?: [number, number] } = {};
    if (displayHistory.length > 0) {
      chartKeys.forEach((key) => {
        newDomains[key] = calculateAdaptiveDomain(displayHistory, key);
      });
    }
    return newDomains;
  }, [displayHistory]);
  
  const handleExport = async () => {
    if (!BACKEND_API_URL) return alert("Backend URL not configured.");
    window.open(`${BACKEND_API_URL}/export_cognitive_state`, '_blank');
  };

  const handleClear = async () => {
    if (!confirm('Are you sure you want to clear the entire cognitive state history? This action cannot be undone.')) return;
    setIsLoading(true);
    try {
      if (!BACKEND_API_URL) throw new Error("Backend URL not configured.");
      const res = await fetch(`${BACKEND_API_URL}/clear_cognitive_state`, { method: 'DELETE' });
      if (res.ok) {
        alert("History cleared successfully.");
        window.location.reload();
      } else {
        throw new Error((await res.json()).detail || "Failed to clear history.");
      }
    } catch (error) {
      alert(`Error clearing history: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main style={{ padding: '1rem' }}>
      <h1 style={{ textAlign: 'center' }}>swapnavue&apos;s real-time status</h1>
      <div style={{ width: '100%', maxWidth: '1200px', margin: '1rem auto', display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem'}}>
            <label htmlFor="window-size-input" style={{color: 'var(--foreground-subtle)'}}>Data Points:</label>
            <input id="window-size-input" type="number" value={windowSize} onChange={(e) => setWindowSize(parseInt(e.target.value, 10) || 1)} min="1"
              style={{ width: '80px', padding: '0.5rem', backgroundColor: 'var(--card-background)', color: 'var(--foreground)', border: '1px solid var(--card-border)', borderRadius: '4px' }}/>
          </div>
          <button onClick={handleExport} disabled={isLoading} style={{padding: '0.5rem 1rem', cursor: 'pointer', border: `1px solid var(--button-secondary-border)`, borderRadius: '4px', backgroundColor: 'var(--button-secondary-bg)', color: 'var(--button-secondary-text)'}}>Export History</button>
          <button onClick={handleClear} disabled={isLoading} style={{padding: '0.5rem 1rem', cursor: 'pointer', backgroundColor: 'var(--button-danger-bg)', color: 'var(--button-danger-text)', border: 'none', borderRadius: '4px'}}>Clear History</button>
      </div>
      <div className="grid w-full max-w-6xl grid-cols-1 gap-6 md:grid-cols-2 mx-auto">
        <div key="focus-chart" className="chart-container"><CognitiveStateChart data={displayHistory} title="Focus" dataKey="focus" strokeColor={COLORS.focus} domain={yDomains.focus} /></div>
        <div key="confidence-chart" className="chart-container"><CognitiveStateChart data={displayHistory} title="Confidence" dataKey="confidence" strokeColor={COLORS.confidence} domain={yDomains.confidence} /></div>
        <div key="meta-error-chart" className="chart-container"><CognitiveStateChart data={displayHistory} title="Meta-Error" dataKey="meta_error" strokeColor={COLORS.metaError} domain={yDomains.meta_error} /></div>
        <div key="drift-chart" className="chart-container"><CognitiveStateChart data={displayHistory} title="State Drift" dataKey="state_drift" strokeColor={COLORS.drift} domain={yDomains.state_drift} /></div>
        <div key="growth-chart" className="chart-container"><CognitiveStateChart data={displayHistory} title="Long-Term Growth (Base Focus)" dataKey="base_focus" strokeColor={COLORS.growth} domain={yDomains.base_focus}/></div>
        <div key="current-amp-chart" className="chart-container"><CognitiveStateChart data={displayHistory} title="Resonator Amplitude (Current)" dataKey="current_amplitude" strokeColor={COLORS.currentAmplitude} domain={yDomains.current_amplitude}/></div>
        <div key="target-amp-chart" className="chart-container"><CognitiveStateChart data={displayHistory} title="Resonator Amplitude (Target)" dataKey="target_amplitude" strokeColor={COLORS.amplitude} domain={yDomains.target_amplitude}/></div>
        <div key="train-loss-chart" className="chart-container"><CognitiveStateChart data={displayHistory} title="Training Loss" dataKey="train_loss" strokeColor={COLORS.trainLoss} domain={yDomains.train_loss} /></div>
        <div key="val-loss-chart" className="chart-container"><CognitiveStateChart data={displayHistory} title="Validation Loss" dataKey="val_loss" strokeColor={COLORS.valLoss} domain={yDomains.val_loss} /></div>
        <div key="cl-loss-chart" className="chart-container"><CognitiveStateChart data={displayHistory} title="Interactive Loss" dataKey="continuous_learning_loss" strokeColor={COLORS.clLoss} domain={yDomains.continuous_learning_loss} /></div>
      </div>
      {trainingStatus && trainingStatus.is_training_active && (
        <div style={{ marginTop: '2rem', backgroundColor: 'var(--card-background)', borderRadius: '8px', padding: '1rem 1.5rem', width: '100%', maxWidth: '1200px', margin: '2rem auto 0', fontSize: '0.9rem', color: 'var(--foreground)', border: '1px solid var(--card-border)' }}>
          <h3 style={{marginTop: 0}}>Training In Progress</h3>
          <strong>Status Message:</strong> {trainingStatus.message}
        </div>
      )}
    </main>
  );
}