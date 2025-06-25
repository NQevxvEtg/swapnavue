'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { useTraining, TrainingStatusResponse } from '../context/TrainingContext';
import MemoryGrid from '../components/MemoryGrid';
import PermanenceHistogram from '../components/PermanenceHistogram';
import { CognitiveStateChart } from '../components/CognitiveStateChart';

const COLORS = {
  predictiveAccuracy: '#ff7300',
  tmSparsity: '#387908'
};

const chartKeys = ['predictive_accuracy', 'tm_sparsity'] as const;
type ChartKey = typeof chartKeys[number];

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


export default function MemoryPage() {
  const { memoryState, cognitiveStateHistory } = useTraining();
  const [displayHistory, setDisplayHistory] = useState<(TrainingStatusResponse & { time: string })[]>([]);
  
  const windowSize = 500;

  useEffect(() => {
    setDisplayHistory(cognitiveStateHistory.slice(-windowSize));
  }, [cognitiveStateHistory]);

  const yDomains = useMemo(() => {
    const newDomains: { [key in ChartKey]?: [number, number] } = {};
    if (displayHistory.length > 0) {
      chartKeys.forEach((key) => {
        const domain = calculateAdaptiveDomain(displayHistory, key);
        newDomains[key] = domain;
      });
    }
    return newDomains;
  }, [displayHistory]);

  const getCombinedCellData = () => {
    if (!memoryState || !memoryState.activeCells || !memoryState.predictiveCells) {
      return [];
    }
    const { activeCells, predictiveCells } = memoryState;
    return activeCells.map((active, i) => {
      const predictive = predictiveCells[i];
      if (active && predictive) return 3;
      if (active) return 2;
      if (predictive) return 1;
      return 0;
    });
  };

  const cellColorMap = {
    0: '#1C1C1E',
    1: '#4A90E2',
    2: '#F5C518',
    3: '#7ED6A5',
  };
  const permanences = memoryState?.permanences;
  const gridDimensions = memoryState?.gridDimensions;

  return (
    <main style={{ padding: '1rem' }}>
      <h1 style={{ textAlign: 'center', marginBottom: '1.5rem' }}>Temporal Memory Dynamics</h1>
      
      <div className="grid w-full max-w-6xl grid-cols-1 gap-6 md:grid-cols-2 mx-auto mb-8">
        <div key="accuracy-chart" className="chart-container">
          <CognitiveStateChart data={displayHistory} title="Predictive Accuracy" dataKey="predictive_accuracy" strokeColor={COLORS.predictiveAccuracy} domain={yDomains.predictive_accuracy} />
        </div>
        <div key="sparsity-chart" className="chart-container">
          <CognitiveStateChart data={displayHistory} title="TM Sparsity" dataKey="tm_sparsity" strokeColor={COLORS.tmSparsity} domain={yDomains.tm_sparsity} />
        </div>
      </div>

      <div className="flex flex-col md:flex-row gap-6 w-full max-w-6xl mx-auto">
        <div className="flex-1">
          <h2 style={{ textAlign: 'center', marginBottom: '1rem' }}>Permanence Distribution</h2>
          <div className="chart-container" style={{height: '400px'}}>
            {permanences ? <PermanenceHistogram data={permanences} /> : <p className="text-center" style={{color: 'var(--foreground-subtle)'}}>No permanence data available.</p>}
          </div>
        </div>
        <div className="flex-1">
          <h2 style={{ textAlign: 'center', marginBottom: '1rem' }}>Temporal Memory Cell Activity</h2>
           <div className="chart-container" style={{height: '400px', padding: '0.5rem'}}>
            {gridDimensions ? (
              <MemoryGrid
                title="Temporal Memory Cell Activity"
                gridData={getCombinedCellData()}
                gridCols={gridDimensions.cells[0]}
                gridRows={gridDimensions.cells[1]}
                colorMap={cellColorMap}
              />
            ) : <p className="text-center" style={{color: 'var(--foreground-subtle)'}}>No grid data available.</p>}
          </div>
        </div>
      </div>
    </main>
  );
}