'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// --- Data Shape Definitions ---

export interface TrainingStatusResponse {
  timestamp?: string;
  is_training_active: boolean;
  message: string;
  focus: number;
  confidence: number;
  meta_error: number;
  curiosity: number;
  predictive_accuracy?: number | null;
  tm_sparsity?: number | null;
  current_epoch?: number;
  current_batch?: number;
  total_batches_in_epoch?: number;
  train_loss?: number | null;
  val_loss?: number | null;
  best_val_loss?: number | null;
  cognitive_stress?: number | null;
  target_amplitude?: number | null;
  current_amplitude?: number | null;
  target_frequency?: number | null;
  current_frequency?: number | null;
  base_focus?: number | null;
  base_curiosity?: number | null;
  state_drift?: number | null;
  continuous_learning_loss?: number | null;
}

export interface PermanenceData {
  volatile: { values: number[]; bins: number[] };
  consolidated: { values: number[]; bins: number[] };
}

export interface GridDimensions {
  sdr: [number, number];
  cells: [number, number];
}

export interface MemoryState {
  sdr: number[];
  activeCells: number[];
  predictiveCells: number[];
  permanences: PermanenceData;
  gridDimensions: GridDimensions;
}

interface TrainingContextType {
  trainingStatus: TrainingStatusResponse | null;
  memoryState: MemoryState | null;
  isConnected: boolean;
  cognitiveStateHistory: (TrainingStatusResponse & { time: string })[];
}

const TrainingContext = createContext<TrainingContextType>({
  trainingStatus: null,
  memoryState: null,
  isConnected: false,
  cognitiveStateHistory: [],
});

export const useTraining = () => useContext(TrainingContext);

export const TrainingProvider = ({ children }: { children: ReactNode }) => {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatusResponse | null>(null);
  const [memoryState, setMemoryState] = useState<MemoryState | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [cognitiveStateHistory, setCognitiveStateHistory] = useState<(TrainingStatusResponse & { time: string })[]>([]);
  const BACKEND_API_URL = process.env.NEXT_PUBLIC_BACKEND_API_URL;

  useEffect(() => {
    const fetchHistory = async () => {
      if (!BACKEND_API_URL) return;
      try {
        const res = await fetch(`${BACKEND_API_URL}/cognitive_state_history`);
        if (res.ok) {
          const data: TrainingStatusResponse[] = await res.json();
          const formattedData = data.map(d => ({
            ...d,
            time: d.timestamp ? new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : 'N/A',
          }));
          setCognitiveStateHistory(formattedData);
        }
      } catch (error) {
        console.error("Failed to fetch history:", error);
      }
    };
    fetchHistory();
  }, [BACKEND_API_URL]);

  useEffect(() => {
    if (!BACKEND_API_URL) return;
    const wsUrl = BACKEND_API_URL.replace(/http/g, 'ws') + '/ws';
    let ws: WebSocket;
    let reconnectTimeout: NodeJS.Timeout;

    function connect() {
      ws = new WebSocket(wsUrl);
      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => {
        setIsConnected(false);
        reconnectTimeout = setTimeout(connect, 5000);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          // CORRECTLY HANDLE BOTH MESSAGE TYPES
          if (message.type === 'memory_state') {
            setMemoryState(message.data as MemoryState);
          } else {
            const newStatus = message as TrainingStatusResponse;
            setTrainingStatus(newStatus);
            const now = new Date();
            const newHistoryPoint = {
              ...newStatus,
              time: now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
            };
            setCognitiveStateHistory(prevHistory => [...prevHistory, newHistoryPoint]);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
    }
    connect();
    return () => {
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, [BACKEND_API_URL]);

  const value = { trainingStatus, memoryState, isConnected, cognitiveStateHistory };

  return <TrainingContext.Provider value={value}>{children}</TrainingContext.Provider>;
};