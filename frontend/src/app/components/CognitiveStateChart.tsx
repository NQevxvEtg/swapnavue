'use client';

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrainingStatusResponse } from '../context/TrainingContext';

// CORRECTED: Added the optional 'domain' property to the interface
interface SingleMetricChartProps {
  data: (TrainingStatusResponse & { time: string })[];
  title: string;
  dataKey: keyof TrainingStatusResponse;
  strokeColor: string;
  domain?: [number | string, number | string];
}

export const CognitiveStateChart = ({ data, title, dataKey, strokeColor, domain }: SingleMetricChartProps) => {
  
  // REMOVED: The internal 'calculateDomain' function is no longer needed here.
  // The domain will be passed in from the parent page.

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ textAlign: 'center', marginTop: 0, marginBottom: '1rem', color: 'var(--foreground-subtle)' }}>
        {title}
      </h3>
      <div style={{ flexGrow: 1 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }} >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--card-border)" />
            <XAxis dataKey="time" fontSize="0.75rem" tick={{ fill: 'var(--foreground-subtle)' }} />
            <YAxis 
              fontSize="0.75rem" 
              tick={{ fill: 'var(--foreground-subtle)' }} 
              // CORRECTED: Use the 'domain' prop, with a fallback to 'auto'
              domain={domain || ['auto', 'auto']}
              tickFormatter={(value) => typeof value === 'number' ? value.toFixed(3) : ''}
              allowDataOverflow={true}
            />
            <Tooltip
              contentStyle={{ backgroundColor: 'var(--background)', border: '1px solid var(--card-border)'}}
              formatter={(value: number) => typeof value === 'number' ? value.toFixed(4) : 'N/A'}
            />
            <Legend verticalAlign="top" height={36}/>
            <Line 
              type="monotone" 
              dataKey={dataKey as string}
              stroke={strokeColor} 
              strokeWidth={2} 
              name={title} 
              dot={false} 
              isAnimationActive={false}
              connectNulls={true} 
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};