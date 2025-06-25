// frontend/src/app/components/PermanenceHistogram.tsx
'use client';

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { PermanenceData } from '../context/TrainingContext';

interface PermanenceHistogramProps {
  data: PermanenceData;
}

const PermanenceHistogram: React.FC<PermanenceHistogramProps> = ({ data }) => {
  if (!data?.volatile?.values || !data?.consolidated?.values) {
    return <div>Loading permanence data...</div>;
  }

  const chartData = data.volatile.bins.slice(0, -1).map((bin, index) => ({
    name: bin.toFixed(2),
    volatile: data.volatile.values[index],
    consolidated: data.consolidated.values[index],
  }));

  return (
    <div className="flex flex-col h-full">
      <h3 className="font-semibold text-lg mb-2 text-gray-900 dark:text-gray-100">Synaptic Permanence Distribution</h3>
      <div className="flex-grow">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(128, 128, 128, 0.2)" />
            <XAxis dataKey="name" stroke="rgba(128, 128, 128, 0.7)" />
            <YAxis stroke="rgba(128, 128, 128, 0.7)" />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(30, 41, 59, 0.9)',
                borderColor: 'rgba(128, 128, 128, 0.5)',
              }}
              labelStyle={{ color: '#cbd5e1' }}
            />
            <Legend />
            <Bar dataKey="volatile" fill="#8884d8" name="Volatile" />
            <Bar dataKey="consolidated" fill="#82ca9d" name="Consolidated" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default PermanenceHistogram;