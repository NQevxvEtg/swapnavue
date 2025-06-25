'use client';

import React, { useRef, useEffect } from 'react';

interface MemoryGridProps {
  title: string;
  gridData: number[];
  gridCols: number;
  gridRows: number;
  colorMap: { [key: number]: string };
}

const MemoryGrid: React.FC<MemoryGridProps> = ({ title, gridData, gridCols, gridRows, colorMap }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !gridData || gridData.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas resolution to match its display size for clarity
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    const cellWidth = canvas.width / gridCols;
    const cellHeight = canvas.height / gridRows;

    const inactiveColor = colorMap[0] || '#000000'; // Default to black for inactive cells

    // Loop through all cells and draw a colored rectangle for each one
    for (let row = 0; row < gridRows; row++) {
      for (let col = 0; col < gridCols; col++) {
        const index = row * gridCols + col;
        const value = gridData[index] || 0;

        ctx.fillStyle = colorMap[value] || inactiveColor;
        ctx.fillRect(col * cellWidth, row * cellHeight, cellWidth, cellHeight);
      }
    }
  }, [gridData, gridCols, gridRows, colorMap]);

  return (
    <div className="flex flex-col h-full">
      <h3 className="font-semibold text-lg mb-2 text-center text-gray-900 dark:text-gray-100">{title}</h3>
      {/* The container now has a simple black background */}
      <div className="flex-1 bg-black rounded-md">
        <canvas
          ref={canvasRef}
          className="w-full h-full"
        />
      </div>
    </div>
  );
};

export default MemoryGrid;