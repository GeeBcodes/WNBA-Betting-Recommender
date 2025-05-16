import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ChartProps {
  // TODO: Define props for player data, selected stat, etc.
  playerName?: string;
}

const PlayerPerformanceChart: React.FC<ChartProps> = ({ playerName = "Player" }) => {
  // Placeholder data for player performance (e.g., points per game over last 5 games)
  const data = {
    labels: ['Game 1', 'Game 2', 'Game 3', 'Game 4', 'Game 5'],
    datasets: [
      {
        label: `${playerName} Points Performance`,
        data: [18, 22, 15, 25, 20], // Example points data
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
      // TODO: Add more datasets for other stats or players for comparison
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: `${playerName} - Performance Trend`,
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: 'Points' // TODO: Make this dynamic based on selected stat
        }
      }
    }
  };

  return <Line options={options} data={data} />;
};

export default PlayerPerformanceChart; 