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
  TimeScale,
  TimeSeriesScale
} from 'chart.js';
import 'chartjs-adapter-date-fns';
import { PlayerStatFull } from '../services/api';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  TimeSeriesScale
);

interface ChartProps {
  playerStats: PlayerStatFull[];
  playerName: string;
  statToDisplay: keyof PlayerStatFull;
}

const PlayerPerformanceChart: React.FC<ChartProps> = ({ playerStats, playerName, statToDisplay }) => {
  const processedChartData = playerStats
    .filter(stat => {
      const statValue = stat[statToDisplay];
      return stat.game_date && typeof statValue === 'number' && !isNaN(new Date(stat.game_date).getTime());
    })
    .map(stat => ({
      x: new Date(stat.game_date as string),
      y: stat[statToDisplay] as number,
    }))
    .sort((a, b) => a.x.getTime() - b.x.getTime());

  if (!processedChartData || processedChartData.length === 0) {
    return <p>No valid performance data available for {playerName} for the selected stat ({String(statToDisplay)}).</p>;
  }

  const chartData = {
    datasets: [
      {
        label: `${playerName} - ${String(statToDisplay)}`,
        data: processedChartData,
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
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
        text: `${playerName} - ${String(statToDisplay).charAt(0).toUpperCase() + String(statToDisplay).slice(1)} Trend`,
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false,
      },
    },
    scales: {
      x: {
        type: 'time' as const,
        time: {
          unit: 'day' as const,
          tooltipFormat: 'MMM d, yyyy',
          displayFormats: {
            day: 'MMM d'
          }
        },
        title: {
          display: true,
          text: 'Game Date'
        }
      },
      y: {
        beginAtZero: true,
        title: {
          display: true,
          text: String(statToDisplay).charAt(0).toUpperCase() + String(statToDisplay).slice(1)
        }
      }
    }
  };

  return <Line options={options} data={chartData} />;
};

export default PlayerPerformanceChart; 