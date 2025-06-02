import React, { useState } from 'react';
import { Chart } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
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
  BarElement,
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
  const [bettingLineValue, setBettingLineValue] = useState<number | null>(null);

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
    labels: processedChartData.map(d => d.x),
    datasets: [
      {
        type: 'bar' as const,
        label: `${playerName} - ${String(statToDisplay)}`,
        data: processedChartData,
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        borderColor: 'rgb(75, 192, 192)',
        borderWidth: 1,
        fill: false,
        tension: 0.1,
      },
      ...(bettingLineValue !== null ? [{
        type: 'line' as const,
        label: 'Betting Line',
        data: processedChartData.map(d => ({ x: d.x, y: bettingLineValue })),
        borderColor: 'rgb(255, 99, 132)',
        borderWidth: 2,
        borderDash: [5, 5],
        pointRadius: 0,
        fill: false,
        tension: 0.1,
      }] : [])
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

  return (
    <div>
      <div style={{ marginBottom: '10px', display: 'flex', alignItems: 'center' }}>
        <label htmlFor="betting-line-input" style={{ marginRight: '10px' }}>Betting Line:</label>
        <input
          type="number"
          id="betting-line-input"
          value={bettingLineValue === null ? '' : bettingLineValue}
          onChange={(e) => {
            const value = e.target.value;
            setBettingLineValue(value === '' ? null : parseFloat(value));
          }}
          placeholder="Enter line (e.g., 15.5)"
          style={{ padding: '5px' }}
        />
      </div>
      <Chart type='bar' options={options} data={chartData} />
    </div>
  );
};

export default PlayerPerformanceChart; 