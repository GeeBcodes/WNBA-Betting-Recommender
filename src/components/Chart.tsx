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

  const chartLabels = playerStats.map(stat => {
    const gameDate = new Date(stat.game_date as string);
    let opponentName = 'N/A';
    let venue = '';

    if (stat.game && stat.player && stat.player.team_name) {
      if (stat.game.away_team === stat.player.team_name) { // Player was the away team
        opponentName = stat.game.home_team; // Opponent is the home team
        venue = '@';
      } else if (stat.game.home_team === stat.player.team_name) { // Player was the home team
        opponentName = stat.game.away_team; // Opponent is the away team
        venue = 'vs ';
      } else {
        // Fallback if player's team doesn't match home or away (shouldn't happen with good data)
        opponentName = `${stat.game.home_team} or ${stat.game.away_team}`;
        venue = 'vs/ရှ '; // Indicate uncertainty
      }
    } else {
        // Fallback if game or player team_name is missing
        opponentName = 'Unknown Opponent';
        venue = '';
    }
    // Attempt to create a short name / abbreviation (e.g., first 3 chars)
    // This is a simple approach; a mapping would be more robust for official abbreviations.
    const opponentAbbreviation = opponentName.substring(0, 3).toUpperCase();

    return `${venue}${opponentAbbreviation} (${gameDate.getMonth() + 1}/${gameDate.getDate()})`;
  });

  const chartDataset = playerStats.map(stat => stat[statToDisplay] as number);

  if (!playerStats || playerStats.length === 0) {
    return <p>No valid performance data available for {playerName} for the selected stat ({String(statToDisplay)}).</p>;
  }

  const chartData = {
    labels: chartLabels,
    datasets: [
      {
        type: 'bar' as const,
        label: `${playerName} - ${String(statToDisplay)}`,
        data: chartDataset,
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        borderColor: 'rgb(75, 192, 192)',
        borderWidth: 1,
        fill: false,
        tension: 0.1,
      },
      ...(bettingLineValue !== null ? [{
        type: 'line' as const,
        label: 'Betting Line',
        data: chartLabels.map(() => bettingLineValue),
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
        type: 'category' as const,
        labels: chartLabels,
        title: {
          display: true,
          text: 'Game Sequence'
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