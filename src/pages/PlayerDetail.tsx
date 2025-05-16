import React from 'react';
import { useParams } from 'react-router-dom';

const PlayerDetail = () => {
  const { playerId } = useParams<{ playerId: string }>();

  return (
    <div>
      <h1>Player Detail Page</h1>
      <p>Displaying details for player: {playerId}</p>
      {/* Player stats and prop probabilities will go here */}
    </div>
  );
};

export default PlayerDetail; 