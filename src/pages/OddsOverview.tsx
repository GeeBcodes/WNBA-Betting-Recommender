import React from 'react';
import OddsTables from '../components/OddsTables';

const OddsOverviewPage: React.FC = () => {
  return (
    <div>
      <h1>Odds Overview</h1>
      <p>Displaying current game odds and player prop odds.</p>
      <OddsTables />
    </div>
  );
};

export default OddsOverviewPage; 