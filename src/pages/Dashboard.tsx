import React, { useEffect, useState, useMemo } from 'react';
import DataGrid from '../components/DataGrid';
// import PlayerPerformanceChart from '../components/Chart';
import ModelVersionInfo from '../components/ModelVersionInfo';
// import PlayerStatsTable from '../components/PlayerStatsTable';
import { getPredictions, Prediction, PlayerStatFull, getPlayerStats, postParlay, ParlayData } from '../services/api';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

// Interface for a staged parlay leg
export interface StagedParlayLeg {
  prediction: Prediction;
  type: 'over' | 'under';
}

// Constants for dropdowns
export const availableStats: { value: keyof PlayerStatFull; label: string }[] = [
    { value: 'points', label: 'Points' },
    { value: 'rebounds', label: 'Rebounds' },
    { value: 'assists', label: 'Assists' },
    { value: 'steals', label: 'Steals' },
    { value: 'blocks', label: 'Blocks' },
    { value: 'turnovers', label: 'Turnovers' },
    { value: 'three_pointers_made', label: '3-Pointers Made' },
    { value: 'pra', label: 'PRA (Pts+Reb+Ast)' },
    { value: 'points_plus_rebounds', label: 'Points + Rebounds' },
    { value: 'points_plus_assists', label: 'Points + Assists' },
    { value: 'rebounds_plus_assists', label: 'Rebounds + Assists' },
    { value: 'blocks_plus_steals', label: 'Blocks + Steals' },
];

export const dateRangeFilterOptions = [
    { value: 'allTime', label: 'All Time' },
    { value: 'entireSeason', label: 'Selected Season' },
    { value: 'last5', label: 'Last 5 Games' },
    { value: 'last10', label: 'Last 10 Games' },
    { value: 'last20', label: 'Last 20 Games' },
];

export const locationFilterOptions = [
    { value: 'all', label: 'All' },
    { value: 'home', label: 'Home' },
    { value: 'away', label: 'Away' },
];

const Dashboard = () => {
    const [predictions, setPredictions] = useState<Prediction[]>([]);
    // const [allPlayerStats, setAllPlayerStats] = useState<PlayerStatFull[]>([]);
    const [stagedParlayLegs, setStagedParlayLegs] = useState<StagedParlayLeg[]>([]);
    const [loadingPredictions, setLoadingPredictions] = useState<boolean>(true);
    // const [loadingPlayerStats, setLoadingPlayerStats] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const [showModelInfoPopover, setShowModelInfoPopover] = useState<boolean>(false);
    const infoIconRef = React.useRef<HTMLSpanElement>(null);
    // const [selectedChartStat, setSelectedChartStat] = useState<keyof PlayerStatFull>("points");
    // const [selectedPlayerIdForChart, setSelectedPlayerIdForChart] = useState<string | null>(null);
    // const [selectedDateRangeFilter, setSelectedDateRangeFilter] = useState<string>('allTime');
    // const [selectedLocationFilter, setSelectedLocationFilter] = useState<string>('all');
    // const [selectedOpponentTeamNameForH2H, setSelectedOpponentTeamNameForH2H] = useState<string>('all');
    // const [playerSearchTerm, setPlayerSearchTerm] = useState<string>('');
    // const [selectedSeasonForChart, setSelectedSeasonForChart] = useState<string>('all');

    const combinedParlayProbability = useMemo(() => {
        if (stagedParlayLegs.length === 0) return null;
        return stagedParlayLegs.reduce((prob, leg) => {
            const legProb = leg.type === 'over' 
                ? leg.prediction.over_p_value_calibrated ?? leg.prediction.predicted_over_probability 
                : leg.prediction.under_p_value_calibrated ?? leg.prediction.predicted_under_probability;
            return prob * (legProb ?? 0);
        }, 1);
    }, [stagedParlayLegs]);

    useEffect(() => {
        const fetchPredictionsData = async () => {
            try {
                setLoadingPredictions(true);
                const data = await getPredictions();
                setPredictions(data);
            } catch (err) {
                console.error("Failed to fetch predictions:", err);
                setError(prev => prev ? `${prev}\nFailed to load predictions.` : 'Failed to load predictions.');
                setPredictions([]);
            } finally {
                setLoadingPredictions(false);
            }
        };
        fetchPredictionsData();
    }, []);

    /*
    useEffect(() => {
        const fetchAllPlayerStats = async () => {
            try {
                setLoadingPlayerStats(true);
                const statsData = await getPlayerStats();
                setAllPlayerStats(statsData);
                if (statsData.length > 0 && !selectedPlayerIdForChart) {
                    setSelectedPlayerIdForChart(statsData[0].player_id);
                }
            } catch (err) {
                console.error("Failed to fetch player stats:", err);
                setError(prev => prev ? `${prev}\nFailed to load player stats.` : 'Failed to load player stats.');
                setAllPlayerStats([]);
            } finally {
                setLoadingPlayerStats(false);
            }
        };
        fetchAllPlayerStats();
    }, []);
    */

    /*
    const uniqueOpponentTeamNames = useMemo(() => {
        const names = new Set<string>();
        allPlayerStats.forEach(stat => {
            if (stat.game && stat.player && stat.game.home_team && stat.game.away_team) {
                const opponent = stat.game.home_team === stat.player.team_name ? stat.game.away_team : stat.game.home_team;
                if (opponent) names.add(opponent);
            }
        });
        return Array.from(names).sort();
    }, [allPlayerStats]);

    const uniquePlayersForChart = useMemo(() => {
        const playerMap = new Map<string, { id: string; name: string }>();
        allPlayerStats.forEach(stat => {
            if (stat.player && !playerMap.has(stat.player_id)) {
                playerMap.set(stat.player_id, { id: stat.player_id, name: stat.player.player_name });
            }
        });
        return Array.from(playerMap.values());
    }, [allPlayerStats]);

    const uniqueSeasonsForChart = useMemo(() => {
        const seasons = new Set<string>();
        allPlayerStats.forEach(stat => {
            if (stat.game_date) {
                seasons.add(new Date(stat.game_date).getFullYear().toString());
            }
        });
        return ['all', ...Array.from(seasons).sort((a, b) => parseInt(b) - parseInt(a))];
    }, [allPlayerStats]);

    const filteredUniquePlayersForChart = useMemo(() => {
        if (!playerSearchTerm) return uniquePlayersForChart;
        return uniquePlayersForChart.filter(p => p.name.toLowerCase().includes(playerSearchTerm.toLowerCase()));
    }, [uniquePlayersForChart, playerSearchTerm]);

    const selectedPlayerForChart = useMemo(() => {
        return filteredUniquePlayersForChart.find(p => p.id === selectedPlayerIdForChart) || null;
    }, [selectedPlayerIdForChart, filteredUniquePlayersForChart]);
    
    const chartPlayerName = selectedPlayerForChart ? selectedPlayerForChart.name : "Select a Player";

    const chartPlayerStats = useMemo(() => {
        let stats = allPlayerStats.filter(stat => stat.player_id === selectedPlayerIdForChart);
        
        if (selectedSeasonForChart !== 'all') {
            stats = stats.filter(stat => stat.game_date && new Date(stat.game_date).getFullYear().toString() === selectedSeasonForChart);
        }
        
        stats = stats.filter(stat => stat[selectedChartStat] != null);
        
        if (selectedOpponentTeamNameForH2H !== 'all') {
            stats = stats.filter(stat => {
                if (!stat.game || !stat.player.team_name) return false;
                const opponent = stat.game.home_team === stat.player.team_name ? stat.game.away_team : stat.game.home_team;
                return opponent === selectedOpponentTeamNameForH2H;
            });
        }
        
        if (selectedLocationFilter !== 'all' && selectedOpponentTeamNameForH2H === 'all') {
            stats = stats.filter(stat => {
                if (!stat.game || !stat.player.team_name) return false;
                const isHome = stat.game.home_team === stat.player.team_name;
                return selectedLocationFilter === 'home' ? isHome : !isHome;
            });
        }
        
        stats.sort((a, b) => new Date(b.game_date as string).getTime() - new Date(a.game_date as string).getTime());

        switch (selectedDateRangeFilter) {
            case 'last5': return stats.slice(0, 5).reverse();
            case 'last10': return stats.slice(0, 10).reverse();
            case 'last20': return stats.slice(0, 20).reverse();
            default: return stats.reverse();
        }
    }, [allPlayerStats, selectedPlayerIdForChart, selectedChartStat, selectedDateRangeFilter, selectedLocationFilter, selectedOpponentTeamNameForH2H, selectedSeasonForChart]);
    */

    const handlePredictionSelectionChange = (selectedRows: Prediction[]) => {
        const newStagedLegs = selectedRows.map(p => {
            const existing = stagedParlayLegs.find(leg => leg.prediction.id === p.id);
            return { prediction: p, type: existing ? existing.type : 'over' };
        });
        setStagedParlayLegs(newStagedLegs);
    };

    const handleParlayLegTypeChange = (predictionId: string, newType: 'over' | 'under') => {
        setStagedParlayLegs(legs => legs.map(leg => 
            leg.prediction.id === predictionId ? { ...leg, type: newType } : leg
        ));
    };

    const handleCreateParlayFromStaged = async () => {
        if (stagedParlayLegs.length === 0) {
            alert("Please select at least one prediction.");
            return;
        }

        const payload: ParlayData = {
            selections: stagedParlayLegs.map(leg => ({
                prediction_id: leg.prediction.id,
                player_prop_id: leg.prediction.player_prop?.id || '',
                player_name: leg.prediction.player_prop?.player?.player_name || 'N/A',
                market_key: leg.prediction.player_prop?.market?.key || 'N/A',
                game_id: leg.prediction.player_prop?.game_id || '',
                line_point: leg.prediction.player_prop?.outcomes?.[0]?.point ?? null,
                chosen_outcome: leg.type,
                chosen_probability: leg.type === 'over' 
                    ? leg.prediction.over_p_value_calibrated ?? leg.prediction.predicted_over_probability ?? 0
                    : leg.prediction.under_p_value_calibrated ?? leg.prediction.predicted_under_probability ?? 0,
            })),
            combined_probability: combinedParlayProbability,
        };

        try {
            const created = await postParlay(payload);
            alert(`Parlay created with ID: ${created.id}`);
            setStagedParlayLegs([]);
        } catch (error) {
            console.error("Failed to create parlay:", error);
            alert("Error creating parlay.");
        }
    };

    const handleDownloadPdf = () => {
        const doc = new jsPDF();
        doc.text("WNBA Player Predictions", 14, 16);

        const tableColumns = [
            { header: 'Player', dataKey: 'player' },
            { header: 'Game', dataKey: 'game' },
            { header: 'Market', dataKey: 'market' },
            { header: 'Line', dataKey: 'line' },
            { header: 'Over Prob.', dataKey: 'over_prob' },
            { header: 'Under Prob.', dataKey: 'under_prob' },
            { header: 'ICP Interval', dataKey: 'icp_interval' },
        ];

        const tableRows = predictions.map(p => {
            const game = p.player_prop?.game;
            return {
                player: p.player_prop?.player?.player_name || 'N/A',
                game: game ? `${game.away_team} @ ${game.home_team}` : 'N/A',
                market: p.player_prop?.market?.description || p.player_prop?.market?.key || 'N/A',
                line: p.player_prop?.outcomes?.[0]?.point?.toString() ?? 'N/A',
                over_prob: p.predicted_over_probability ? `${(p.predicted_over_probability * 100).toFixed(1)}%` : 'N/A',
                under_prob: p.predicted_under_probability ? `${(p.predicted_under_probability * 100).toFixed(1)}%` : 'N/A',
                icp_interval: (p.predicted_value_interval_lower != null && p.predicted_value_interval_upper != null) 
                    ? `[${p.predicted_value_interval_lower.toFixed(2)}, ${p.predicted_value_interval_upper.toFixed(2)}]`
                    : 'N/A',
            };
        });

        autoTable(doc, {
            head: [tableColumns.map(c => c.header)],
            body: tableRows.map(row => tableColumns.map(col => row[col.dataKey as keyof typeof row])),
            startY: 20,
            styles: { fontSize: 8 },
            headStyles: { fillColor: [22, 160, 133] },
        });

        doc.save('wnba_predictions.pdf');
    };

    const isLoading = loadingPredictions; // || loadingPlayerStats;

    return (
        <div>
            <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                <h1>WNBA Player Dashboard</h1>
                <span
                    ref={infoIconRef}
                    onMouseEnter={() => setShowModelInfoPopover(true)}
                    style={{ marginLeft: '15px', cursor: 'pointer', fontSize: '1.5em', position: 'relative' }}
                    title="View Model Version Info"
                >
                    ⓘ
                </span>
                {showModelInfoPopover && infoIconRef.current && (
                    <div
                        onMouseLeave={() => setShowModelInfoPopover(false)}
                        style={{
                            position: 'absolute',
                            top: `${infoIconRef.current.offsetTop + infoIconRef.current.offsetHeight + 5}px`,
                            left: `${infoIconRef.current.offsetLeft}px`,
                            backgroundColor: 'white',
                            border: '1px solid #ccc',
                            borderRadius: '5px',
                            padding: '0px',
                            boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
                            zIndex: 1000,
                            minWidth: '300px'
                        }}
                    >
                        <ModelVersionInfo />
                    </div>
                )}
            </div>
            <p>Displaying player stats and recommendations.</p>

            {isLoading && <p>Loading dashboard data...</p>}
            {error && <p style={{ color: 'red' }}>{error.split('\n').map((line, idx) => <span key={idx}>{line}<br/></span>)}</p>}

            {!isLoading && !error && (
                <>
                    <div style={{ width: '100%', marginBottom: '20px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                            <h2>Predictions</h2>
                            <button onClick={handleDownloadPdf} style={{ padding: '8px 12px', cursor: 'pointer', border: '1px solid #ccc', borderRadius: '4px' }}>
                                Download as PDF
                            </button>
                        </div>
                        <p style={{ fontSize: '0.9em', color: '#666', marginTop: '-5px', marginBottom: '10px' }}>
                            Select predictions from the table below to add them to the Parlay Builder.
                        </p>
                        <DataGrid predictions={predictions} onSelectionChanged={handlePredictionSelectionChange} />
                        
                        {stagedParlayLegs.length > 0 && (
                            <div style={{ marginTop: '15px', padding: '10px', border: '1px solid #eee', borderRadius: '5px' }}>
                                <h4>Parlay Builder</h4>
                                <ul>
                                    {stagedParlayLegs.map(leg => (
                                        <li key={leg.prediction.id} style={{ marginBottom: '10px', paddingBottom: '10px', borderBottom: '1px solid #eee'}}>
                                            <strong>{leg.prediction.player_prop?.player?.player_name}</strong> - {leg.prediction.player_prop?.market?.description}: {leg.prediction.player_prop?.outcomes?.[0]?.point}
                                            <br />
                                            Selected: {leg.type} (Prob: {(() => {
                                                const prob = leg.type === 'over' ? leg.prediction.over_p_value_calibrated ?? leg.prediction.predicted_over_probability : leg.prediction.under_p_value_calibrated ?? leg.prediction.predicted_under_probability;
                                                const isCalibrated = (leg.type === 'over' && leg.prediction.over_p_value_calibrated != null) || (leg.type === 'under' && leg.prediction.under_p_value_calibrated != null);
                                                if (prob == null) return 'N/A';
                                                return `${(prob * 100).toFixed(1)}%${isCalibrated ? ' (calib.)' : ''}`;
                                            })()})
                                            <div style={{ marginTop: '5px' }}>
                                                <label style={{ marginRight: '10px'}}><input type="radio" name={`parlay-${leg.prediction.id}`} value="over" checked={leg.type === 'over'} onChange={() => handleParlayLegTypeChange(leg.prediction.id, 'over')} /> Over</label>
                                                <label><input type="radio" name={`parlay-${leg.prediction.id}`} value="under" checked={leg.type === 'under'} onChange={() => handleParlayLegTypeChange(leg.prediction.id, 'under')} /> Under</label>
                                            </div>
                                        </li>
                                    ))}
                                </ul>
                                {combinedParlayProbability != null && (
                                    <p style={{ fontWeight: 'bold' }}>Combined Probability: {(combinedParlayProbability * 100).toFixed(2)}%</p>
                                )}
                                <button onClick={handleCreateParlayFromStaged}>Create Parlay</button>
                            </div>
                        )}
                    </div>
                    
                    {/*
                    <div style={{ width: '100%' }}>
                        <h2>Performance Charts</h2>
                        {loadingPlayerStats ? <p>Loading chart data...</p> : (
                        <>
                            {allPlayerStats.length > 0 ? (
                            <>
                            <div style={{ marginBottom: '10px', display: 'flex', alignItems: 'center', flexWrap: 'wrap' }}>
                                <div style={{ marginRight: '20px', marginBottom: '10px' }}>
                                    <label>Player: <input type="text" placeholder="Search..." value={playerSearchTerm} onChange={e => setPlayerSearchTerm(e.target.value)} /></label>
                                    <select value={selectedPlayerIdForChart || ''} onChange={e => setSelectedPlayerIdForChart(e.target.value || null)}>
                                        <option value="" disabled>{playerSearchTerm ? 'No matches' : 'Select'}</option>
                                        {filteredUniquePlayersForChart.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                                    </select>
                                </div>
                                <div style={{ marginRight: '20px', marginBottom: '10px' }}>
                                    <label>Stat: <select value={selectedChartStat} onChange={e => setSelectedChartStat(e.target.value as keyof PlayerStatFull)}>
                                        {availableStats.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
                                    </select></label>
                                </div>
                                <div style={{ marginRight: '20px', marginBottom: '10px' }}>
                                    <label>Season: <select value={selectedSeasonForChart} onChange={e => setSelectedSeasonForChart(e.target.value)}>
                                        {uniqueSeasonsForChart.map(s => <option key={s} value={s}>{s === 'all' ? 'All' : s}</option>)}
                                    </select></label>
                                </div>
                                <div style={{ marginRight: '20px', marginBottom: '10px' }}>
                                    <label>Range: <select value={selectedDateRangeFilter} onChange={e => setSelectedDateRangeFilter(e.target.value)}>
                                        {dateRangeFilterOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                                    </select></label>
                                </div>
                                <div style={{ marginRight: '20px', marginBottom: '10px' }}>
                                    <label>Location: <select value={selectedLocationFilter} onChange={e => setSelectedLocationFilter(e.target.value)}>
                                        {locationFilterOptions.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
                                    </select></label>
                                </div>
                                <div style={{ marginBottom: '10px' }}>
                                    <label>Vs: <select value={selectedOpponentTeamNameForH2H} onChange={e => setSelectedOpponentTeamNameForH2H(e.target.value)}>
                                        <option value="all">All</option>
                                        {uniqueOpponentTeamNames.map(t => <option key={t} value={t}>{t}</option>)}
                                    </select></label>
                                </div>
                            </div>
                            <PlayerPerformanceChart playerName={chartPlayerName} playerStats={chartPlayerStats} statToDisplay={selectedChartStat} />
                            </>
                            ) : <p>No historical data for charts.</p>}
                        </>
                        )}
                    </div>
                    
                    <div style={{ width: '100%', marginBottom: '20px' }}>
                        <PlayerStatsTable />
                    </div>
                    */}
                </>
            )}
        </div>
    );
};

export default Dashboard;