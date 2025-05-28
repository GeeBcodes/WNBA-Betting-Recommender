import React, { useEffect, useState } from 'react';
import { getModelVersions, ModelVersion } from '../services/api';

const ModelVersionInfo: React.FC = () => {
  const [latestModelVersion, setLatestModelVersion] = useState<ModelVersion | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchModelVersions = async () => {
      try {
        setLoading(true);
        const versions = await getModelVersions();
        if (versions && versions.length > 0) {
          // Assuming the first one is the latest, or sort by trained_at if available and needed
          // For now, let's just pick the first one returned by the API
          setLatestModelVersion(versions[0]);
        } else {
          setLatestModelVersion(null); // No versions found
        }
        setError(null);
      } catch (err) {
        console.error("Failed to fetch model versions:", err);
        setError('Failed to load model version information.');
        setLatestModelVersion(null);
      } finally {
        setLoading(false);
      }
    };

    fetchModelVersions();
  }, []);

  const textStyle = { color: '#333' }; // Define a common dark text style

  if (loading) {
    return <p style={textStyle}>Loading model version information...</p>;
  }

  if (error) {
    return <p style={{ ...textStyle, color: 'red' }}>{error}</p>; // Keep error red, but ensure base style if red is not contrasting enough
  }

  if (!latestModelVersion) {
    return <p style={textStyle}>No model version information available.</p>;
  }

  return (
    <div style={{ border: '1px solid #ccc', padding: '15px', marginBottom: '20px', borderRadius: '5px' }}>
      <h3 style={textStyle}>Latest Model Version</h3>
      <p style={textStyle}><strong>Version Name:</strong> {latestModelVersion.version_name}</p>
      <p style={textStyle}><strong>Description:</strong> {latestModelVersion.description || 'N/A'}</p>
      <p style={textStyle}><strong>Trained At:</strong> {new Date(latestModelVersion.trained_at).toLocaleString()}</p>
    </div>
  );
};

export default ModelVersionInfo; 