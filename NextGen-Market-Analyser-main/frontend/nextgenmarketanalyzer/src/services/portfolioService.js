const API_BASE_URL = 'http://localhost:8001';

export const getClientList = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/clients`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch client list');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching client list:', error);
    throw error;
  }
};

export const analyzePortfolioById = async (clientId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze-portfolio/${clientId}`);
    
    
    if (!response.ok) {
      throw new Error(`Failed to analyze portfolio for client ${clientId}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Error analyzing portfolio for client ${clientId}:`, error);
    throw error;
  }
};

export const getAISummary = async (clientId) => {
  try {
    const response = await fetch(`${API_BASE_URL}/ai-summary/${clientId}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get AI summary for client ${clientId}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Error getting AI summary for client ${clientId}:`, error);
    throw error;
  }
};