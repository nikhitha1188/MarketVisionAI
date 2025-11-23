import React, { useState, useEffect } from "react";
import { Pie } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import ChartDataLabels from "chartjs-plugin-datalabels";
import {
  getClientList,
  analyzePortfolioById,
  getAISummary,
} from "../../../services/portfolioService";
import "./PortfolioAnalyzer.css";

ChartJS.register(ArcElement, Tooltip, Legend, ChartDataLabels);

// ✅ Blur overlay component
const BlurOverlay = ({
  isVisible,
  message,
  title = "AI Analysis in Progress",
}) => {
  if (!isVisible) return null;
  return (
    <div className="blur-overlay">
      <div className="blur-overlay-content">
        <h2>{title}</h2>
        <p>{message}</p>
      </div>
    </div>
  );
};

const PortfolioAnalyzer = () => {
  const [clients, setClients] = useState([]);
  const [selectedClient, setSelectedClient] = useState("");
  const [portfolioData, setPortfolioData] = useState(null);
  const [aiSummary, setAiSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [aiLoading, setAiLoading] = useState(false);
  const [error, setError] = useState(null);
  const [overlayMessage, setOverlayMessage] = useState("");

  const loadingMessages = {
    portfolio: "Crunching your portfolio data...",
    ai: "AI is analyzing your investment strategy...",
    dropdown: "Fetching insights for your selected portfolio...",
  };

  // 🟢 Fetch clients list
  useEffect(() => {
    const fetchClients = async () => {
      try {
        setLoading(true);
        setOverlayMessage(loadingMessages.portfolio);
        const clientList = await getClientList();
        setClients(clientList);
        if (clientList?.length > 0) {
          setSelectedClient(clientList[0].clientId);
        }
        setError(null);
      } catch (err) {
        console.error(err);
        setError("Failed to load client list. Please try again later.");
        setClients([]);
      } finally {
        setLoading(false);
      }
    };
    fetchClients();
  }, []);

  // 🟢 Fetch portfolio + AI summary
  useEffect(() => {
    const fetchPortfolioAnalysis = async () => {
      if (!selectedClient) return;
      setLoading(true);
      setPortfolioData(null);
      setAiSummary(null);

      try {
        const result = await analyzePortfolioById(selectedClient);
        setPortfolioData(result);
        console.log(result.json);
        /* print(result); */
        setError(null);

        // 🔎 Start AI Analysis
        setAiLoading(true);
        
        try {
          const aiResult = await getAISummary(selectedClient);
          setAiSummary(aiResult);
          console.log("AI Summary:", aiResult);
        } catch (aiError) {
          console.error("AI summary error:", aiError);
        } finally {
          setAiLoading(false);
        }
      } catch (error) {
        console.error(error);
        setError(`Failed to analyze portfolio for client ${selectedClient}`);
        setPortfolioData(null);
        setAiSummary(null);
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolioAnalysis();
  }, [selectedClient]);

  // 📊 Sector Diversification Chart
  const renderSectorChart = () => {
    if (!portfolioData?.sectorDiversification) return null;

    const data = {
      labels: Object.keys(portfolioData.sectorDiversification),
      datasets: [
        {
          data: Object.values(portfolioData.sectorDiversification),
          backgroundColor: [
            "#FF6384",
            "#36A2EB",
            "#FFCE56",
            "#4CAF50",
            "#9966FF",
            "#FF9F40",
            "#EA526F",
            "#23C9FF",
            "#7CB342",
            "#FFD54F",
          ],
        },
      ],
    };

    return (
      <div className="sector-chart">
        <h3>Sector Diversification</h3>
        <Pie
          data={data}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              tooltip: {
                callbacks: {
                  label: (context) => {
                    const total = context.dataset.data.reduce(
                      (a, b) => a + b,
                      0
                    );
                    const value = context.raw;
                    const pct = ((value / total) * 100).toFixed(1) + "%";
                    return `${context.label}: ${pct}`;
                  },
                },
              },
              legend: {
                position: "right",
                labels: { boxWidth: 15, padding: 10 },
              },
              datalabels: {
                color: "#fff",
                font: { weight: "bold" },
                formatter: (value, context) => {
                  const total = context.chart.data.datasets[0].data.reduce(
                    (a, b) => a + b,
                    0
                  );
                  const pct = ((value / total) * 100).toFixed(1) + "%";
                  return pct;
                },
              },
            },
          }}
          width={300}
          height={300}
        />
        
      </div>
    );
  };

  // 📈 Fund Overlap
  const renderFundOverlap = () => {
    if (!portfolioData?.pairwiseOverlaps) return null;

    const maxOverlap = Math.max(
      ...Object.values(portfolioData.pairwiseOverlaps)
    );

    return (
      <div className="fund-overlap">
        {portfolioData.averageOverlap > 40 && (
          <div className="warning-message mb-3">
            <span className="warning-icon">!</span> High Overlap (
            {portfolioData.averageOverlap}%)
          </div>
        )}
        <h3>Fund Overlap Analysis</h3>
        <div className="overlap-items-container">
          {Object.entries(portfolioData.pairwiseOverlaps).map(
            ([pair, overlap]) => (
              <div key={pair} className="overlap-item">
                <div className="overlap-bar-container">
                  <div
                    className="overlap-bar"
                    style={{
                      height: `${(overlap / maxOverlap) * 100}%`,
                      backgroundColor: "#4CAF50",
                    }}
                  >
                    <span className="overlap-bar-label">{overlap}%</span>
                  </div>
                </div>
                <div className="overlap-label">{pair}</div>
              </div>
            )
          )}
        </div>
        <div className="overlap-summary mt-2">
          <span className="">
            <span className="avg_text">Average Overlap:</span>{" "}
            <span className="score-value">{portfolioData.averageOverlap}%</span>
          </span>
          <span>
            <span className="avg_text">Overlap Score: </span>
            <span className="score-value">{portfolioData.overlapScore}</span>
          </span>
        </div>
      </div>
    );
  };

  // 🤖 AI Summary
  const renderAISummary = () => {
    if (!aiSummary) return null;

    return (
      <>
      
        <div className="ai-summary">
          {aiSummary && (
            <div className="fund1-overlap p-1 rounded-4 shadow-sm">
              <div className="summary-value mb-2 ms-2">
                <span className="trader">Trader Type: </span>
                <span className="">{aiSummary.traderType}</span>
              </div>
            </div>
          )}

          <h3 className="mt-5">Recommendations</h3>
          <ul>
            {aiSummary.possibleDiversification?.map((item, i) => (
              <li key={i}>
                <strong>{item.sector}:</strong> {item.recommendation}
              </li>
            ))}
          </ul>
        </div>
      </>
    );
  };

  const renderPortfolioSummary = () => {
    if (!portfolioData) return null;
    return (
      <div className="fund1-overlap p-2 rounded-4 shadow-sm mb-4">
        <div className="summary-value mb-2 ms-2">
          Portfolio Value: {portfolioData.currency}{" "}
          <span className="score-value">
            {portfolioData.totalValue?.toLocaleString()}
          </span>
          <br></br>
          Final Diversification Score:{" "}
          <span className="score-value">
            {" "}
            {portfolioData.finalDiversificationScore.toFixed(2)}
          </span>
          <br />
          <span className="sector-score">Sector Score:</span>
          <span className="score-value"> {portfolioData.sectorScore}</span>
        </div>
      </div>
    );
  };


  if (loading && !portfolioData) {
    return <div className="loading-state">Loading portfolio analysis...</div>;
  }
  if (error && !portfolioData) {
    return <div className="error-state">{error}</div>;
  }

  return (
    <div className="portfolio-analyzer">
      {/* 🔄 Blur overlay */}

      <div className="portfolio-header d-flex justify-content-between align-items-center">
        <select
          className="client-selector"
          value={selectedClient}
          onChange={(e) => setSelectedClient(e.target.value)}
          disabled={loading || clients.length === 0}
        >
          {clients.map((client) => (
            <option key={client.clientId} value={client.clientId}>
              Client {client.clientId} ({client.currency})
            </option>
          ))}
        </select>
      </div>

      <div className="dashboard-layout">
        {/* Left Block */}
        <div className="left-panel">
          {renderPortfolioSummary()}

          {renderSectorChart()}
        </div>

        {/* Middle Block */}
        <div className="middle-panel">{renderFundOverlap()}</div>

        {/* Right Block */}
        <div className="right-panel">
          {aiLoading ? (
            <div className="ai-summary">
              <div className="blur-overlay-spinner"></div>

              <p className="blur-overlay-message glowing-text">
                Generating insights...
              </p>
            </div>
          ) : (
            renderAISummary()
          )}
        </div>
      </div>
    </div>
  );
};

export default PortfolioAnalyzer;
