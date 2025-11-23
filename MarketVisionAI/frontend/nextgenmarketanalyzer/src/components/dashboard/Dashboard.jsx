import React, { useState, useEffect } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import StockEvaluator from "./StockEvaluator";
import "./Dashboard.css";
import { useNavigate } from "react-router-dom";
import PortfolioAnalyzer from "./Portfolio/PortfolioAnalyzer";
const StockAnalysis = () => (
  <div className="analysis-section">
    <h2 className="section-title">Stock Analysis</h2>
    <p className="section-subtitle">
      Analyze real-time stock performance, trends, and insights to make informed
      decisions.
    </p>
    {/* Cards */}
    <div className="row g-4 mt-4">
      <div className="col-md-4">
        <div className="info-card">
          <h5>📈 Top Gainers</h5>
          <p>Explore the best performing stocks of the day.</p>
        </div>
      </div>
      <div className="col-md-4">
        <div className="info-card">
          <h5>📉 Top Losers</h5>
          <p>Monitor underperforming stocks to manage risk.</p>
        </div>
      </div>
      <div className="col-md-4">
        <div className="info-card">
          <h5>📊 Market Trends</h5>
          <p>Get insights into overall market movement.</p>
        </div>
      </div>
    </div>
  </div>
);

const PortfolioAnalysis = () => (
  <div className="analysis-section">
    <h2 className="section-title">💼 Portfolio Analysis</h2>
    <p className="section-subtitle">
      Track your investments and optimize your portfolio for maximum returns.
    </p>
    <div className="row g-4 mt-4">
      <div className="col-md-6">
        <div className="info-card">
          <h5>📊 Portfolio Performance</h5>
          <p>View growth, risk, and returns across your investments.</p>
        </div>
      </div>
      <div className="col-md-6">
        <div className="info-card">
          <h5>🔍 Allocation Insights</h5>
          <p>See how your investments are distributed across sectors.</p>
        </div>
      </div>
    </div>
  </div>
);

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState("stock");
  const [menuOpen, setMenuOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
   const navigate = useNavigate();

  // Check localStorage for authentication status
  useEffect(() => {
    const authStatus = localStorage.getItem("isAuthenticated");
    setIsLoggedIn(authStatus === "true");
  }, []);

   const handleLogout = () => {
     // ✅ Clear everything related to auth
     localStorage.removeItem("token");
     localStorage.removeItem("user");
     localStorage.removeItem("isAuthenticated");

     setIsLoggedIn(false);

     // ✅ Redirect to login or home page
     navigate("/"); // or navigate("/") if you want to go home
   };

  return (
    <div className="dashboard-container">
      {/* Top Navbar */}
      <nav className="navbar shadow-sm fixed-top">
        <div className="container-fluid d-flex justify-content-between align-items-center">
          <a className="navbar-brand fw-bold text-white" href="#">
            💲
          </a>

          <button
            className="hamburger-btn d-lg-none"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            ☰
          </button>

          {/* Navbar links */}
          <div
            className={`nav-links d-lg-flex flex-column flex-lg-row ${
              menuOpen ? "show" : ""
            }`}
          >
            <button
              className={`btn nav-btn ${
                activeTab === "stock" ? "active-btn" : ""
              }`}
              onClick={() => {
                setActiveTab("stock");
                setMenuOpen(false);
              }}
            >
              📈 Stock Analysis
            </button>
            <button
              className={`btn nav-btn ${
                activeTab === "portfolio" ? "active-btn" : ""
              }`}
              onClick={() => {
                setActiveTab("portfolio");
                setMenuOpen(false);
              }}
            >
              💼 Portfolio Analysis
            </button>

            <div className="auth-buttons d-lg-flex mt-3 mt-lg-0 ms-lg-3">
              {!isLoggedIn && (
                <>
                  <button className="btn btn-outline-light me-2">Login</button>
                  <button className="btn btn-light">Sign Up</button>
                </>
              )}
              {isLoggedIn && (
                <button className="btn btn-danger" onClick={handleLogout}>
                  Logout
                </button>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="main-content flex-grow-1 p-4 mt-5">
        {activeTab === "stock" && <StockEvaluator />}
        {activeTab === "portfolio" && <PortfolioAnalyzer />}
      </div>
    </div>
  );
};

export default Dashboard;
